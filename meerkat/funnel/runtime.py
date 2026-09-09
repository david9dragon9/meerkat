from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from meerkat.benchmarks.metrics import Metrics
from meerkat.events import EventType, GateFire, StreamEvent, now_ms
from meerkat.funnel.factory import build_gate
from meerkat.funnel.gate_types import MODEL_BACKED_GATE_TYPES
from meerkat.funnel.spec import FunnelSpec, StateUpdateSpec
from meerkat.gates.base import Gate
from meerkat.models.responders import Responder
from meerkat.runtime.event_bus import EventBus
from meerkat.runtime.logging import RuntimeLogger
from meerkat.runtime.state import RuntimeState


@dataclass
class RuntimeHandle:
    runtime: "FunnelRuntime"

    async def stop(self) -> None:
        await self.runtime.stop()


class FunnelRuntime:
    def __init__(
        self,
        spec: FunnelSpec,
        bus: EventBus,
        responder: Responder,
        metrics: Metrics,
        logger: Optional[RuntimeLogger] = None,
        initial_state: Optional[dict[str, Any]] = None,
    ) -> None:
        self.spec = spec
        self.bus = bus
        self.responder = responder
        self.metrics = metrics
        self.logger = logger
        self.state = RuntimeState(initial_state)
        self._last_response_wall_ms = -int(spec.response.cooldown_seconds * 1000)
        self._last_response_signature: Optional[str] = None
        #: When each distinct message was last delivered, so the cooldown can
        #: throttle repetition without ever withholding new information.
        self._recent_signatures: Dict[str, int] = {}
        self._response_count = 0
        self._tasks: List[asyncio.Task] = []
        self._subscriptions: List[asyncio.Queue[StreamEvent]] = []
        self._running = False
        self._active_events = 0
        self._configure_spec(spec)
        if self.logger and hasattr(self.responder, "set_logger"):
            self.responder.set_logger(self.logger)

    def _configure_spec(self, spec: FunnelSpec) -> None:
        self.spec = spec
        self.gates: List[Gate] = [build_gate(gate_spec) for gate_spec in spec.gates]
        self._gate_state_updates = {
            gate_spec.id: _parse_state_updates(gate_spec.params.get("state_updates", []))
            for gate_spec in spec.gates
        }
        self._requires_model_confirmation = any(
            gate_spec.type in MODEL_BACKED_GATE_TYPES for gate_spec in spec.gates
        )
        downstream_gate_ids = {
            str(gate_spec.params["upstream_gate_id"])
            for gate_spec in self.spec.gates
            if gate_spec.params.get("upstream_gate_id")
        }
        for gate_spec in self.spec.gates:
            raw_gate_ids = gate_spec.params.get("gate_ids")
            if isinstance(raw_gate_ids, list):
                downstream_gate_ids.update(str(gate_id) for gate_id in raw_gate_ids)
        self._terminal_gate_ids = {
            gate.gate_id
            for gate in self.gates
            if gate.terminal_response and gate.gate_id not in downstream_gate_ids
        }
        if not self._terminal_gate_ids and self.gates:
            self._terminal_gate_ids = {self.gates[-1].gate_id}
        if self.logger:
            for gate in self.gates:
                gate.set_logger(self.logger)

    async def warmup(self) -> None:
        await asyncio.gather(*(gate.warmup() for gate in self.gates))

    def start(self) -> RuntimeHandle:
        self._start_tasks()
        self._running = True
        return RuntimeHandle(runtime=self)

    async def replace_spec(self, spec: FunnelSpec) -> None:
        await self._stop_tasks()
        self._configure_spec(spec)
        self._last_response_wall_ms = -int(spec.response.cooldown_seconds * 1000)
        self._last_response_signature = None
        self._recent_signatures.clear()
        self._response_count = 0
        await self.warmup()
        if self._running:
            self._start_tasks()

    async def stop(self) -> None:
        self._running = False
        await self._stop_tasks()

    async def drain(self, quiet_seconds: float = 0.5, timeout_seconds: float = 30.0) -> bool:
        """Wait for queued and currently-running gate/responder work to settle."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        quiet_started: Optional[float] = None
        while loop.time() < deadline:
            pending = self._active_events + sum(queue.qsize() for queue in self._subscriptions)
            if pending == 0:
                quiet_started = quiet_started or loop.time()
                if loop.time() - quiet_started >= quiet_seconds:
                    return True
            else:
                quiet_started = None
            await asyncio.sleep(0.05)
        return False

    async def _stop_tasks(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        for queue in self._subscriptions:
            self.bus.unsubscribe(queue)
        self._subscriptions = []

    def _start_tasks(self) -> None:
        tasks = []
        for gate in self.gates:
            queue = self.bus.subscribe(gate.input_types, queue_size=gate.queue_size)
            self._subscriptions.append(queue)
            gate.set_runtime_queue(queue)
            for worker_index in range(gate.worker_count):
                tasks.append(
                    asyncio.create_task(
                        self._run_gate(gate, queue),
                        name=f"gate:{gate.gate_id}:{worker_index}",
                    )
                )
        response_queue = self.bus.subscribe([EventType.GATE_FIRE])
        self._subscriptions.append(response_queue)
        tasks.append(asyncio.create_task(self._run_responder(response_queue), name="funnel:responder"))
        self._tasks = tasks

    async def _run_gate(self, gate: Gate, queue: asyncio.Queue[StreamEvent]) -> None:
        while True:
            event = await queue.get()
            self._active_events += 1
            if self._max_responses_reached() and gate.gate_id in self._terminal_gate_ids:
                self.metrics.inc(f"gate.{gate.gate_id}.max_response_skips")
                self._active_events -= 1
                continue
            try:
                event.payload["state"] = await self.state.snapshot()
                start = self.metrics.start_timer()
                try:
                    fire = await gate.process(event)
                except Exception as exc:
                    self.metrics.inc(f"gate.{gate.gate_id}.errors")
                    if self.logger:
                        self.logger.log(event.stream_time_ms, f"Gate error gate={gate.gate_id} error={exc}")
                    continue
                finally:
                    self.metrics.stop_timer(f"gate.{gate.gate_id}.latency", start)
                self.metrics.inc(f"gate.{gate.gate_id}.events")
                if fire is not None:
                    upstream_fire = event.payload.get("gate_fire")
                    if (
                        isinstance(upstream_fire, GateFire)
                        and upstream_fire.evidence.get("model_confirmed")
                        and "model_confirmed" not in fire.evidence
                    ):
                        fire.evidence["model_confirmed"] = True
                        fire.evidence["upstream_evidence"] = upstream_fire.evidence
                    gate_state_updates = self._gate_state_updates.get(gate.gate_id, [])
                    if gate_state_updates:
                        state_updates = await self.state.apply(gate_state_updates, evidence=fire.evidence)
                        state_snapshot = await self.state.snapshot()
                        fire.evidence["state_updates"] = state_updates
                        fire.evidence["state"] = state_snapshot
                        if self.logger:
                            self.logger.log(
                                event.stream_time_ms,
                                f"State updated trigger={fire.gate_id} updates={state_updates}",
                            )
                    self.metrics.inc(f"gate.{gate.gate_id}.fires")
                    fire_stream_time_ms = _fire_stream_time_ms(fire, event.stream_time_ms)
                    if self.logger:
                        self.logger.log(
                            fire_stream_time_ms,
                            f"Gate fired gate={fire.gate_id} confidence={fire.confidence:.2f} reason={fire.reason}",
                        )
                    await self.bus.publish(
                        StreamEvent(
                            type=EventType.GATE_FIRE,
                            source_id=event.source_id,
                            stream_time_ms=fire_stream_time_ms,
                            sequence_id=event.sequence_id,
                            payload={"gate_fire": fire},
                        )
                    )
            finally:
                self._active_events -= 1

    async def _run_responder(self, queue: asyncio.Queue[StreamEvent]) -> None:
        while True:
            event = await queue.get()
            self._active_events += 1
            try:
                fire = event.payload.get("gate_fire")
                if not isinstance(fire, GateFire):
                    continue
                if self._terminal_gate_ids and fire.gate_id not in self._terminal_gate_ids:
                    continue
                if self._requires_model_confirmation and not fire.evidence.get("model_confirmed"):
                    self.metrics.inc("response.model_confirmation_skips")
                    if self.logger:
                        self.logger.log(event.stream_time_ms, f"Response skipped; missing model confirmation trigger={fire.gate_id}")
                    continue
                if self._max_responses_reached():
                    self.metrics.inc("response.max_response_skips")
                    if self.logger:
                        self.logger.log(event.stream_time_ms, f"Response skipped by max_responses trigger={fire.gate_id}")
                    continue
                # A funnel whose message is a fixed string can only ever repeat
                # itself, so the cooldown can be applied before doing any work.
                # When the message interpolates evidence or state, the only way
                # to know whether there is news is to produce it first.
                if not self._response_text_can_vary() and not self._cooldown_elapsed():
                    self.metrics.inc("response.cooldown_skips")
                    if self.logger:
                        self.logger.log(event.stream_time_ms, f"Response skipped by cooldown trigger={fire.gate_id}")
                    continue
                if self.spec.response.state_updates:
                    state_updates = await self.state.apply(self.spec.response.state_updates, evidence=fire.evidence)
                    if not state_updates:
                        self.metrics.inc("response.state_duplicate_skips")
                        if self.logger:
                            self.logger.log(
                                event.stream_time_ms,
                                f"Response skipped by state dedupe trigger={fire.gate_id}",
                            )
                        continue
                    state_snapshot = await self.state.snapshot()
                    fire.evidence["state_updates"] = state_updates
                    fire.evidence["state"] = state_snapshot
                    if self.logger:
                        self.logger.log(
                            event.stream_time_ms,
                            f"State updated trigger={fire.gate_id} updates={state_updates}",
                        )
                elif "state" not in fire.evidence:
                    fire.evidence["state"] = await self.state.snapshot()
                start = self.metrics.start_timer()
                response = await self.responder.respond(self.spec, fire)
                self.metrics.stop_timer("response.latency", start)
                response_signature = _response_signature(response.text)
                if response_signature == self._last_response_signature:
                    self.metrics.inc("response.duplicate_state_skips")
                    if self.logger:
                        self.logger.log(
                            event.stream_time_ms,
                            f"Response skipped by duplicate state trigger={fire.gate_id}",
                        )
                    continue
                if self._said_recently(response_signature):
                    self.metrics.inc("response.cooldown_skips")
                    if self.logger:
                        self.logger.log(
                            event.stream_time_ms,
                            f"Response skipped by cooldown trigger={fire.gate_id}",
                        )
                    continue
                self.metrics.inc("response.count")
                self._response_count += 1
                self._last_response_wall_ms = now_ms()
                self._last_response_signature = response_signature
                self._remember_signature(response_signature)
                if self.logger:
                    self.logger.log(
                        event.stream_time_ms,
                        f"USER RESPONSE trigger={response.trigger_gate_id}: {response.text}",
                        bold=True,
                    )
                await self.bus.publish(
                    StreamEvent(
                        type=EventType.MODEL_RESPONSE,
                        source_id=event.source_id,
                        stream_time_ms=event.stream_time_ms,
                        sequence_id=event.sequence_id,
                        payload={"model_response": response},
                    )
                )
            finally:
                self._active_events -= 1

    def _cooldown_elapsed(self) -> bool:
        cooldown_ms = int(self.spec.response.cooldown_seconds * 1000)
        return now_ms() - self._last_response_wall_ms >= cooldown_ms

    def _response_text_can_vary(self) -> bool:
        """Whether two firings of this funnel could say different things.

        A fixed `on_match_text` always produces the same sentence. Anything
        that interpolates evidence or state, or that a model writes, can carry
        a new value each time.
        """
        template = self.spec.response.on_match_text
        return not template or "{" in template

    def _said_recently(self, signature: str) -> bool:
        """Whether this exact message was already delivered inside the cooldown.

        The cooldown throttles repetition, not novelty: a message the user has
        not just been told is always worth delivering, which is what makes
        "tell me the current value" report again as soon as the value changes.
        """
        cooldown_ms = int(self.spec.response.cooldown_seconds * 1000)
        if cooldown_ms <= 0:
            return False
        said_at = self._recent_signatures.get(signature)
        return said_at is not None and now_ms() - said_at < cooldown_ms

    def _remember_signature(self, signature: str) -> None:
        cooldown_ms = int(self.spec.response.cooldown_seconds * 1000)
        now = now_ms()
        self._recent_signatures[signature] = now
        self._recent_signatures = {
            seen: at for seen, at in self._recent_signatures.items() if now - at < max(cooldown_ms, 1)
        }

    def _max_responses_reached(self) -> bool:
        max_responses = self.spec.response.max_responses
        return max_responses is not None and self._response_count >= max_responses


def _parse_state_updates(raw_updates: object) -> List[StateUpdateSpec]:
    if not isinstance(raw_updates, list):
        return []
    updates: List[StateUpdateSpec] = []
    for raw_update in raw_updates:
        if isinstance(raw_update, StateUpdateSpec):
            updates.append(raw_update)
        elif isinstance(raw_update, dict) and raw_update.get("key"):
            updates.append(
                StateUpdateSpec(
                    key=str(raw_update["key"]),
                    operation=str(raw_update.get("operation", "increment")),
                    value=raw_update.get("value", 1),
                )
            )
    return updates


def _response_signature(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.!?]+$", "", normalized)
    return normalized


def _fire_stream_time_ms(fire: GateFire, fallback_ms: int) -> int:
    value = fire.evidence.get("stream_time_ms")
    if isinstance(value, (int, float)):
        return int(value)
    return fallback_ms
