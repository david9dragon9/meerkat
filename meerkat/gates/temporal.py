from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.gates.base import Gate


class TemporalCountGate(Gate):
    """Fires when another gate fires enough times inside a sliding window."""

    def __init__(
        self,
        gate_id: str,
        upstream_gate_id: Optional[str] = None,
        required_count: int = 1,
        window_ms: int = 1000,
        upstream_gate_ids: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(gate_id, [EventType.GATE_FIRE])
        self.upstream_gate_id = upstream_gate_id
        self.upstream_gate_ids = {str(item) for item in (upstream_gate_ids or []) if str(item)}
        if upstream_gate_id:
            self.upstream_gate_ids.add(str(upstream_gate_id))
        self.required_count = required_count
        self.window_ms = window_ms
        self._fires: Deque[tuple[int, GateFire]] = deque()

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        fire = event.payload.get("gate_fire")
        if not isinstance(fire, GateFire) or fire.gate_id not in self.upstream_gate_ids:
            return None

        current_time_ms = event.stream_time_ms
        self._fires.append((current_time_ms, fire))
        while self._fires and current_time_ms - self._fires[0][0] > self.window_ms:
            self._fires.popleft()

        if len(self._fires) < self.required_count:
            return None

        fires = list(self._fires)
        self._fires.clear()
        latest_fire = fires[-1][1]
        upstream_ids = sorted({item.gate_id for _, item in fires})
        return GateFire(
            gate_id=self.gate_id,
            confidence=max(item.confidence for _, item in fires),
            reason=f"{', '.join(upstream_ids)} fired {self.required_count} times within {self.window_ms}ms",
            evidence={
                "upstream_gate_id": latest_fire.gate_id,
                "upstream_gate_ids": upstream_ids,
                "stream_time_ms": current_time_ms,
                "frame": latest_fire.evidence.get("frame"),
                "model_confirmed": any(item.evidence.get("model_confirmed") for _, item in fires),
                "upstream_evidence": latest_fire.evidence,
                "counted_evidence": {item.gate_id: item.evidence for _, item in fires},
            },
        )
