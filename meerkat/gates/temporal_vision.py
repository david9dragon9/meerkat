from __future__ import annotations

import asyncio
import base64
import json
from collections import defaultdict, deque
from time import perf_counter
from typing import Any, Deque, Dict, Iterable, Optional, Sequence

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.gates.base import Gate
from meerkat.models.config import ModelConfig
from meerkat.models.provider import ImagePart, ModelBacked, ModelRequest, TextPart


class ModelFrameChangeGate(ModelBacked, Gate):
    """Model vision gate that verifies a temporal change between frames."""

    def __init__(
        self,
        gate_id: str,
        query: str,
        sample_interval_ms: int = 500,
        min_confidence: float = 0.75,
        model: Optional[str] = None,
        baseline_mode: str = "initial",
        service_tier: str = "default",
        state_interval_ms: int = 3000,
        concurrent_requests: int = 1,
        min_request_interval_ms: int = 200,
        queued_frame_dedupe_window_ms: int = 500,
        required_count: int = 1,
        confirmation_window_ms: int = 1500,
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.worker_count = max(1, concurrent_requests)
        self.queue_size = max(1, self.worker_count * 4)
        self.terminal_response = True
        self.query = query
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.model = model or ModelConfig.from_env().cheap_model
        self.baseline_mode = baseline_mode
        self.service_tier = "default" if service_tier == "standard" else service_tier
        self.state_interval_ms = state_interval_ms
        self.min_request_interval_ms = min_request_interval_ms
        self.queued_frame_dedupe_window_ms = queued_frame_dedupe_window_ms
        self.required_count = max(1, required_count)
        self.confirmation_window_ms = confirmation_window_ms
        self._last_sample_time_ms = -sample_interval_ms
        self._baseline_frame = None
        self._baseline_time_ms: Optional[int] = None
        self._previous_frame = None
        self._previous_time_ms: Optional[int] = None
        self._initial_state = ""
        self._last_state_time_ms = -state_interval_ms
        self._state_task: Optional[asyncio.Task[None]] = None
        self._sent_request_stream_times_ms: Deque[int] = deque(maxlen=128)
        self._matched_request_stream_times_ms: Deque[int] = deque(maxlen=16)
        self._sample_lock = asyncio.Lock()
        self._request_spacing_lock = asyncio.Lock()
        self._next_request_wall_time = 0.0

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        frame = event.payload.get("frame")
        if frame is None:
            return None
        async with self._sample_lock:
            if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
                return None
            self._last_sample_time_ms = event.stream_time_ms
            if self._baseline_frame is None:
                self._baseline_frame = frame
                self._baseline_time_ms = event.stream_time_ms
                self._previous_frame = frame
                self._previous_time_ms = event.stream_time_ms
                self._schedule_initial_state_update(frame, event.stream_time_ms)
                return None
            reference_frame = self._baseline_frame if self.baseline_mode == "initial" else self._previous_frame
            reference_time_ms = self._baseline_time_ms if self.baseline_mode == "initial" else self._previous_time_ms
        if reference_frame is None or reference_time_ms is None:
            return None
        if await self._should_skip_due_to_nearby_queued_request(event.stream_time_ms):
            return None

        reference_task = asyncio.create_task(asyncio.to_thread(self._encode_frame_data_url, reference_frame))
        current_task = asyncio.create_task(asyncio.to_thread(self._encode_frame_data_url, frame))
        await self._wait_for_request_slot()
        reference_url, current_url = await asyncio.gather(reference_task, current_task)
        async with self._sample_lock:
            self._sent_request_stream_times_ms.append(event.stream_time_ms)
        if self.logger:
            self.logger.log(
                event.stream_time_ms,
                (
                    f"Sending frame-change verifier request gate={self.gate_id} model={self.model} "
                    f"mode={self.baseline_mode} reference_video={reference_time_ms / 1000:.2f}s"
                ),
            )
        start = perf_counter()
        state_snapshot = event.payload.get("state", {})
        output_text = await asyncio.to_thread(
            self._stream_change_verification,
            reference_url,
            current_url,
            state_snapshot,
        )
        elapsed = perf_counter() - start
        matched = output_text.strip().upper().startswith("YES")
        confidence = 1.0 if matched else 0.0
        if self.logger:
            status = "matched" if matched else "no match"
            self.logger.log(
                event.stream_time_ms,
                f"Frame-change verifier returned gate={self.gate_id} status={status} request={elapsed:.2f}s",
            )
        async with self._sample_lock:
            if self._previous_time_ms is None or event.stream_time_ms >= self._previous_time_ms:
                self._previous_frame = frame
                self._previous_time_ms = event.stream_time_ms
        if not matched or confidence < self.min_confidence:
            return None
        if not await self._confirmation_reached(event.stream_time_ms):
            if self.logger:
                self.logger.log(
                    event.stream_time_ms,
                    (
                        f"Frame-change match awaiting confirmation gate={self.gate_id} "
                        f"required_count={self.required_count}"
                    ),
                )
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=confidence,
            reason="Frame-change verification returned YES.",
            evidence={
                "query": self.query,
                "model": self.model,
                "model_confirmed": True,
                "baseline_mode": self.baseline_mode,
                "reference_stream_time_ms": reference_time_ms,
                "stream_time_ms": event.stream_time_ms,
                "initial_state": self._initial_state,
                "frame": frame,
            },
        )

    async def _confirmation_reached(self, stream_time_ms: int) -> bool:
        async with self._sample_lock:
            self._matched_request_stream_times_ms.append(stream_time_ms)
            if self.confirmation_window_ms > 0:
                cutoff = stream_time_ms - self.confirmation_window_ms
                while self._matched_request_stream_times_ms and self._matched_request_stream_times_ms[0] < cutoff:
                    self._matched_request_stream_times_ms.popleft()
            return len(self._matched_request_stream_times_ms) >= self.required_count

    def _schedule_initial_state_update(self, frame: object, stream_time_ms: int) -> None:
        if self.state_interval_ms <= 0 or stream_time_ms - self._last_state_time_ms < self.state_interval_ms:
            return
        self._last_state_time_ms = stream_time_ms
        self._state_task = asyncio.create_task(self._update_initial_state(frame, stream_time_ms))

    async def _update_initial_state(self, frame: object, stream_time_ms: int) -> None:
        image_url = await asyncio.to_thread(self._encode_frame_data_url, frame)
        try:
            self._initial_state = await asyncio.to_thread(self._request_state_summary, image_url, "initial")
            if self.logger and self._initial_state:
                self.logger.log(stream_time_ms, f"Initial state gate={self.gate_id}: {self._initial_state}")
        except Exception as exc:
            if self.logger:
                self.logger.log(stream_time_ms, f"Initial state request failed gate={self.gate_id} error={exc}")

    async def _should_skip_due_to_nearby_queued_request(self, stream_time_ms: int) -> bool:
        if self.queued_frame_dedupe_window_ms <= 0:
            return False
        queue_depth = self.runtime_queue.qsize() if self.runtime_queue is not None else 0
        if queue_depth <= 0:
            return False
        async with self._sample_lock:
            return any(
                abs(stream_time_ms - sent_time_ms) <= self.queued_frame_dedupe_window_ms
                for sent_time_ms in self._sent_request_stream_times_ms
            )

    async def _wait_for_request_slot(self) -> None:
        if self.min_request_interval_ms <= 0:
            return
        async with self._request_spacing_lock:
            now = perf_counter()
            wait_seconds = self._next_request_wall_time - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = perf_counter()
            self._next_request_wall_time = now + self.min_request_interval_ms / 1000

    def _stream_change_verification(self, reference_url: str, current_url: str, state_snapshot: object = None) -> str:
        request = ModelRequest(
            model=self.model,
            service_tier=self.service_tier,
            effort="none",
            max_output_tokens=16,
            system=(
                "You are a strict realtime video change verifier. Reply with exactly YES or NO. "
                "You receive a reference frame followed by a current frame. Reply YES only if "
                "the user's requested change clearly happened between them and the current frame "
                "clearly shows the completed post-change state. Reply NO for partial progress, "
                "ambiguous motion, or an event that only appears to be starting. Reply NO if uncertain."
            ),
            content=[
                TextPart(f"Initial/reference state: {self._initial_state or 'unknown'}"),
                TextPart(f"Current internal state: {self._state_text(state_snapshot)}"),
                TextPart(f"Requested change: {self.query}"),
                TextPart("Reference frame:"),
                ImagePart(reference_url),
                TextPart("Current frame:"),
                ImagePart(current_url),
            ],
        )
        return self._first_binary_decision(request)

    def _request_state_summary(self, image_url: str, label: str) -> str:
        text = self.provider.complete_text(
            ModelRequest(
                model=self.model,
                service_tier=self.service_tier,
                effort="none",
                max_output_tokens=48,
                system=(
                    "Briefly describe the visual state relevant to the monitoring request. "
                    "Do not answer YES or NO; describe what is visible."
                ),
                content=[
                    TextPart(
                        f"Monitoring request: {self.query}. Frame role: {label}. "
                        "Describe the current state in one short sentence."
                    ),
                    ImagePart(image_url),
                ],
            )
        )
        return text.strip()

    def _first_binary_decision(self, request: ModelRequest) -> str:
        """Stream the verdict and stop at the first token that settles it."""
        output_text = ""
        for delta in self.provider.stream_text(request):
            output_text += delta
            normalized = output_text.strip().upper()
            if normalized.startswith("YES"):
                return "YES"
            if normalized.startswith("NO"):
                return "NO"
        return output_text

    def _state_text(self, state_snapshot: object) -> str:
        if not state_snapshot:
            return "empty"
        try:
            return json.dumps(state_snapshot, sort_keys=True)
        except TypeError:
            return str(state_snapshot)

    def _encode_frame_data_url(self, frame: object) -> str:
        import cv2  # type: ignore

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("Could not encode video frame as JPEG.")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"



class ModelStateMonitorGate(ModelBacked, Gate):
    """Periodic model-written summary of the stream's current state."""

    def __init__(
        self,
        gate_id: str,
        query: str,
        sample_interval_ms: int = 3000,
        model: Optional[str] = None,
        service_tier: str = "default",
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.query = query
        self.sample_interval_ms = sample_interval_ms
        self.model = model or ModelConfig.from_env().cheap_model
        self.service_tier = "default" if service_tier == "standard" else service_tier
        self._last_sample_time_ms = -sample_interval_ms

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        image_url = await asyncio.to_thread(self._encode_frame_data_url, frame)
        start = perf_counter()
        state_snapshot = event.payload.get("state", {})
        state = await asyncio.to_thread(self._request_state_summary, image_url, state_snapshot)
        elapsed = perf_counter() - start
        if self.logger:
            self.logger.log(event.stream_time_ms, f"State monitor returned gate={self.gate_id} request={elapsed:.2f}s state={state}")
        return GateFire(
            gate_id=self.gate_id,
            confidence=0.5,
            reason=f"State observation: {state}",
            evidence={"state_summary": state, "stream_time_ms": event.stream_time_ms, "frame": frame},
        )

    def _request_state_summary(self, image_url: str, state_snapshot: object = None) -> str:
        text = self.provider.complete_text(
            ModelRequest(
                model=self.model,
                service_tier=self.service_tier,
                effort="none",
                max_output_tokens=64,
                system="Summarize current visual state relevant to the monitoring request.",
                content=[
                    TextPart(f"Monitoring request: {self.query}"),
                    TextPart(f"Current internal state: {self._state_text(state_snapshot)}"),
                    ImagePart(image_url),
                ],
            )
        )
        return text.strip()

    def _encode_frame_data_url(self, frame: object) -> str:
        import cv2  # type: ignore

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("Could not encode video frame as JPEG.")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _state_text(self, state_snapshot: object) -> str:
        if not state_snapshot:
            return "empty"
        try:
            return json.dumps(state_snapshot, sort_keys=True)
        except TypeError:
            return str(state_snapshot)



class ObjectTrackGate(Gate):
    """Track object centers from upstream detector fires and detect simple trends."""

    def __init__(
        self,
        gate_id: str,
        upstream_gate_id: str,
        labels: Iterable[str],
        change: str = "moving_apart",
        window_ms: int = 2000,
        min_delta_ratio: float = 0.25,
    ) -> None:
        super().__init__(gate_id, [EventType.GATE_FIRE])
        self.terminal_response = True
        self.upstream_gate_id = upstream_gate_id
        self.labels = {label.lower() for label in labels}
        self.change = change
        self.window_ms = window_ms
        self.min_delta_ratio = min_delta_ratio
        self._history: Dict[str, Deque[tuple[int, tuple[float, float], Sequence[float]]]] = defaultdict(deque)

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        fire = event.payload.get("gate_fire")
        if not isinstance(fire, GateFire) or fire.gate_id != self.upstream_gate_id:
            return None
        stream_time_ms = int(fire.evidence.get("stream_time_ms", event.stream_time_ms))
        detections = fire.evidence.get("detections", [])
        for detection in detections:
            if not isinstance(detection, dict) or not detection.get("bbox"):
                continue
            label = str(detection.get("label", "")).lower()
            if self.labels and label not in self.labels:
                continue
            bbox = detection["bbox"]
            self._history[label].append((stream_time_ms, self._center(bbox), bbox))
            self._trim(label, stream_time_ms)
        match = self._match(stream_time_ms)
        if match is None:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=match["confidence"],
            reason=f"Object tracking detected {self.change}.",
            evidence={**match, "stream_time_ms": stream_time_ms, "frame": fire.evidence.get("frame")},
        )

    def _match(self, stream_time_ms: int) -> Optional[Dict[str, Any]]:
        active = {label: points for label, points in self._history.items() if len(points) >= 2}
        if self.change in {"moving_apart", "moving_together"} and len(active) >= 2:
            labels = sorted(active)[:2]
            first_start, first_end = active[labels[0]][0], active[labels[0]][-1]
            second_start, second_end = active[labels[1]][0], active[labels[1]][-1]
            start_distance = self._distance(first_start[1], second_start[1])
            end_distance = self._distance(first_end[1], second_end[1])
            scale = max(self._bbox_scale(first_end[2]), self._bbox_scale(second_end[2]), 1.0)
            delta_ratio = (end_distance - start_distance) / scale
            if self.change == "moving_together":
                delta_ratio = -delta_ratio
            if delta_ratio >= self.min_delta_ratio:
                return {
                    "labels": labels,
                    "change": self.change,
                    "start_distance": start_distance,
                    "end_distance": end_distance,
                    "delta_ratio": delta_ratio,
                    "confidence": min(1.0, delta_ratio / max(self.min_delta_ratio, 0.001)),
                }
        if self.change == "moving" and active:
            label, points = next(iter(active.items()))
            start, end = points[0], points[-1]
            delta_ratio = self._distance(start[1], end[1]) / max(self._bbox_scale(end[2]), 1.0)
            if delta_ratio >= self.min_delta_ratio:
                return {
                    "labels": [label],
                    "change": self.change,
                    "delta_ratio": delta_ratio,
                    "confidence": min(1.0, delta_ratio / max(self.min_delta_ratio, 0.001)),
                }
        return None

    def _trim(self, label: str, now_ms: int) -> None:
        points = self._history[label]
        while points and now_ms - points[0][0] > self.window_ms:
            points.popleft()

    def _center(self, bbox: Sequence[float]) -> tuple[float, float]:
        return ((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2)

    def _distance(self, first: tuple[float, float], second: tuple[float, float]) -> float:
        return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5

    def _bbox_scale(self, bbox: Sequence[float]) -> float:
        return max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]), 1.0)
