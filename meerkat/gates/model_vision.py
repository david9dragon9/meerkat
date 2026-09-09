from __future__ import annotations

import base64
import asyncio
import json
import re
from collections import deque
from time import perf_counter
from typing import Deque, Iterable, Optional, Set

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.funnel.gate_types import VALUE_EXTRACTION_MODES
from meerkat.gates.base import Gate
from meerkat.models.config import ModelConfig
from meerkat.models.provider import ImagePart, ModelBacked, ModelRequest, TextPart


OBJECT_DETECTION_SCHEMA = {
    "type": "json_schema",
    "name": "object_detection_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["detected_objects", "confidence", "explanation"],
        "properties": {
            "detected_objects": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "explanation": {"type": "string"},
        },
    },
}


QUERY_MATCH_SCHEMA = {
    "type": "json_schema",
    "name": "vision_query_match",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["matched", "confidence", "explanation"],
        "properties": {
            "matched": {"type": "boolean"},
            "confidence": {"type": "number"},
            "explanation": {"type": "string"},
        },
    },
}


class ModelVisionObjectGate(ModelBacked, Gate):
    """Sampled model vision gate that looks for a fixed set of object labels.

    This is intentionally sampled by stream time. It is useful for the first
    end-to-end test on arbitrary video before local cheap detectors are added.
    """

    def __init__(
        self,
        gate_id: str,
        classes: Iterable[str],
        sample_interval_ms: int = 1000,
        min_confidence: float = 0.6,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.classes: Set[str] = {item.lower() for item in classes}
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.model = model or ModelConfig.from_env().mid_model
        self._last_sample_time_ms = -sample_interval_ms

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms

        image_url = self._encode_frame_data_url(frame)
        if self.logger:
            self.logger.log(
                event.stream_time_ms,
                f"Sending vision detector request gate={self.gate_id} model={self.model}",
            )
        request_start = perf_counter()
        data = await asyncio.to_thread(
            self.provider.complete_json,
            ModelRequest(
                model=self.model,
                effort="low",
                schema=OBJECT_DETECTION_SCHEMA,
                system=(
                    "You are a strict visual detector. Return JSON only with keys "
                    "detected_objects, confidence, and explanation. Only include objects "
                    "that are visibly present in the image."
                ),
                content=[
                    TextPart(
                        "Check whether any of these objects are visible: "
                        f"{', '.join(sorted(self.classes))}."
                    ),
                    ImagePart(image_url),
                ],
            ),
        )
        request_elapsed = perf_counter() - request_start
        detected = {str(item).lower() for item in data.get("detected_objects", [])}
        matches = sorted(detected.intersection(self.classes))
        confidence = float(data.get("confidence", 0.0))
        if self.logger:
            status = "matched" if matches and confidence >= self.min_confidence else "no match"
            detected_text = ", ".join(sorted(detected)) if detected else "none"
            self.logger.log(
                event.stream_time_ms,
                (
                    f"Vision detector returned gate={self.gate_id} status={status} "
                    f"confidence={confidence:.2f} detected={detected_text} request={request_elapsed:.2f}s"
                ),
            )
        if not matches or confidence < self.min_confidence:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=confidence,
            reason=str(data.get("explanation", "OpenAI vision object match")),
            evidence={
                "matches": matches,
                "model": self.model,
                "model_confirmed": True,
                "stream_time_ms": event.stream_time_ms,
                "sample_interval_ms": self.sample_interval_ms,
            },
        )

    def _encode_frame_data_url(self, frame: object) -> str:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ModelVisionObjectGate requires the 'media' optional dependencies.") from exc

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("Could not encode video frame as JPEG.")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"



class ModelVisionQueryGate(ModelBacked, Gate):
    """Sampled model gate that checks the user's full visual condition."""

    def __init__(
        self,
        gate_id: str,
        query: str,
        sample_interval_ms: int = 1000,
        min_confidence: float = 0.75,
        model: Optional[str] = None,
        upstream_gate_id: Optional[str] = None,
        verification_mode: str = "binary",
        concurrent_requests: int = 3,
        min_request_interval_ms: int = 200,
        queued_frame_dedupe_window_ms: int = 500,
        service_tier: str = "default",
        window_ms: int = 0,
        required_count: int = 1,
        confirmation_window_ms: int = 1500,
    ) -> None:
        input_types = [EventType.GATE_FIRE] if upstream_gate_id else [EventType.VIDEO_FRAME]
        if upstream_gate_id and window_ms > 0:
            input_types = [EventType.GATE_FIRE, EventType.VIDEO_FRAME]
        super().__init__(gate_id, input_types)
        self.worker_count = max(1, concurrent_requests)
        self.queue_size = max(1, self.worker_count * 4)
        self.terminal_response = True
        self.query = query
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.model = model or ModelConfig.from_env().mid_model
        self.upstream_gate_id = upstream_gate_id
        self.verification_mode = verification_mode
        self.min_request_interval_ms = min_request_interval_ms
        self.queued_frame_dedupe_window_ms = queued_frame_dedupe_window_ms
        self.service_tier = self._normalize_service_tier(service_tier)
        self.window_ms = max(0, window_ms)
        self.required_count = max(1, required_count)
        self.confirmation_window_ms = max(0, confirmation_window_ms)
        self._last_sample_time_ms = -sample_interval_ms
        self._sent_request_stream_times_ms: Deque[int] = deque(maxlen=128)
        self._matched_request_stream_times_ms: Deque[int] = deque(maxlen=16)
        self._recent_frames: Deque[tuple[int, object]] = deque(maxlen=256)
        self._sample_lock = asyncio.Lock()
        self._request_spacing_lock = asyncio.Lock()
        self._next_request_wall_time = 0.0

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.type == EventType.VIDEO_FRAME:
            frame = event.payload.get("frame")
            if frame is not None and self.window_ms > 0:
                async with self._sample_lock:
                    self._recent_frames.append((event.stream_time_ms, frame))
                    self._trim_recent_frames(event.stream_time_ms)
            return None

        if self.upstream_gate_id:
            upstream_fire = event.payload.get("gate_fire")
            if not isinstance(upstream_fire, GateFire) or upstream_fire.gate_id != self.upstream_gate_id:
                return None
            frame = upstream_fire.evidence.get("frame")
            stream_time_ms = int(upstream_fire.evidence.get("stream_time_ms", event.stream_time_ms))
            upstream_evidence = upstream_fire.evidence
        else:
            frame = event.payload.get("frame")
            stream_time_ms = event.stream_time_ms
            upstream_evidence = {}

        async with self._sample_lock:
            if stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
                return None
            self._last_sample_time_ms = stream_time_ms
        if frame is None:
            return None

        if await self._should_skip_due_to_nearby_queued_request(stream_time_ms):
            return None

        frames = await self._frames_for_request(stream_time_ms, frame)
        encode_task = asyncio.create_task(asyncio.to_thread(self._encode_frames_data_url_timed, frames))
        await self._wait_for_request_slot()
        image_urls, encode_elapsed = await encode_task
        async with self._sample_lock:
            self._sent_request_stream_times_ms.append(stream_time_ms)
        if self.logger:
            self.logger.log(
                stream_time_ms,
                (
                    f"Sending vision verifier request gate={self.gate_id} model={self.model} "
                    f"mode={self.verification_mode} service_tier={self.service_tier} "
                    f"encode={encode_elapsed:.2f}s frames={len(image_urls)}"
                ),
            )
        request_start = perf_counter()
        state_snapshot = event.payload.get("state", {})
        output_text = await self._request_verification_text(image_urls, state_snapshot)
        request_elapsed = perf_counter() - request_start
        matched, confidence, explanation, extra_evidence = self._parse_verification_response(output_text)
        if self.logger:
            status = "matched" if matched and confidence >= self.min_confidence else "no match"
            self.logger.log(
                stream_time_ms,
                (
                    f"Vision verifier returned gate={self.gate_id} status={status} "
                    f"confidence={confidence:.2f} request={request_elapsed:.2f}s"
                ),
            )
        if not matched or confidence < self.min_confidence:
            return None
        confirmed_stream_time_ms = await self._confirmation_reached(stream_time_ms)
        if confirmed_stream_time_ms is None:
            if self.logger:
                self.logger.log(
                    stream_time_ms,
                    (
                        f"Vision query match awaiting confirmation gate={self.gate_id} "
                        f"required_count={self.required_count}"
                    ),
                )
            return None
        evidence = {
            "query": self.query,
            "model": self.model,
            "model_confirmed": True,
            "stream_time_ms": confirmed_stream_time_ms,
            "sample_interval_ms": self.sample_interval_ms,
            "upstream_gate_id": self.upstream_gate_id,
            "window_ms": self.window_ms,
            "required_count": self.required_count,
        }
        if upstream_evidence:
            evidence["upstream_evidence"] = upstream_evidence
            for key in ("matches", "text", "state_summary"):
                if key in upstream_evidence:
                    evidence[key] = upstream_evidence[key]
        evidence.update(extra_evidence)
        return GateFire(
            gate_id=self.gate_id,
            confidence=confidence,
            reason=explanation,
            evidence=evidence,
        )

    def _encode_frame_data_url(self, frame: object) -> str:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ModelVisionQueryGate requires the default media dependencies.") from exc

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("Could not encode video frame as JPEG.")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _encode_frame_data_url_timed(self, frame: object) -> tuple[str, float]:
        start = perf_counter()
        image_url = self._encode_frame_data_url(frame)
        return image_url, perf_counter() - start

    def _encode_frames_data_url_timed(self, frames: list[object]) -> tuple[list[str], float]:
        start = perf_counter()
        image_urls = [self._encode_frame_data_url(frame) for frame in frames]
        return image_urls, perf_counter() - start

    async def _frames_for_request(self, stream_time_ms: int, current_frame: object) -> list[object]:
        if self.window_ms <= 0:
            return [current_frame]
        async with self._sample_lock:
            self._trim_recent_frames(stream_time_ms)
            candidates = [
                frame
                for frame_time_ms, frame in self._recent_frames
                if stream_time_ms - self.window_ms <= frame_time_ms <= stream_time_ms
            ]
        if not candidates or candidates[-1] is not current_frame:
            candidates.append(current_frame)
        if len(candidates) <= 4:
            return candidates
        indexes = [0, len(candidates) // 3, (2 * len(candidates)) // 3, len(candidates) - 1]
        return [candidates[index] for index in indexes]

    def _trim_recent_frames(self, stream_time_ms: int) -> None:
        if self.window_ms <= 0:
            return
        cutoff = stream_time_ms - self.window_ms
        while self._recent_frames and self._recent_frames[0][0] < cutoff:
            self._recent_frames.popleft()

    async def _wait_for_request_slot(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._request_spacing_lock:
            now = loop.time()
            wait_seconds = max(0.0, self._next_request_wall_time - now)
            if wait_seconds:
                await asyncio.sleep(wait_seconds)
            self._next_request_wall_time = loop.time() + (self.min_request_interval_ms / 1000)

    async def _should_skip_due_to_nearby_queued_request(self, stream_time_ms: int) -> bool:
        if self.queued_frame_dedupe_window_ms <= 0:
            return False
        if self.runtime_queue is None or self.runtime_queue.qsize() <= 0:
            return False
        async with self._sample_lock:
            return any(
                abs(stream_time_ms - sent_time_ms) <= self.queued_frame_dedupe_window_ms
                for sent_time_ms in self._sent_request_stream_times_ms
            )

    async def _confirmation_reached(self, stream_time_ms: int) -> Optional[int]:
        async with self._sample_lock:
            self._matched_request_stream_times_ms.append(stream_time_ms)
            if self.confirmation_window_ms > 0:
                cutoff = stream_time_ms - self.confirmation_window_ms
                while self._matched_request_stream_times_ms and self._matched_request_stream_times_ms[0] < cutoff:
                    self._matched_request_stream_times_ms.popleft()
            if len(self._matched_request_stream_times_ms) < self.required_count:
                return None
            return max(self._matched_request_stream_times_ms)

    async def _request_verification_text(self, image_urls: list[str], state_snapshot: object = None) -> str:
        if self.verification_mode == "binary":
            return await asyncio.to_thread(self._stream_binary_verification_text, image_urls, state_snapshot)
        if self.verification_mode in VALUE_EXTRACTION_MODES:
            return await asyncio.to_thread(self._stream_value_extraction_text, image_urls, state_snapshot)
        return await asyncio.to_thread(
            self.provider.complete_text,
            self._request(
                system=(
                        "You are a strict realtime visual condition checker. "
                        "Return JSON only in the requested schema. "
                        "Set matched=true only when the image clearly satisfies the user's full condition. "
                        "For event alerts, matched=true means the event is happening now or has just completed "
                        "in the current/latest frame; do not match setup, partial progress, aftermath, replays, summaries, "
                        "or evidence that only implies the event happened earlier. "
                        "When given an ordered frame window, use the sequence for context but make the decision "
                        "from the latest frame: reply match only if that latest frame shows the completed condition "
                        "or immediate post-event state. "
                        "When the image or ordered frame window contains a broadcast scorebug, dashboard, UI, "
                        "caption, or other persistent state display, use visible state transitions as primary "
                        "evidence. For event/outcome alerts, do not match from a swing, throw, miss, attempt, "
                        "or action-looking frame alone when a persistent state display is visible. Match only "
                        "when the latest frame shows a terminal state that is newly changed versus earlier "
                        "context frames, such as outs/count/score/status changing, or an explicit current "
                        "K/strikeout/out/goal/touchdown/result label. If the latest and earlier context frames "
                        "show the same relevant state, do not match. Do not wait for a later recap, replay, "
                        "lower-third, or next-item screen after the state transition is already visible. "
                        "Do not match merely related objects if the requested action/state is not visible."
                ),
                image_urls=image_urls,
                state_snapshot=state_snapshot,
                lead=[TextPart(f"User condition: {self.query}")],
                schema=QUERY_MATCH_SCHEMA,
                effort="low",
            ),
        )

    def _stream_binary_verification_text(self, image_urls: list[str], state_snapshot: object = None) -> str:
        """Stream a YES/NO verdict, stopping as soon as the answer is decided."""
        request = self._request(
            system=(
                            "You are a strict realtime visual verifier. Reply with exactly YES or NO. "
                            "Reply YES only when the image clearly satisfies the user's full condition. "
                            "For event alerts, YES means the event is happening now or has just completed "
                            "in the current/latest frame. Reply NO for setup, partial progress, aftermath, replays, summaries, "
                            "or contextual evidence that only suggests the event happened earlier. "
                            "When given an ordered frame window, use earlier frames only as context and base YES/NO "
                            "on the latest frame; reply YES only if that latest frame shows the completed condition "
                            "or immediate post-event state. "
                            "For sports or game events, a setup frame, ordinary attempt, swing, throw, miss, score/count "
                            "before the decisive outcome, or player motion before the outcome is recorded is NO. "
                            "Reply YES only when the current/latest frame shows the completed outcome, official signal, "
                            "or visible state change that proves the outcome. "
                            "When the image or ordered frame window contains a broadcast scorebug, dashboard, UI, "
                            "caption, or other persistent state display, use visible state transitions as primary "
                            "evidence. For event/outcome alerts, do not reply YES from a swing, throw, miss, "
                            "attempt, or action-looking frame alone when a persistent state display is visible. "
                            "Reply YES only when the latest frame shows a terminal state that is newly changed "
                            "versus earlier context frames, such as outs/count/score/status changing, or an "
                            "explicit current K/strikeout/out/goal/touchdown/result label. If the latest and "
                            "earlier context frames show the same relevant state, reply NO. Do not wait for a "
                            "later recap, replay, lower-third, or next-item screen after the state transition "
                            "is already visible. "
                            "Reply NO if uncertain."
            ),
            image_urls=image_urls,
            state_snapshot=state_snapshot,
            lead=[TextPart(f"User condition: {self.query}")],
            max_output_tokens=16,
        )
        output_text = ""
        for delta in self.provider.stream_text(request):
            output_text += delta
            decision = self._binary_decision_from_text(output_text)
            if decision:
                return decision
        return output_text

    def _stream_value_extraction_text(self, image_urls: list[str], state_snapshot: object = None) -> str:
        """Stream an extracted value, stopping as soon as one is complete."""
        request = self._request(
            system=(
                "You are a strict realtime visual answer extractor. "
                "Answer the user's visual question from this image only. "
                "If the answer is not clearly visible, reply exactly NO. "
                "If the answer is visible, reply with only the answer value, no sentence."
            ),
            image_urls=image_urls,
            state_snapshot=state_snapshot,
            lead=[TextPart(f"Question: {self.query}")],
            max_output_tokens=24,
        )
        output_text = ""
        for delta in self.provider.stream_text(request):
            output_text += delta
            value = self._extracted_value_from_text(output_text)
            if value:
                return value
        return output_text

    def _request(
        self,
        *,
        system: str,
        image_urls: list[str],
        state_snapshot: object,
        lead: list[TextPart],
        max_output_tokens: Optional[int] = None,
        schema: Optional[dict] = None,
        effort: str = "none",
    ) -> ModelRequest:
        """Assemble one verifier request: the question, current state, then frames."""
        return ModelRequest(
            model=self.model,
            system=system,
            content=[
                *lead,
                TextPart(f"Current internal state: {self._state_text(state_snapshot)}"),
                *self._image_content(image_urls),
            ],
            max_output_tokens=max_output_tokens,
            schema=schema,
            effort=effort,
            service_tier=self.service_tier,
        )

    def _binary_decision_from_text(self, output_text: str) -> str:
        normalized = output_text.strip().upper()
        if normalized.startswith("YES"):
            return "YES"
        if normalized.startswith("NO"):
            return "NO"
        return ""

    def _image_content(self, image_urls: list[str]) -> list[object]:
        """Frames, labelled so the model knows which one the verdict is about."""
        if len(image_urls) <= 1:
            return [ImagePart(image_urls[0])]
        content: list[object] = [
            TextPart(
                "The following images are ordered from earlier to later within the recent stream window. "
                "Only the LAST/CURRENT frame controls YES or NO; earlier frames are context only."
            )
        ]
        for index, image_url in enumerate(image_urls, start=1):
            label = "LAST/CURRENT FRAME" if index == len(image_urls) else "context frame"
            content.append(TextPart(f"Frame {index} of {len(image_urls)} ({label}):"))
            content.append(ImagePart(image_url))
        return content

    def _parse_verification_response(self, output_text: str) -> tuple[bool, float, str, dict[str, object]]:
        if self.verification_mode == "binary":
            normalized = output_text.strip().upper()
            matched = normalized.startswith("YES")
            confidence = 1.0 if matched else 0.0
            reason = f"Binary vision verification returned {'YES' if matched else 'NO'}."
            return matched, confidence, reason, {}

        if self.verification_mode in VALUE_EXTRACTION_MODES:
            value = self._extracted_value_from_text(output_text)
            matched = bool(value)
            confidence = 1.0 if matched else 0.0
            reason = f"Vision verifier extracted value: {value}." if matched else "Vision verifier extraction returned NO."
            evidence = {"text": value, "value": value, "answer": value} if matched else {}
            return matched, confidence, reason, evidence

        data = json.loads(output_text)
        matched = bool(data.get("matched", False))
        confidence = float(data.get("confidence", 0.0))
        explanation = str(data.get("explanation", "Vision verifier result."))
        return matched, confidence, explanation, {}

    def _extracted_value_from_text(self, output_text: str) -> str:
        value = output_text.strip().strip("`\"' ")
        if not value:
            return ""
        first_line = value.splitlines()[0].strip()
        normalized = first_line.upper().rstrip(".")
        if normalized.startswith("NO") or normalized in {"N/A", "UNKNOWN", "UNCLEAR"}:
            return ""
        if re.search(r"\b(number|digits?)\b", self.query, flags=re.I):
            match = re.search(r"\b\d{1,3}\b", first_line)
            return match.group(0) if match else ""
        return first_line

    def _state_text(self, state_snapshot: object) -> str:
        if not state_snapshot:
            return "empty"
        try:
            return json.dumps(state_snapshot, sort_keys=True)
        except TypeError:
            return str(state_snapshot)


    def _normalize_service_tier(self, service_tier: str) -> str:
        if service_tier == "standard":
            return "default"
        return service_tier
