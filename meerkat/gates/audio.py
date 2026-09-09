from __future__ import annotations

import io
import math
import re
import wave
import asyncio
from collections import deque
from time import perf_counter
from typing import Any, Iterable, Optional

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.funnel.gate_types import VALUE_EXTRACTION_MODES
from meerkat.gates.base import Gate
from meerkat.models.config import ModelConfig
from meerkat.models.provider import ModelBacked, ModelRequest, TextPart


class AudioVolumeGate(Gate):
    def __init__(
        self,
        gate_id: str,
        min_volume_db: float = -35.0,
        max_volume_db: Optional[float] = None,
    ) -> None:
        super().__init__(gate_id, [EventType.AUDIO_CHUNK])
        self.min_volume_db = min_volume_db
        self.max_volume_db = max_volume_db

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        samples = _samples(event)
        if samples is None or samples.size == 0:
            return None
        rms = float((samples.astype("float32") ** 2).mean() ** 0.5)
        volume_db = 20 * math.log10(max(rms, 1e-6))
        if volume_db < self.min_volume_db:
            return None
        if self.max_volume_db is not None and volume_db > self.max_volume_db:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=min(1.0, max(0.0, (volume_db - self.min_volume_db) / 20.0)),
            reason=f"Audio volume matched: {volume_db:.1f} dBFS",
            evidence={"volume_db": volume_db, "stream_time_ms": event.stream_time_ms},
        )


class AudioPitchGate(Gate):
    def __init__(
        self,
        gate_id: str,
        min_pitch_hz: Optional[float] = None,
        max_pitch_hz: Optional[float] = None,
        min_confidence: float = 0.4,
    ) -> None:
        super().__init__(gate_id, [EventType.AUDIO_CHUNK])
        self.min_pitch_hz = min_pitch_hz
        self.max_pitch_hz = max_pitch_hz
        self.min_confidence = min_confidence

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        samples = _samples(event)
        sample_rate = int(event.payload.get("sample_rate", 16000))
        if samples is None or samples.size < 8 or sample_rate <= 0:
            return None
        pitch_hz, confidence = _zero_crossing_pitch(samples, sample_rate)
        if confidence < self.min_confidence:
            return None
        if self.min_pitch_hz is not None and pitch_hz < self.min_pitch_hz:
            return None
        if self.max_pitch_hz is not None and pitch_hz > self.max_pitch_hz:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=confidence,
            reason=f"Audio pitch matched: {pitch_hz:.0f} Hz",
            evidence={"pitch_hz": pitch_hz, "stream_time_ms": event.stream_time_ms},
        )


class SpeakingCadenceGate(Gate):
    def __init__(
        self,
        gate_id: str,
        min_words_per_minute: Optional[float] = None,
        max_words_per_minute: Optional[float] = None,
        window_ms: int = 5000,
        upstream_gate_id: Optional[str] = None,
    ) -> None:
        input_types = [EventType.GATE_FIRE] if upstream_gate_id else [EventType.TRANSCRIPT_DELTA]
        super().__init__(gate_id, input_types)
        self.min_words_per_minute = min_words_per_minute
        self.max_words_per_minute = max_words_per_minute
        self.window_ms = window_ms
        self.upstream_gate_id = upstream_gate_id
        self._items: deque[tuple[int, int]] = deque()

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        text = str(event.payload.get("text", ""))
        upstream_evidence = None
        if event.type == EventType.GATE_FIRE:
            fire = event.payload.get("gate_fire")
            if not isinstance(fire, GateFire) or fire.gate_id != self.upstream_gate_id:
                return None
            upstream_evidence = fire.evidence
            text = str(upstream_evidence.get("text", ""))
        word_count = len([word for word in text.split() if word.strip()])
        if word_count <= 0:
            return None
        self._items.append((event.stream_time_ms, word_count))
        cutoff = event.stream_time_ms - self.window_ms
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()
        total_words = sum(count for _, count in self._items)
        duration_ms = max(1000, min(self.window_ms, event.stream_time_ms - self._items[0][0] + 1000))
        wpm = total_words * 60000 / duration_ms
        if self.min_words_per_minute is not None and wpm < self.min_words_per_minute:
            return None
        if self.max_words_per_minute is not None and wpm > self.max_words_per_minute:
            return None
        evidence = {"words_per_minute": wpm, "text": text, "stream_time_ms": event.stream_time_ms}
        if upstream_evidence:
            evidence["upstream_evidence"] = upstream_evidence
        return GateFire(self.gate_id, 1.0, f"Speaking cadence matched: {wpm:.0f} wpm", evidence)


class LocalRealtimeTranscriptionGate(Gate):
    def __init__(
        self,
        gate_id: str,
        model: str = "tiny.en",
        model_path: Optional[str] = None,
        compute_type: str = "int8",
        language: Optional[str] = "en",
        buffer_ms: int = 2000,
        sample_interval_ms: int = 1000,
        min_volume_db: float = -45.0,
        text_window_ms: int = 12000,
    ) -> None:
        super().__init__(gate_id, [EventType.AUDIO_CHUNK])
        self.model = model
        self.model_path = model_path
        self.compute_type = compute_type
        self.language = language
        self.buffer_ms = buffer_ms
        self.sample_interval_ms = sample_interval_ms
        self.min_volume_db = min_volume_db
        self.text_window_ms = text_window_ms
        self._last_sample_time_ms = -sample_interval_ms
        self._chunks: deque[tuple[int, object]] = deque()
        self._transcript_history: deque[tuple[int, str]] = deque()
        self._name_memory: list[str] = []
        self._whisper_model = None
        self.queue_size = 8

    async def warmup(self) -> None:
        if self.logger:
            self.logger.log(None, f"Warming up local transcription gate gate={self.gate_id} model={self.model_path or self.model}")
        start = perf_counter()
        await asyncio.to_thread(self._load_model)
        if self.logger:
            self.logger.log(None, f"Local transcription gate warmed gate={self.gate_id} elapsed={perf_counter() - start:.2f}s")

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        events = self._coalesce_queued_audio_events(event)
        latest_event = events[-1]
        for audio_event in events:
            samples = _samples(audio_event)
            if samples is None or samples.size == 0:
                continue
            self._chunks.append((audio_event.stream_time_ms, samples))
        if not self._chunks:
            return None
        cutoff = latest_event.stream_time_ms - self.buffer_ms
        while self._chunks and self._chunks[0][0] < cutoff:
            self._chunks.popleft()
        if latest_event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        audio = _concat_chunks([chunk for _, chunk in self._chunks])
        if audio is None or audio.size == 0:
            return None
        volume_db = _volume_db(audio)
        if volume_db < self.min_volume_db:
            return None
        self._last_sample_time_ms = latest_event.stream_time_ms
        if self.logger:
            skipped = len(events) - 1
            suffix = f" coalesced_chunks={skipped}" if skipped else ""
            self.logger.log(
                latest_event.stream_time_ms,
                f"Sending local transcription request gate={self.gate_id} model={self.model_path or self.model}{suffix}",
            )
        start = perf_counter()
        text = await asyncio.to_thread(self._transcribe, audio)
        elapsed = perf_counter() - start
        text = text.strip()
        if self.logger:
            status = "matched" if text else "no speech"
            self.logger.log(latest_event.stream_time_ms, f"Local transcription returned gate={self.gate_id} status={status} request={elapsed:.2f}s")
        if not text:
            return None
        self._transcript_history.append((latest_event.stream_time_ms, text))
        text_cutoff = latest_event.stream_time_ms - self.text_window_ms
        while self._transcript_history and self._transcript_history[0][0] < text_cutoff:
            self._transcript_history.popleft()
        text_window = " ".join(item_text for _, item_text in self._transcript_history if item_text).strip()
        for name in _candidate_names(text):
            if name not in self._name_memory:
                self._name_memory.append(name)
        name_context = " ".join(self._name_memory)
        return GateFire(
            self.gate_id,
            1.0,
            "Local realtime transcription produced text.",
            {
                "text": text,
                "text_window": text_window,
                "name_context": name_context,
                "volume_db": volume_db,
                "model": self.model_path or self.model,
                "stream_time_ms": latest_event.stream_time_ms,
            },
        )

    def _coalesce_queued_audio_events(self, event: StreamEvent) -> list[StreamEvent]:
        events = [event]
        if self.runtime_queue is None:
            return events
        while True:
            try:
                queued_event = self.runtime_queue.get_nowait()
            except asyncio.QueueEmpty:
                return events
            if queued_event.type == EventType.AUDIO_CHUNK:
                events.append(queued_event)

    def _load_model(self) -> object:
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "LocalRealtimeTranscriptionGate requires faster-whisper. Run `uv sync` to install default dependencies."
            ) from exc
        model_name = self.model_path or self.model
        self._whisper_model = WhisperModel(model_name, device="cpu", compute_type=self.compute_type)
        return self._whisper_model

    def _transcribe(self, audio) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


class ModelTranscriptQueryGate(ModelBacked, Gate):
    def __init__(
        self,
        gate_id: str,
        query: str,
        model: Optional[str] = None,
        upstream_gate_id: Optional[str] = None,
        verification_mode: str = "binary",
        service_tier: str = "default",
    ) -> None:
        input_types = [EventType.GATE_FIRE] if upstream_gate_id else [EventType.TRANSCRIPT_DELTA]
        super().__init__(gate_id, input_types)
        self.query = query
        self.model = model or ModelConfig.from_env().cheap_model
        self.upstream_gate_id = upstream_gate_id
        self.verification_mode = verification_mode
        self.service_tier = "default" if service_tier == "standard" else service_tier
        self.terminal_response = True

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        text = str(event.payload.get("text", ""))
        upstream_evidence: Optional[dict[str, Any]] = None
        if event.type == EventType.GATE_FIRE:
            fire = event.payload.get("gate_fire")
            if not isinstance(fire, GateFire) or fire.gate_id != self.upstream_gate_id:
                return None
            upstream_evidence = fire.evidence
            text = str(upstream_evidence.get("text", ""))
        if not text:
            return None
        if self.logger:
            self.logger.log(event.stream_time_ms, f"Sending transcript verifier request gate={self.gate_id} model={self.model}")
        start = perf_counter()
        output_text = await self._request_text(text, event.payload.get("state", {}))
        elapsed = perf_counter() - start
        matched, confidence, reason, extra = self._parse_output(output_text, text)
        if self.logger:
            status = "matched" if matched else "no match"
            self.logger.log(
                event.stream_time_ms,
                f"Transcript verifier returned gate={self.gate_id} status={status} confidence={confidence:.2f} request={elapsed:.2f}s",
            )
        if not matched:
            return None
        evidence = {
            "text": text,
            "query": self.query,
            "model": self.model,
            "model_confirmed": True,
            "stream_time_ms": event.stream_time_ms,
            **extra,
        }
        if upstream_evidence:
            evidence["upstream_evidence"] = upstream_evidence
        return GateFire(self.gate_id, confidence, reason, evidence)

    async def _request_text(self, transcript: str, state_snapshot: object) -> str:
        import asyncio

        return await asyncio.to_thread(self._stream_text, transcript, state_snapshot)

    def _stream_text(self, transcript: str, state_snapshot: object) -> str:
        instruction = (
            "You are a strict realtime transcript verifier. Reply exactly YES or NO. "
            "Reply YES only if the transcript satisfies the user's condition."
        )
        if self.verification_mode in VALUE_EXTRACTION_MODES:
            instruction = (
                "You are a strict realtime transcript answer extractor. "
                "Use the transcript context to answer the user's question. "
                "If the answer is not clearly present in the transcript, reply exactly NO. "
                "If the user asks who, reply with only the person's name. "
                "Otherwise reply with only the answer value. Do not answer with unrelated words."
            )
        request = ModelRequest(
            model=self.model,
            service_tier=self.service_tier,
            effort="none",
            max_output_tokens=24,
            system=instruction,
            content=[
                TextPart(
                    f"User condition/question: {self.query}\n"
                    f"Current internal state: {state_snapshot or 'empty'}\n"
                    f"Transcript: {transcript}"
                )
            ],
        )
        output = ""
        for delta in self.provider.stream_text(request):
            output += delta
            parsed = self._parse_partial(output)
            if parsed:
                return parsed
        return output

    def _parse_partial(self, output: str) -> str:
        stripped = output.strip()
        normalized = stripped.upper()
        if normalized.startswith("YES"):
            return "YES"
        if normalized.startswith("NO"):
            return "NO"
        return ""

    def _parse_output(self, output: str, transcript_context: str = "") -> tuple[bool, float, str, dict[str, object]]:
        text = output.strip().strip("`\"' ")
        if self.verification_mode in VALUE_EXTRACTION_MODES:
            if not text or text.upper().startswith("NO"):
                return False, 0.0, "Transcript verifier extraction returned NO.", {}
            if _looks_like_partial_name_answer(self.query, text):
                return False, 0.0, f"Transcript verifier extraction returned partial value: {text}.", {}
            text = _normalize_extracted_name(self.query, text, transcript_context)
            return True, 1.0, f"Transcript verifier extracted value: {text}.", {"text": text, "answer": text, "value": text}
        matched = text.upper().startswith("YES")
        return matched, 1.0 if matched else 0.0, f"Transcript verification returned {'YES' if matched else 'NO'}.", {}



class HostedTranscriptionGate(ModelBacked, Gate):
    def __init__(
        self,
        gate_id: str,
        model: str = "gpt-4o-mini-transcribe",
        sample_interval_ms: int = 1000,
        min_volume_db: float = -45.0,
    ) -> None:
        super().__init__(gate_id, [EventType.AUDIO_CHUNK])
        self.model = model
        self.sample_interval_ms = sample_interval_ms
        self.min_volume_db = min_volume_db
        self._last_sample_time_ms = -sample_interval_ms

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        samples = _samples(event)
        if samples is None or samples.size == 0:
            return None
        rms = float((samples.astype("float32") ** 2).mean() ** 0.5)
        volume_db = 20 * math.log10(max(rms, 1e-6))
        if volume_db < self.min_volume_db:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        sample_rate = int(event.payload.get("sample_rate", 16000))
        wav_bytes = _wav_bytes(samples, sample_rate)
        if self.logger:
            self.logger.log(event.stream_time_ms, f"Sending audio transcription request gate={self.gate_id} model={self.model}")
        start = perf_counter()
        text = await self._transcribe(wav_bytes)
        elapsed = perf_counter() - start
        text = text.strip()
        if self.logger:
            status = "matched" if text else "no speech"
            self.logger.log(event.stream_time_ms, f"Audio transcription returned gate={self.gate_id} status={status} request={elapsed:.2f}s")
        if not text:
            return None
        return GateFire(
            self.gate_id,
            1.0,
            "Audio transcription produced text.",
            {"text": text, "stream_time_ms": event.stream_time_ms, "model": self.model},
        )

    async def _transcribe(self, wav_bytes: bytes) -> str:
        return await asyncio.to_thread(self.provider.transcribe, wav_bytes, self.model)



class TemporalJoinGate(Gate):
    def __init__(self, gate_id: str, gate_ids: Iterable[str], join_window_ms: int = 1000) -> None:
        super().__init__(gate_id, [EventType.GATE_FIRE])
        self.gate_ids = {str(gate_id) for gate_id in gate_ids}
        self.join_window_ms = join_window_ms
        self._recent: dict[str, tuple[int, GateFire]] = {}

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        fire = event.payload.get("gate_fire")
        if not isinstance(fire, GateFire) or fire.gate_id not in self.gate_ids:
            return None
        self._recent[fire.gate_id] = (event.stream_time_ms, fire)
        cutoff = event.stream_time_ms - self.join_window_ms
        self._recent = {gate_id: item for gate_id, item in self._recent.items() if item[0] >= cutoff}
        if not self.gate_ids.issubset(self._recent):
            return None
        fires = {gate_id: item[1] for gate_id, item in self._recent.items() if gate_id in self.gate_ids}
        evidence = {
            "joined_gate_ids": sorted(self.gate_ids),
            "joined_evidence": {gate_id: joined_fire.evidence for gate_id, joined_fire in fires.items()},
            "stream_time_ms": event.stream_time_ms,
        }
        frame = fire.evidence.get("frame")
        if frame is None:
            frame = next(
                (
                    joined_fire.evidence.get("frame")
                    for _, joined_fire in sorted(self._recent.values(), key=lambda item: item[0], reverse=True)
                    if joined_fire.evidence.get("frame") is not None
                ),
                None,
            )
        if frame is not None:
            evidence["frame"] = frame
        if any(joined_fire.evidence.get("model_confirmed") for joined_fire in fires.values()):
            evidence["model_confirmed"] = True
        return GateFire(
            self.gate_id,
            min(joined_fire.confidence for joined_fire in fires.values()),
            f"Joined gates fired within {self.join_window_ms}ms: {', '.join(sorted(self.gate_ids))}",
            evidence,
        )


def _samples(event: StreamEvent):
    samples = event.payload.get("samples")
    if samples is None:
        return None
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Audio gates require numpy.") from exc
    array = np.asarray(samples, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array


def _concat_chunks(chunks: list[object]):
    if not chunks:
        return None
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Audio gates require numpy.") from exc
    return np.concatenate(chunks).astype("float32")


def _volume_db(samples) -> float:
    rms = float((samples.astype("float32") ** 2).mean() ** 0.5)
    return 20 * math.log10(max(rms, 1e-6))


def _zero_crossing_pitch(samples, sample_rate: int) -> tuple[float, float]:
    import numpy as np

    centered = samples - float(np.mean(samples))
    crossings = np.where(np.diff(np.signbit(centered)))[0]
    if len(crossings) < 2:
        return 0.0, 0.0
    duration_seconds = len(samples) / sample_rate
    pitch_hz = len(crossings) / (2 * duration_seconds)
    rms = float((centered.astype("float32") ** 2).mean() ** 0.5)
    confidence = min(1.0, max(0.0, rms * 10))
    return float(pitch_hz), confidence


def _wav_bytes(samples, sample_rate: int) -> bytes:
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _looks_like_partial_name_answer(query: str, text: str) -> bool:
    if not re.search(r"\b(who|name)\b", query, flags=re.I):
        return False
    value = text.strip().strip(".")
    return len(value) <= 1


def _normalize_extracted_name(query: str, text: str, transcript_context: str) -> str:
    if not re.search(r"\b(who|name)\b", query, flags=re.I):
        return text
    value = text.strip().strip(".")
    candidates = _candidate_names(transcript_context)
    for candidate in candidates:
        if _same_name_variant(value, candidate):
            return candidate
    return value


def _candidate_names(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _same_name_variant(value: str, candidate: str) -> bool:
    left = value.lower()
    right = candidate.lower()
    if left == right:
        return True
    if left in {"soph", "sophi", "sophie", "sofia"} and right == "sophia":
        return True
    if left.startswith("sof") and right.startswith("soph"):
        return True
    return False
