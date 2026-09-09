from __future__ import annotations

import math
import asyncio

import numpy as np

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.funnel.factory import build_gate
from meerkat.funnel.spec import GateSpec
from meerkat.gates.audio import (
    AudioPitchGate,
    AudioVolumeGate,
    LocalRealtimeTranscriptionGate,
    ModelTranscriptQueryGate,
    SpeakingCadenceGate,
    TemporalJoinGate,
)
from meerkat.gates.transcript import TranscriptKeywordGate

from conftest import StubProvider


def _audio_event(samples, stream_time_ms: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.AUDIO_CHUNK,
        source_id="test",
        stream_time_ms=stream_time_ms,
        sequence_id=0,
        payload={"samples": samples, "sample_rate": 16000},
    )


async def test_audio_volume_gate_fires_for_loud_chunk() -> None:
    gate = AudioVolumeGate("loud_audio", min_volume_db=-20)

    fire = await gate.process(_audio_event(np.ones(1600, dtype="float32") * 0.5))

    assert fire is not None
    assert fire.gate_id == "loud_audio"
    assert fire.evidence["volume_db"] > -20


async def test_audio_pitch_gate_estimates_tone() -> None:
    sample_rate = 16000
    t = np.arange(sample_rate // 2, dtype="float32") / sample_rate
    samples = np.sin(2 * math.pi * 440 * t).astype("float32") * 0.5
    gate = AudioPitchGate("pitch", min_pitch_hz=400, max_pitch_hz=480)

    fire = await gate.process(_audio_event(samples))

    assert fire is not None
    assert 400 <= fire.evidence["pitch_hz"] <= 480


async def test_audio_cadence_gate_uses_transcript_word_rate() -> None:
    gate = SpeakingCadenceGate("fast_speech", min_words_per_minute=200, window_ms=2000)
    event = StreamEvent(
        type=EventType.TRANSCRIPT_DELTA,
        source_id="test",
        stream_time_ms=0,
        sequence_id=0,
        payload={"text": "one two three four five"},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.evidence["words_per_minute"] >= 200


async def test_local_realtime_transcription_gate_emits_text_with_fake_model() -> None:
    class Segment:
        text = " urgent now "

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return [Segment()], object()

    gate = LocalRealtimeTranscriptionGate(
        "cheap_realtime_transcription",
        sample_interval_ms=500,
        buffer_ms=1000,
        min_volume_db=-60,
    )
    gate._whisper_model = FakeModel()

    fire = await gate.process(_audio_event(np.ones(1600, dtype="float32") * 0.25, stream_time_ms=0))

    assert fire is not None
    assert fire.evidence["text"] == "urgent now"
    assert fire.evidence["text_window"] == "urgent now"
    assert fire.evidence["model"] == "tiny.en"


async def test_local_realtime_transcription_gate_keeps_text_window_with_fake_model() -> None:
    class Segment:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, **kwargs):
            self.calls += 1
            return [Segment(" Sophia " if self.calls == 1 else " brought the blue umbrella ")], object()

    gate = LocalRealtimeTranscriptionGate(
        "cheap_realtime_transcription",
        sample_interval_ms=500,
        buffer_ms=1000,
        text_window_ms=3000,
        min_volume_db=-60,
    )
    gate._whisper_model = FakeModel()

    first = await gate.process(_audio_event(np.ones(1600, dtype="float32") * 0.25, stream_time_ms=0))
    second = await gate.process(_audio_event(np.ones(1600, dtype="float32") * 0.25, stream_time_ms=1000))

    assert first is not None
    assert second is not None
    assert second.evidence["text"] == "brought the blue umbrella"
    assert second.evidence["text_window"] == "Sophia brought the blue umbrella"
    assert second.evidence["name_context"] == "Sophia"


async def test_local_realtime_transcription_gate_coalesces_queued_audio_to_latest_chunk() -> None:
    class Segment:
        text = " latest speech "

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return [Segment()], object()

    gate = LocalRealtimeTranscriptionGate(
        "cheap_realtime_transcription",
        sample_interval_ms=1000,
        buffer_ms=2000,
        min_volume_db=-60,
    )
    gate._whisper_model = FakeModel()
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    await queue.put(_audio_event(np.ones(1600, dtype="float32") * 0.25, stream_time_ms=500))
    await queue.put(_audio_event(np.ones(1600, dtype="float32") * 0.25, stream_time_ms=1000))
    gate.set_runtime_queue(queue)

    fire = await gate.process(_audio_event(np.ones(1600, dtype="float32") * 0.25, stream_time_ms=0))

    assert fire is not None
    assert fire.evidence["text"] == "latest speech"
    assert fire.evidence["stream_time_ms"] == 1000
    assert queue.empty()


def test_factory_caps_realtime_transcription_buffer_for_latency() -> None:
    gate = build_gate(
        GateSpec(
            id="cheap_realtime_transcription",
            type="local_realtime_transcription",
            params={"model": "small.en", "buffer_ms": 4000, "sample_interval_ms": 1500},
        )
    )

    assert isinstance(gate, LocalRealtimeTranscriptionGate)
    assert gate.buffer_ms == 2000
    assert gate.sample_interval_ms == 1000


async def test_transcript_keyword_gate_can_follow_transcription_gate() -> None:
    gate = TranscriptKeywordGate("keyword", ["urgent"], upstream_gate_id="transcribe")
    upstream_fire = GateFire("transcribe", 1.0, "transcribed", {"text": "urgent", "text_window": "this is urgent"})
    event = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=100,
        sequence_id=1,
        payload={"gate_fire": upstream_fire},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.evidence["matches"] == ["urgent"]
    assert fire.evidence["text"] == "this is urgent"
    assert fire.evidence["current_text"] == "urgent"


async def test_transcript_keyword_gate_preserves_name_context() -> None:
    gate = TranscriptKeywordGate("keyword", ["blue", "umbrella"], upstream_gate_id="transcribe")
    upstream_fire = GateFire(
        "transcribe",
        1.0,
        "transcribed",
        {
            "text": "brought a blue umbrella",
            "text_window": "Sophie brought a blue umbrella",
            "name_context": "Sophia",
        },
    )
    event = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=100,
        sequence_id=1,
        payload={"gate_fire": upstream_fire},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.evidence["matches"] == ["blue", "umbrella"]
    assert fire.evidence["name_context"] == "Sophia"
    assert fire.evidence["text"] == "Sophia Sophie brought a blue umbrella"


async def test_temporal_join_combines_audio_and_video_gate_fires() -> None:
    gate = TemporalJoinGate("audio_video_join", ["person_seen", "keyword"], join_window_ms=500)
    first = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"gate_fire": GateFire("person_seen", 0.8, "person", {"label": "person"})},
    )
    second = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1300,
        sequence_id=2,
        payload={"gate_fire": GateFire("keyword", 1.0, "said phrase", {"text": "hello"})},
    )

    assert await gate.process(first) is None
    fire = await gate.process(second)

    assert fire is not None
    assert fire.gate_id == "audio_video_join"
    assert fire.evidence["joined_gate_ids"] == ["keyword", "person_seen"]


async def test_temporal_join_propagates_latest_joined_frame() -> None:
    frame = object()
    gate = TemporalJoinGate("visual_join", ["visual_candidate", "motion_candidate"], join_window_ms=500)
    first = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"gate_fire": GateFire("visual_candidate", 0.8, "visual", {"frame": frame})},
    )
    second = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1200,
        sequence_id=2,
        payload={"gate_fire": GateFire("motion_candidate", 1.0, "motion", {"motion_ratio": 0.2})},
    )

    assert await gate.process(first) is None
    fire = await gate.process(second)

    assert fire is not None
    assert fire.evidence["frame"] is frame


async def test_transcript_extract_waits_for_full_streamed_answer() -> None:
    gate = ModelTranscriptQueryGate.__new__(ModelTranscriptQueryGate)
    gate.provider = StubProvider(deltas=["S", "oph", "ia"])
    gate.model = "gpt-5.6-luna"
    gate.query = "who brings the umbrella?"
    gate.verification_mode = "extract"
    gate.service_tier = "default"

    output = await gate._request_text("Sophia brought a blue umbrella.", {})

    assert output == "Sophia"
    request = gate.provider.request
    assert request.max_output_tokens == 24
    assert "Sophia brought a blue umbrella." in request.text()
    assert gate._parse_output(output) == (
        True,
        1.0,
        "Transcript verifier extracted value: Sophia.",
        {"text": "Sophia", "answer": "Sophia", "value": "Sophia"},
    )


def test_transcript_extract_normalizes_name_from_context() -> None:
    gate = ModelTranscriptQueryGate.__new__(ModelTranscriptQueryGate)
    gate.query = "who brings the blue umbrella?"
    gate.verification_mode = "extract"

    assert gate._parse_output(
        "Sofia",
        "Emma, Liam, Sophia, and Noah met at the park. Sofia brought a blue umbrella.",
    ) == (
        True,
        1.0,
        "Transcript verifier extracted value: Sophia.",
        {"text": "Sophia", "answer": "Sophia", "value": "Sophia"},
    )
    assert gate._parse_output(
        "Sophie",
        "Emma, Liam, Sophia, and Noah met at the park. Sophie brought a blue umbrella.",
    ) == (
        True,
        1.0,
        "Transcript verifier extracted value: Sophia.",
        {"text": "Sophia", "answer": "Sophia", "value": "Sophia"},
    )
