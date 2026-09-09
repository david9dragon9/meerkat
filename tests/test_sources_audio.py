from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from meerkat.events import EventType
from meerkat.ingest.sources import LiveCameraSource, VideoFileSource, _LiveMicrophoneCapture


def _write_wav(path: Path, sample_rate: int = 8000) -> None:
    samples = (np.sin(np.linspace(0, np.pi * 8, sample_rate)) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def test_media_source_splits_pyav_audio_into_chunks(tmp_path) -> None:
    audio_path = tmp_path / "fake.wav"
    _write_wav(audio_path)
    source = VideoFileSource(str(audio_path), audio_chunk_ms=500, audio_sample_rate=8000)

    chunks = source._load_audio_chunks()

    assert [time_ms for time_ms, _ in chunks] == [0, 500]
    assert len(chunks[0][1]) == 4000


async def test_audio_extension_source_emits_audio_without_importing_cv2(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "fake.mp3"
    _write_wav(audio_path)
    source = VideoFileSource(str(audio_path), realtime=False, audio_chunk_ms=500, audio_sample_rate=8000)

    def fail_import(name, *args, **kwargs):
        if name == "cv2":
            raise AssertionError("cv2 should not be imported for audio-only extensions")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)

    events = [event async for event in source.events()]

    assert [event.type for event in events] == [EventType.AUDIO_CHUNK, EventType.AUDIO_CHUNK]
    assert [event.stream_time_ms for event in events] == [0, 500]


async def test_live_camera_source_preload_is_noop() -> None:
    source = LiveCameraSource(include_audio=False)

    assert await source.preload() is None


def test_disabled_live_microphone_capture_drains_no_chunks() -> None:
    capture = _LiveMicrophoneCapture(enabled=False, sample_rate=16000, chunk_ms=500)

    capture.start()

    assert capture.drain(1000) == []
