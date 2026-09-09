from __future__ import annotations

import asyncio
import queue
from pathlib import Path
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple

from meerkat.events import EventType, StreamEvent


_AUDIO_ONLY_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".aif",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


class StreamSource(ABC):
    @abstractmethod
    async def events(self) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

    async def preload(self) -> None:
        """Do any decoding that should happen before the stream clock starts."""
        return None


class SyntheticStreamSource(StreamSource):
    """Deterministic realtime-ish source for demos and tests.

    Frames carry labels in payload["objects"] so object gates can be exercised
    without installing a detector. Audio transcript events are optional.
    """

    def __init__(
        self,
        source_id: str = "synthetic",
        fps: float = 5.0,
        duration_seconds: float = 4.0,
        object_schedule: Optional[Dict[int, Iterable[str]]] = None,
        transcript_schedule: Optional[Dict[int, str]] = None,
        realtime: bool = True,
    ) -> None:
        self.source_id = source_id
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.object_schedule = object_schedule or {}
        self.transcript_schedule = transcript_schedule or {}
        self.realtime = realtime

    async def events(self) -> AsyncIterator[StreamEvent]:
        frame_count = int(self.duration_seconds * self.fps)
        frame_interval_ms = 1000.0 / self.fps
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        for seq in range(frame_count):
            stream_time_ms = int(round(seq * frame_interval_ms))
            objects: List[str] = list(self.object_schedule.get(seq, []))
            yield StreamEvent(
                type=EventType.VIDEO_FRAME,
                source_id=self.source_id,
                stream_time_ms=stream_time_ms,
                sequence_id=seq,
                payload={"objects": objects, "frame": None},
            )
            if seq in self.transcript_schedule:
                yield StreamEvent(
                    type=EventType.TRANSCRIPT_DELTA,
                    source_id=self.source_id,
                    stream_time_ms=stream_time_ms,
                    sequence_id=seq,
                    payload={"text": self.transcript_schedule[seq], "is_final": True},
                )
            if self.realtime:
                next_time = start_time + (((seq + 1) * frame_interval_ms) / 1000)
                await asyncio.sleep(max(0.0, next_time - loop.time()))


class VideoFileSource(StreamSource):
    """Realtime media file source for .mp4 video/audio or audio-only files.

    OpenCV is used for video frames. ffmpeg is used for audio chunks. If a file
    has audio but no decodable video, this source emits audio chunks only.
    """

    def __init__(
        self,
        path: str,
        source_id: str = "video_file",
        realtime: bool = True,
        include_audio: bool = True,
        audio_chunk_ms: int = 500,
        audio_sample_rate: int = 16000,
        playback_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.path = path
        self.source_id = source_id
        self.realtime = realtime
        self.include_audio = include_audio
        self.audio_chunk_ms = audio_chunk_ms
        self.audio_sample_rate = audio_sample_rate
        self.playback_event = playback_event
        self._audio_chunks_cache: Optional[List[Tuple[int, object]]] = None

    async def preload(self) -> None:
        if self.include_audio and self._audio_chunks_cache is None:
            self._audio_chunks_cache = await asyncio.to_thread(self._load_audio_chunks)

    async def events(self) -> AsyncIterator[StreamEvent]:
        if self._is_audio_only_path():
            audio_chunks = await self._audio_chunks()
            if not audio_chunks:
                raise RuntimeError(f"Could not open audio file: {self.path}")
            async for event in self._audio_only_events(audio_chunks):
                yield event
            return

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("VideoFileSource requires the 'media' optional dependencies.") from exc

        audio_chunks = await self._audio_chunks()
        audio_index = 0
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            if audio_chunks:
                async for event in self._audio_only_events(audio_chunks):
                    yield event
                return
            raise RuntimeError(f"Could not open media file: {self.path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval_ms = 1000.0 / fps
        sequence_id = 0
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        try:
            while True:
                if self.realtime:
                    start_time = await _wait_for_playback(self.playback_event, start_time, loop)
                ok, frame = capture.read()
                if not ok:
                    break
                stream_time_ms = int(round(sequence_id * frame_interval_ms))
                while audio_index < len(audio_chunks) and audio_chunks[audio_index][0] <= stream_time_ms:
                    audio_time_ms, samples = audio_chunks[audio_index]
                    yield StreamEvent(
                        type=EventType.AUDIO_CHUNK,
                        source_id=self.source_id,
                        stream_time_ms=audio_time_ms,
                        sequence_id=audio_index,
                        payload={
                            "samples": samples,
                            "sample_rate": self.audio_sample_rate,
                            "chunk_ms": self.audio_chunk_ms,
                        },
                    )
                    audio_index += 1
                yield StreamEvent(
                    type=EventType.VIDEO_FRAME,
                    source_id=self.source_id,
                    stream_time_ms=stream_time_ms,
                    sequence_id=sequence_id,
                    payload={"frame": frame, "objects": []},
                )
                sequence_id += 1
                if self.realtime:
                    start_time = await _sleep_until_stream_time(
                        sequence_id * frame_interval_ms,
                        start_time,
                        loop,
                        self.playback_event,
                    )
            while audio_index < len(audio_chunks):
                if self.realtime:
                    start_time = await _wait_for_playback(self.playback_event, start_time, loop)
                audio_time_ms, samples = audio_chunks[audio_index]
                yield StreamEvent(
                    type=EventType.AUDIO_CHUNK,
                    source_id=self.source_id,
                    stream_time_ms=audio_time_ms,
                    sequence_id=audio_index,
                    payload={
                        "samples": samples,
                        "sample_rate": self.audio_sample_rate,
                        "chunk_ms": self.audio_chunk_ms,
                    },
                )
                audio_index += 1
        finally:
            capture.release()

    async def _audio_chunks(self) -> List[Tuple[int, object]]:
        if not self.include_audio:
            return []
        if self._audio_chunks_cache is None:
            self._audio_chunks_cache = await asyncio.to_thread(self._load_audio_chunks)
        return self._audio_chunks_cache

    async def _audio_only_events(self, audio_chunks: List[Tuple[int, object]]) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        for audio_index, (audio_time_ms, samples) in enumerate(audio_chunks):
            if self.realtime:
                start_time = await _wait_for_playback(self.playback_event, start_time, loop)
            yield StreamEvent(
                type=EventType.AUDIO_CHUNK,
                source_id=self.source_id,
                stream_time_ms=audio_time_ms,
                sequence_id=audio_index,
                payload={
                    "samples": samples,
                    "sample_rate": self.audio_sample_rate,
                    "chunk_ms": self.audio_chunk_ms,
                },
            )
            if self.realtime:
                start_time = await _sleep_until_stream_time(
                    audio_time_ms + self.audio_chunk_ms,
                    start_time,
                    loop,
                    self.playback_event,
                )

    def _load_audio_chunks(self) -> List[Tuple[int, object]]:
        try:
            import av  # type: ignore
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Audio media decoding requires the uv-managed 'av' dependency.") from exc
        try:
            container = av.open(self.path)
        except Exception:
            return []
        try:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                return []
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.audio_sample_rate,
            )
            arrays = []
            for packet in container.demux(audio_streams[0]):
                for frame in packet.decode():
                    for resampled in _as_list(resampler.resample(frame)):
                        array = resampled.to_ndarray()
                        arrays.append(np.asarray(array).reshape(-1))
            if not arrays:
                return []
            pcm = np.concatenate(arrays).astype("float32") / 32768.0
            samples_per_chunk = max(1, int(self.audio_sample_rate * self.audio_chunk_ms / 1000))
            chunks: List[Tuple[int, object]] = []
            for index, start in enumerate(range(0, len(pcm), samples_per_chunk)):
                chunk = pcm[start : start + samples_per_chunk]
                if len(chunk) == 0:
                    continue
                chunks.append((index * self.audio_chunk_ms, chunk))
            return chunks
        finally:
            container.close()

    def _is_audio_only_path(self) -> bool:
        return is_audio_only_media_path(self.path)


class LiveCameraSource(StreamSource):
    """Realtime source for the default webcam and optional microphone."""

    def __init__(
        self,
        camera_index: int = 0,
        source_id: str = "live_camera",
        fps: float = 10.0,
        include_audio: bool = True,
        audio_chunk_ms: int = 500,
        audio_sample_rate: int = 16000,
        playback_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.camera_index = camera_index
        self.source_id = source_id
        self.fps = fps
        self.include_audio = include_audio
        self.audio_chunk_ms = audio_chunk_ms
        self.audio_sample_rate = audio_sample_rate
        self.playback_event = playback_event

    async def events(self) -> AsyncIterator[StreamEvent]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("LiveCameraSource requires the uv-managed OpenCV dependency.") from exc

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open live camera index {self.camera_index}.")

        loop = asyncio.get_running_loop()
        start_time = loop.time()
        frame_interval_ms = 1000.0 / max(1.0, self.fps)
        sequence_id = 0
        audio_source = _LiveMicrophoneCapture(
            enabled=self.include_audio,
            sample_rate=self.audio_sample_rate,
            chunk_ms=self.audio_chunk_ms,
        )
        audio_source.start()
        try:
            while True:
                start_time = await _wait_for_playback(self.playback_event, start_time, loop)
                stream_time_ms = int(round((loop.time() - start_time) * 1000))
                for audio_time_ms, samples in audio_source.drain(stream_time_ms):
                    yield StreamEvent(
                        type=EventType.AUDIO_CHUNK,
                        source_id=self.source_id,
                        stream_time_ms=audio_time_ms,
                        sequence_id=audio_time_ms // max(1, self.audio_chunk_ms),
                        payload={
                            "samples": samples,
                            "sample_rate": self.audio_sample_rate,
                            "chunk_ms": self.audio_chunk_ms,
                        },
                    )
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok:
                    await asyncio.sleep(0.05)
                    continue
                yield StreamEvent(
                    type=EventType.VIDEO_FRAME,
                    source_id=self.source_id,
                    stream_time_ms=stream_time_ms,
                    sequence_id=sequence_id,
                    payload={"frame": frame, "objects": []},
                )
                sequence_id += 1
                await _sleep_until_stream_time(
                    (sequence_id * frame_interval_ms),
                    start_time,
                    loop,
                    self.playback_event,
                )
        finally:
            audio_source.stop()
            capture.release()


class _LiveMicrophoneCapture:
    def __init__(self, enabled: bool, sample_rate: int, chunk_ms: int) -> None:
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._queue: "queue.Queue[object]" = queue.Queue()
        self._stream = None
        self._started = False
        self._chunk_index = 0

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return

        def callback(indata, _frames, _time, status) -> None:
            if status:
                return
            self._queue.put(indata.copy().reshape(-1))

        try:
            blocksize = max(1, int(self.sample_rate * self.chunk_ms / 1000))
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
            self._started = True
        except Exception:
            self._stream = None

    def drain(self, latest_stream_time_ms: int) -> List[Tuple[int, object]]:
        if not self._started:
            return []
        chunks: List[Tuple[int, object]] = []
        while True:
            try:
                samples = self._queue.get_nowait()
            except queue.Empty:
                return chunks
            audio_time_ms = min(latest_stream_time_ms, self._chunk_index * self.chunk_ms)
            chunks.append((audio_time_ms, samples))
            self._chunk_index += 1

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def is_audio_only_media_path(path: str) -> bool:
    return Path(path).suffix.lower() in _AUDIO_ONLY_EXTENSIONS


def media_has_audio_stream(path: str) -> bool:
    if is_audio_only_media_path(path):
        return True
    try:
        import av  # type: ignore
    except ImportError:
        return False
    try:
        container = av.open(path)
    except Exception:
        return False
    try:
        return any(stream.type == "audio" for stream in container.streams)
    finally:
        container.close()


async def _wait_for_playback(
    playback_event: Optional[asyncio.Event],
    start_time: float,
    loop: asyncio.AbstractEventLoop,
) -> float:
    if playback_event is None or playback_event.is_set():
        return start_time
    paused_at = loop.time()
    await playback_event.wait()
    return start_time + (loop.time() - paused_at)


async def _sleep_until_stream_time(
    stream_time_ms: float,
    start_time: float,
    loop: asyncio.AbstractEventLoop,
    playback_event: Optional[asyncio.Event],
) -> float:
    while True:
        start_time = await _wait_for_playback(playback_event, start_time, loop)
        delay = start_time + (stream_time_ms / 1000) - loop.time()
        if delay <= 0:
            return start_time
        if playback_event is None:
            await asyncio.sleep(delay)
            return start_time
        await asyncio.sleep(min(delay, 0.05))


def load_initial_video_frame(path: str) -> Optional[object]:
    if is_audio_only_media_path(path):
        return None
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Initial video frame context requires the uv-managed OpenCV dependency.") from exc
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return None
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()
