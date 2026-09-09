from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from meerkat import __version__
from meerkat.env import load_dotenv
from meerkat.events import EventType, StreamEvent
from meerkat.funnel.spec import FunnelSpec
from meerkat.ingest.sources import StreamSource, is_audio_only_media_path
from meerkat.models.provider import ModelBacked, resolve_provider_name
from meerkat.runtime.logging import RuntimeLogger
from meerkat.runtime.runner import StreamRunner
from meerkat.runtime.session import (
    FileTarget,
    LiveTarget,
    MonitorOptions,
    MonitorSession,
    MonitorTarget,
)
from meerkat.runtime.speech import NullSpeechSink, SpeechSink


MEDIA_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".aif",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mov",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
WEB_BUILD_ID = f"meerkat-{__version__}"
LIVE_CAMERA_MEDIA_ID = "__live_camera__"
LIVE_SCREEN_MEDIA_ID = "__live_screen__"
LIVE_MEDIA_IDS = {LIVE_CAMERA_MEDIA_ID, LIVE_SCREEN_MEDIA_ID}


class StartRequest(BaseModel):
    """What the browser sends to open a session.

    Everything but `media` and `speak` maps straight onto `MonitorOptions`, so
    the UI cannot end up with settings the CLI has no way to express.
    """

    media: str
    prompt: str
    planner_model: Optional[str] = None
    verifier_model: Optional[str] = None
    no_planner: bool = False
    initial_frame_context: bool = True
    model_responder: bool = False
    speak: bool = True
    vision_sample_seconds: float = 1.0
    verifier_sample_interval_ms: Optional[int] = None
    verifier_concurrent_requests: Optional[int] = 4
    verifier_min_request_interval_ms: Optional[int] = 200
    verifier_queued_frame_dedupe_window_ms: Optional[int] = 300

    def to_options(self) -> MonitorOptions:
        return MonitorOptions(
            use_planner=not self.no_planner,
            planner_model=self.planner_model,
            verifier_model=self.verifier_model,
            initial_frame_context=self.initial_frame_context,
            model_responder=self.model_responder,
            vision_sample_seconds=self.vision_sample_seconds,
            verifier_sample_interval_ms=self.verifier_sample_interval_ms,
            verifier_concurrent_requests=self.verifier_concurrent_requests,
            verifier_min_request_interval_ms=self.verifier_min_request_interval_ms,
            verifier_queued_frame_dedupe_window_ms=self.verifier_queued_frame_dedupe_window_ms,
        )


class PromptRequest(BaseModel):
    prompt: str


class BrowserLiveSource(StreamSource):
    def __init__(self, playback_event: asyncio.Event) -> None:
        self.playback_event = playback_event
        self._queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue(maxsize=32)
        self._closed = False

    async def push_frame(self, image_data_url: str, stream_time_ms: int, sequence_id: int) -> None:
        if self._closed:
            return
        frame = await asyncio.to_thread(_decode_data_url_frame, image_data_url)
        event = StreamEvent(
            type=EventType.VIDEO_FRAME,
            source_id="browser_live_camera",
            stream_time_ms=max(0, int(stream_time_ms)),
            sequence_id=max(0, int(sequence_id)),
            payload={"frame": frame, "objects": []},
        )
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(event)

    async def push_audio(
        self,
        samples_base64: str,
        stream_time_ms: int,
        sequence_id: int,
        sample_rate: int,
        chunk_ms: int,
    ) -> None:
        if self._closed:
            return
        samples = await asyncio.to_thread(_decode_audio_samples, samples_base64)
        event = StreamEvent(
            type=EventType.AUDIO_CHUNK,
            source_id="browser_live_microphone",
            stream_time_ms=max(0, int(stream_time_ms)),
            sequence_id=max(0, int(sequence_id)),
            payload={
                "samples": samples,
                "sample_rate": max(1, int(sample_rate)),
                "chunk_ms": max(1, int(chunk_ms)),
            },
        )
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(event)

    async def close(self) -> None:
        self._closed = True
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[StreamEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            if not self.playback_event.is_set():
                await self.playback_event.wait()
            yield event


class WebRuntimeLogger(RuntimeLogger):
    def __init__(self, session: "WebSession") -> None:
        super().__init__(enabled=True)
        self.session = session

    def log(self, stream_time_ms: Optional[int], message: str, bold: bool = False) -> None:
        """Mirror every runtime log line to the browser.

        User-facing responses are not scraped out of these lines; they arrive
        as typed MODEL_RESPONSE events on the session's event bus, the same way
        the CLI receives them.
        """
        self.session.emit_nowait(
            {
                "type": "log",
                "wall_ms": self.elapsed_ms(),
                "stream_time_ms": stream_time_ms,
                "message": message,
                "bold": bold,
            }
        )


class TTSStream:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=64)


class StreamingTTS(ModelBacked):
    """Stream provider speech audio to the browser as it is generated."""

    def __init__(self, session: "WebSession", model: Optional[str] = None, voice: str = "alloy") -> None:
        self.session = session
        self.model = model
        self.voice = voice
        self._streams: Dict[str, TTSStream] = {}

    def start(self, text: str, stream_time_ms: Optional[int]) -> str:
        stream_id = str(uuid4())
        stream = TTSStream()
        self._streams[stream_id] = stream
        loop = asyncio.get_running_loop()
        asyncio.create_task(
            asyncio.to_thread(self._produce, stream_id, stream, text, stream_time_ms, loop),
            name=f"tts-stream:{self.session.session_id}:{stream_id}",
        )
        return stream_id

    async def chunks(self, stream_id: str):
        stream = self._streams.get(stream_id)
        if stream is None:
            raise HTTPException(status_code=404, detail="TTS stream not found.")
        try:
            while True:
                chunk = await stream.queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            self._streams.pop(stream_id, None)

    def _produce(
        self,
        stream_id: str,
        stream: TTSStream,
        text: str,
        stream_time_ms: Optional[int],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            self.session.emit_nowait(
                {
                    "type": "log",
                    "wall_ms": self.session.logger.elapsed_ms(),
                    "stream_time_ms": stream_time_ms,
                    "message": f"Sending streaming TTS request model={self.model} voice={self.voice}",
                    "bold": False,
                }
            )
            for chunk in self.provider.stream_speech(text, model=self.model, voice=self.voice):
                asyncio.run_coroutine_threadsafe(stream.queue.put(chunk), loop).result()
        except Exception as exc:
            self.session.emit_nowait(
                {
                    "type": "log",
                    "wall_ms": self.session.logger.elapsed_ms(),
                    "stream_time_ms": stream_time_ms,
                    "message": f"Streaming TTS failed stream={stream_id} error={exc}",
                    "bold": False,
                }
            )
        finally:
            asyncio.run_coroutine_threadsafe(stream.queue.put(None), loop).result()


class WebSpeechSink(SpeechSink):
    def __init__(self, session: "WebSession", logger: RuntimeLogger) -> None:
        self.session = session

    async def speak(self, text: str, stream_time_ms: Optional[int] = None) -> Optional[Path]:
        cleaned = " ".join(text.strip().split())
        if not cleaned or self.session.tts is None:
            return None
        stream_id = self.session.tts.start(cleaned, stream_time_ms)
        self.session.emit_nowait(
            {
                "type": "audio",
                "wall_ms": self.session.logger.elapsed_ms(),
                "stream_time_ms": stream_time_ms,
                "text": cleaned,
                "url": f"/api/sessions/{self.session.session_id}/tts/{stream_id}",
            }
        )
        return None

class WebSession(MonitorSession):
    """A browser-facing monitoring session.

    Everything about planning and running is inherited from `MonitorSession`,
    so a UI session takes exactly the same code path as a CLI run. This class
    only adds what a browser needs on top: a target built from the request, an
    event queue the WebSocket drains, streamed text-to-speech, and pause /
    resume of playback.
    """

    def __init__(
        self,
        session_id: str,
        store: "MediaStore",
        media_path: Optional[Path],
        request: StartRequest,
    ) -> None:
        self.session_id = session_id
        self.store = store
        self.media_path = media_path
        self.request = request
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self.audio_dir = store.audio_dir(session_id)
        self._assistant_message_keys: set[tuple[Optional[int], str, str]] = set()
        self.run_task: Optional[asyncio.Task[None]] = None
        self.started = False
        self.tts: Optional[StreamingTTS] = None

        logger = WebRuntimeLogger(self)
        speech: SpeechSink = NullSpeechSink()
        if request.speak:
            self.tts = StreamingTTS(self)
            speech = WebSpeechSink(self, logger)
        super().__init__(
            target=self._build_target(),
            prompt=request.prompt,
            options=request.to_options(),
            logger=logger,
            speech=speech,
        )

    def _build_target(self) -> MonitorTarget:
        if self.request.media == LIVE_CAMERA_MEDIA_ID:
            return self._live_target("Source type: live webcam video with live microphone audio.")
        if self.request.media == LIVE_SCREEN_MEDIA_ID:
            return self._live_target(
                "Source type: live shared screen, browser tab, window, or website page "
                "with optional shared audio."
            )
        return FileTarget(str(self.media_path))

    def _live_target(self, description: str) -> MonitorTarget:
        return LiveTarget(lambda event: BrowserLiveSource(playback_event=event), description)

    @property
    def is_live(self) -> bool:
        return self.request.media in LIVE_MEDIA_IDS

    @property
    def live_source(self) -> Optional[BrowserLiveSource]:
        """The browser-fed source, once the session has started."""
        return self.source if isinstance(self.source, BrowserLiveSource) else None

    # -- browser plumbing -------------------------------------------------

    def emit_nowait(self, event: Dict[str, Any]) -> None:
        event = _json_safe(event)
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)

    async def start(self) -> StreamRunner:
        runner = await super().start()
        if not self.started:
            self.started = True
            self.run_task = asyncio.create_task(self._run(), name=f"web-run:{self.session_id}")
            self.emit_nowait({"type": "started"})
        return runner

    async def restart(self) -> None:
        """Ready this session to replay its funnel, without re-planning.

        Ingest does not begin here: the browser presses play and the usual
        "start" command follows, so playback and ingest stay on one clock.
        """
        if self.run_task:
            self.run_task.cancel()
            await asyncio.gather(self.run_task, return_exceptions=True)
            self.run_task = None
        self.started = False
        # Alerts from the previous run must not suppress the same alerts now.
        self._assistant_message_keys.clear()
        await super().restart()
        self.emit_nowait({"type": "restarted", "plan": _plan_payload(self.spec)})

    async def _run(self) -> None:
        try:
            await self.runner.run()
            self.emit_nowait({"type": "done", "metrics": self.runner.metrics.snapshot()})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.emit_nowait(
                {"type": "error", "wall_ms": self.logger.elapsed_ms(), "stream_time_ms": None, "message": str(exc)}
            )

    async def stop(self) -> None:
        self.resume()
        if self.live_source is not None:
            await self.live_source.close()
        if self.run_task:
            self.run_task.cancel()
            await asyncio.gather(self.run_task, return_exceptions=True)
            self.run_task = None
        await self.close()

    async def push_live_frame(self, image_data_url: str, stream_time_ms: int, sequence_id: int) -> None:
        if self.live_source is None:
            return
        await self.live_source.push_frame(image_data_url, stream_time_ms, sequence_id)

    async def push_live_audio(
        self,
        samples_base64: str,
        stream_time_ms: int,
        sequence_id: int,
        sample_rate: int,
        chunk_ms: int,
    ) -> None:
        if self.live_source is None:
            return
        await self.live_source.push_audio(samples_base64, stream_time_ms, sequence_id, sample_rate, chunk_ms)

    def pause(self) -> None:
        if self.playback_event.is_set():
            super().pause()
            self.logger.log(None, "Playback paused; realtime ingest paused")

    def resume(self) -> None:
        if not self.playback_event.is_set():
            super().resume()
            self.logger.log(None, "Playback resumed; realtime ingest resumed")

    # -- session hooks ----------------------------------------------------

    def on_planning(self) -> None:
        self.emit_nowait({"type": "planning"})

    def on_plan(self, spec: FunnelSpec) -> None:
        self.emit_nowait({"type": "plan", "plan": _plan_payload(spec)})

    def on_planned(self, spec: FunnelSpec) -> None:
        self.emit_nowait({"type": "plan_completed", "plan": _plan_payload(spec)})

    def on_user_prompt(self, prompt: str, stream_time_ms: int) -> None:
        self.emit_nowait(
            {
                "type": "user_instruction",
                "wall_ms": self.logger.elapsed_ms(),
                "stream_time_ms": stream_time_ms,
                "message": prompt,
            }
        )

    def on_assistant_message(
        self,
        text: str,
        stream_time_ms: Optional[int],
        trigger_gate_id: str,
        evidence: Dict[str, Any],
    ) -> None:
        key = (stream_time_ms, trigger_gate_id, text)
        if key in self._assistant_message_keys:
            return
        self._assistant_message_keys.add(key)
        self.emit_nowait(
            {
                "type": "user_message",
                "wall_ms": self.logger.elapsed_ms(),
                "stream_time_ms": stream_time_ms,
                "message": text,
                "role": "assistant",
                "trigger_gate_id": trigger_gate_id,
                "evidence": evidence,
            }
        )


class MediaStore:
    """Files the browser uploaded, plus the scratch space sessions write into.

    The server never browses the machine it runs on: a run watches either a
    file the user picked in the browser or one of the live sources. Uploads are
    keyed by an opaque id, so nothing a client sends is ever used as a path.
    """

    def __init__(self, workspace: Optional[Path] = None) -> None:
        self.workspace = (workspace or Path(tempfile.mkdtemp(prefix="meerkat-"))).resolve()
        self.uploads_dir = self.workspace / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._names: Dict[str, str] = {}

    def save(self, filename: str, source: BinaryIO) -> str:
        """Store an uploaded file and return its id."""
        suffix = Path(filename or "").suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported media type '{suffix or filename}'. "
                f"Supported: {', '.join(sorted(MEDIA_EXTENSIONS))}.",
            )
        media_id = f"{uuid4().hex}{suffix}"
        with (self.uploads_dir / media_id).open("wb") as handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
        self._names[media_id] = Path(filename).name
        return media_id

    def path(self, media_id: str) -> Path:
        path = self.uploads_dir / media_id
        if "/" in media_id or "\\" in media_id or not _is_relative_to(path, self.uploads_dir):
            raise HTTPException(status_code=400, detail="Invalid media id.")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Uploaded media not found.")
        return path

    def name(self, media_id: str) -> str:
        return self._names.get(media_id, media_id)

    def audio_dir(self, session_id: str) -> Path:
        path = self.workspace / "audio" / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


class SessionManager:
    def __init__(self, store: MediaStore) -> None:
        self.store = store
        self.sessions: Dict[str, WebSession] = {}

    async def create(self, request: StartRequest) -> WebSession:
        media_path = None if request.media in LIVE_MEDIA_IDS else self.store.path(request.media)
        session = WebSession(str(uuid4()), self.store, media_path, request)
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> WebSession:
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session


def create_app(workspace: Optional[Path] = None) -> FastAPI:
    """Build the app.

    `workspace` is scratch space the server writes uploads and speech into. It
    is never browsed for media: the user picks a file in the browser instead.
    """
    load_dotenv()
    static_dir = Path(__file__).resolve().parent / "web_static"
    store = MediaStore(workspace)
    manager = SessionManager(store)
    app = FastAPI(title="Meerkat")
    app.state.store = store
    app.state.manager = manager
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (static_dir / "index.html").read_text().replace("__BUILD__", _static_build_id(static_dir))

    @app.get("/api/version")
    async def version() -> Dict[str, Any]:
        build = _static_build_id(static_dir)
        try:
            provider = resolve_provider_name()
        except RuntimeError as exc:
            return {"build": build, "provider": None, "provider_error": str(exc)}
        return {"build": build, "provider": provider}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """Browsers ask for this path directly, whatever the page declares."""
        return FileResponse(static_dir / "favicon.png", media_type="image/png")

    @app.post("/api/media")
    async def upload_media(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Accept a file chosen in the browser and describe it back."""
        media_id = await asyncio.to_thread(store.save, file.filename or "", file.file)
        path = store.path(media_id)
        info = _media_info(path)
        return {
            "media_id": media_id,
            "name": store.name(media_id),
            "media_url": _media_url(media_id, info),
            "audio_only": is_audio_only_media_path(str(path)),
            "media_info": info,
        }

    @app.get("/api/media/{media_id}")
    async def media_file(media_id: str) -> FileResponse:
        return FileResponse(
            store.path(media_id),
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.post("/api/sessions")
    async def create_session(request: StartRequest) -> Dict[str, Any]:
        session = await manager.create(request)
        media_info = _live_media_info() if session.is_live else _media_info(session.media_path)
        return {
            "session_id": session.session_id,
            "build": WEB_BUILD_ID,
            "media_url": "" if session.is_live else _media_url(request.media, media_info),
            "name": "" if session.is_live else store.name(request.media),
            "audio_only": False if session.is_live else is_audio_only_media_path(str(session.media_path)),
            "live": session.is_live,
            "media_info": media_info,
        }

    @app.post("/api/sessions/{session_id}/prompts")
    async def update_prompt(session_id: str, request: PromptRequest) -> Dict[str, str]:
        session = manager.get(session_id)
        await session.add_prompt(request.prompt)
        return {"status": "ok"}

    @app.get("/api/sessions/{session_id}/audio/{file_name}")
    async def session_audio(session_id: str, file_name: str) -> FileResponse:
        session = manager.get(session_id)
        path = (session.audio_dir / file_name).resolve()
        if not _is_relative_to(path, session.audio_dir) or not path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found.")
        return FileResponse(path, media_type="audio/mpeg")

    @app.get("/api/sessions/{session_id}/tts/{stream_id}")
    async def session_tts(session_id: str, stream_id: str) -> StreamingResponse:
        session = manager.get(session_id)
        if session.tts is None:
            raise HTTPException(status_code=404, detail="TTS is not enabled for this session.")
        return StreamingResponse(session.tts.chunks(stream_id), media_type="audio/mpeg")

    @app.get("/api/sessions/{session_id}/frame")
    async def session_frame(session_id: str, time_ms: int) -> Response:
        session = manager.get(session_id)
        if session.is_live:
            raise HTTPException(status_code=400, detail="Live camera streams do not support frame preview rewind.")
        if is_audio_only_media_path(str(session.media_path)):
            raise HTTPException(status_code=400, detail="Audio-only media does not have video frames.")
        return Response(_extract_frame_jpeg(session.media_path, time_ms), media_type="image/jpeg")

    @app.websocket("/ws/{session_id}")
    async def websocket(websocket: WebSocket, session_id: str) -> None:
        session = manager.get(session_id)
        await websocket.accept()
        sender = asyncio.create_task(_send_ws_events(websocket, session))
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("type") == "plan":
                    await session.plan()
                elif data.get("type") == "start":
                    await session.start()
                elif data.get("type") == "restart":
                    await session.restart()
                elif data.get("type") == "pause":
                    session.pause()
                elif data.get("type") == "resume":
                    session.resume()
                elif data.get("type") == "prompt":
                    await session.add_prompt(str(data.get("prompt", "")))
                elif data.get("type") == "live_frame":
                    await session.push_live_frame(
                        str(data.get("image", "")),
                        int(data.get("stream_time_ms", 0) or 0),
                        int(data.get("sequence_id", 0) or 0),
                    )
                elif data.get("type") == "live_audio":
                    await session.push_live_audio(
                        str(data.get("samples", "")),
                        int(data.get("stream_time_ms", 0) or 0),
                        int(data.get("sequence_id", 0) or 0),
                        int(data.get("sample_rate", 16000) or 16000),
                        int(data.get("chunk_ms", 500) or 500),
                    )
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    return app


async def _send_ws_events(websocket: WebSocket, session: WebSession) -> None:
    while True:
        event = await session.queue.get()
        try:
            await websocket.send_json(event)
        except (TypeError, ValueError) as exc:
            # One unsendable payload must not silence the whole session: the
            # receive loop would keep accepting commands while the browser
            # never heard another word.
            await websocket.send_json(
                {
                    "type": "log",
                    "wall_ms": session.logger.elapsed_ms(),
                    "stream_time_ms": None,
                    "message": f"Dropped an unsendable {event.get('type')} event: {exc}",
                    "bold": False,
                }
            )


def _static_build_id(static_dir: Path) -> str:
    """Cache marker for the browser assets: the version plus a digest of them.

    Keying only on the package version means an edited or upgraded stylesheet
    or script keeps its old URL, so browsers go on serving the old copy. The
    digest covers every asset's name, size, and timestamp, so any change moves
    the marker — including a timestamp that moves backwards, which taking the
    newest mtime alone would miss.
    """
    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(static_dir.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns};".encode())
    return f"{WEB_BUILD_ID}-{digest.hexdigest()}"


def _json_safe(value: Any) -> Any:
    """Reduce a value to something that can be sent to the browser.

    Gate evidence carries working data — decoded frames, numpy arrays — that
    means nothing in JSON. Left alone, a single frame makes the whole event
    unserializable, which used to kill the send loop and leave the UI silent
    for the rest of the session.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0 and hasattr(value, "item"):
            return _json_safe(value.item())
        return f"<{type(value).__name__} {'x'.join(str(int(n)) for n in shape)}>"
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _plan_payload(spec: Optional[FunnelSpec]) -> Dict[str, Any]:
    if spec is None:
        return {}
    return {
        "goal": spec.goal,
        "gates": [_gate_payload(gate) for gate in spec.gates],
        "response": asdict(spec.response),
    }


def _gate_payload(gate: Any) -> Dict[str, Any]:
    payload = asdict(gate)
    if payload.get("type") == "local_realtime_transcription":
        params = dict(payload.get("params") or {})
        params["buffer_ms"] = min(int(params.get("buffer_ms", 2000) or 2000), 2000)
        params["sample_interval_ms"] = max(250, min(int(params.get("sample_interval_ms", 1000) or 1000), 1000))
        payload["params"] = params
    return payload


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _extract_frame_jpeg(path: Path, time_ms: int) -> bytes:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="Frame preview requires the uv-managed OpenCV dependency.") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video for frame preview.")
    try:
        target_ms = max(0, int(time_ms))
        capture.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
            ok, frame = capture.read()
        if not ok:
            raise HTTPException(status_code=404, detail="Could not extract frame preview.")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode frame preview.")
        return bytes(encoded)
    finally:
        capture.release()


def _decode_data_url_frame(image_data_url: str) -> object:
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Browser live frames require the uv-managed OpenCV and numpy dependencies.") from exc
    if "," in image_data_url:
        image_data_url = image_data_url.split(",", 1)[1]
    data = base64.b64decode(image_data_url, validate=True)
    encoded = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode browser live camera frame.")
    return frame


def _decode_audio_samples(samples_base64: str) -> object:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Browser live audio requires the uv-managed numpy dependency.") from exc
    data = base64.b64decode(samples_base64, validate=True)
    if not data:
        return np.asarray([], dtype="float32")
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0


def _media_info(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    audio_only = is_audio_only_media_path(str(path))
    info: Dict[str, Any] = {
        "audio_only": audio_only,
        "media_version": _media_version(path),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "video_duration_ms": None,
        "video_frame_count": None,
        "video_fps": None,
    }
    if audio_only:
        return info
    try:
        import cv2  # type: ignore
    except ImportError:
        return info
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return info
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and frame_count > 0:
            info["video_fps"] = fps
            info["video_frame_count"] = frame_count
            info["video_duration_ms"] = int(round((frame_count / fps) * 1000))
    finally:
        capture.release()
    demuxed_ms = _video_track_duration_ms(path)
    if demuxed_ms is not None:
        info["video_duration_ms"] = demuxed_ms
    return info


def _video_track_duration_ms(path: Path) -> Optional[int]:
    """How long the picture lasts, read from the container.

    OpenCV's frame count is an estimate for many files and can be wildly wrong,
    and a container's audio track often outlives its video track. Either one
    makes the UI clock run past the last frame, so ask the demuxer for the
    video stream's own duration.
    """
    try:
        import av  # type: ignore
    except ImportError:  # pragma: no cover - declared in pyproject
        return None
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is not None and stream.duration is not None and stream.time_base:
                return max(0, int(round(float(stream.duration * stream.time_base) * 1000)))
            if container.duration:
                return max(0, int(round(container.duration / 1000)))
    except Exception:
        return None
    return None


def _live_media_info() -> Dict[str, Any]:
    return {
        "audio_only": False,
        "live": True,
        "media_version": "live",
        "file_size": None,
        "mtime_ns": None,
        "video_duration_ms": None,
        "video_frame_count": None,
        "video_fps": None,
    }


def _media_version(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def _media_url(media_id: str, info: Dict[str, Any]) -> str:
    version = info.get("media_version")
    suffix = f"?v={version}" if version else ""
    return f"/api/media/{media_id}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="meerkat-web", description="Run the Meerkat web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--workspace",
        help="Directory for uploads and generated speech. Defaults to a temporary directory.",
    )
    args = parser.parse_args()
    import uvicorn

    workspace = Path(args.workspace).resolve() if args.workspace else None
    uvicorn.run(create_app(workspace=workspace), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
