"""One monitoring run, independent of how it was started.

`MonitorSession` owns the whole flow: describe the media to the planner,
compile the prompt into a funnel, build the runner, and re-plan when a new
prompt arrives mid-stream. The CLI and the web UI are both thin front ends
over this class — they choose a target, a logger, and a speech sink, and
otherwise take identical code paths, so the same prompt against the same media
produces the same funnel and the same responses either way.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

from meerkat.events import EventType, ModelResponse, StreamEvent
from meerkat.funnel.compiler import build_media_spec, initial_frame_context, log_plan
from meerkat.funnel.spec import FunnelSpec
from meerkat.ingest.sources import (
    StreamSource,
    VideoFileSource,
    is_audio_only_media_path,
    media_has_audio_stream,
)
from meerkat.models.config import ModelConfig
from meerkat.models.responders import LocalResponder, Responder, create_model_responder
from meerkat.runtime.logging import RuntimeLogger
from meerkat.runtime.runner import StreamRunner
from meerkat.runtime.speech import NullSpeechSink, SpeechSink


@dataclass(frozen=True)
class MonitorOptions:
    """Everything that shapes a run, other than the prompt and the media itself.

    Both front ends fill in the same object, so a UI session and a CLI
    invocation with matching options behave identically.
    """

    use_planner: bool = True
    planner_model: Optional[str] = None
    verifier_model: Optional[str] = None
    initial_frame_context: bool = True
    model_responder: bool = False
    vision_sample_seconds: float = 1.0
    verifier_sample_interval_ms: Optional[int] = None
    verifier_concurrent_requests: Optional[int] = None
    verifier_min_request_interval_ms: Optional[int] = None
    verifier_queued_frame_dedupe_window_ms: Optional[int] = None
    include_audio: bool = True
    audio_chunk_ms: int = 500
    realtime: bool = True

    def context_model(self) -> str:
        """Model used to summarize the first frame before planning."""
        return self.verifier_model or self.planner_model or ModelConfig.from_env().cheap_model


class MonitorTarget(ABC):
    """What a session watches, and how to describe it to the planner."""

    @property
    @abstractmethod
    def audio_only(self) -> bool:
        """True when there is no video to look at, so visual gates are pointless."""

    @property
    @abstractmethod
    def has_audio(self) -> bool:
        """True when audio is available, so transcript and audio gates are usable."""

    @abstractmethod
    def open(self, options: MonitorOptions, playback_event: asyncio.Event) -> StreamSource:
        """Build the stream source for a run."""

    def planner_context(self, options: MonitorOptions, logger: RuntimeLogger) -> Optional[str]:
        """Optional description of the source, given to the planner before it plans."""
        return None


class FileTarget(MonitorTarget):
    """A video or audio-only file on disk."""

    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def audio_only(self) -> bool:
        return is_audio_only_media_path(self.path)

    @property
    def has_audio(self) -> bool:
        return media_has_audio_stream(self.path)

    def open(self, options: MonitorOptions, playback_event: asyncio.Event) -> StreamSource:
        return VideoFileSource(
            self.path,
            realtime=options.realtime,
            include_audio=options.include_audio,
            audio_chunk_ms=options.audio_chunk_ms,
            playback_event=playback_event,
        )

    def planner_context(self, options: MonitorOptions, logger: RuntimeLogger) -> Optional[str]:
        if not options.initial_frame_context or self.audio_only or not options.use_planner:
            return None
        try:
            return initial_frame_context(self.path, options.context_model(), logger)
        except Exception as exc:
            logger.log(None, f"Initial frame context failed; planning without it error={exc}")
            return None


class LiveTarget(MonitorTarget):
    """A live source fed from outside the session, such as a browser or a camera.

    There is no first frame to inspect before planning, so the planner is told
    what kind of source it is getting instead.
    """

    def __init__(self, factory: Callable[[asyncio.Event], StreamSource], description: str) -> None:
        self.factory = factory
        self.description = description

    @property
    def audio_only(self) -> bool:
        return False

    @property
    def has_audio(self) -> bool:
        return True

    def open(self, options: MonitorOptions, playback_event: asyncio.Event) -> StreamSource:
        return self.factory(playback_event)

    def planner_context(self, options: MonitorOptions, logger: RuntimeLogger) -> Optional[str]:
        return self.description


def acknowledgement_text(prompt: str) -> str:
    """Short confirmation echoed back as soon as a prompt is accepted."""
    cleaned = prompt.strip().rstrip(".!?")
    if not cleaned:
        return "Okay."
    if len(cleaned) > 90:
        cleaned = cleaned[:87].rstrip() + "..."
    return f"Okay, I will watch for: {cleaned}."


class MonitorSession:
    """Plan a funnel for a prompt, run it against a target, and keep it current."""

    def __init__(
        self,
        target: MonitorTarget,
        prompt: str,
        options: Optional[MonitorOptions] = None,
        logger: Optional[RuntimeLogger] = None,
        speech: Optional[SpeechSink] = None,
    ) -> None:
        self.target = target
        self.prompt = prompt
        self.options = options or MonitorOptions()
        self.logger = logger or RuntimeLogger(enabled=False)
        self.speech = speech or NullSpeechSink()
        self.playback_event = asyncio.Event()
        self.spec: Optional[FunnelSpec] = None
        self.source: Optional[StreamSource] = None
        self.runner: Optional[StreamRunner] = None
        self.initial_context: Optional[str] = None
        self._planned = False
        self._response_task: Optional[asyncio.Task] = None

    # -- planning ---------------------------------------------------------

    async def plan(self) -> FunnelSpec:
        """Compile the initial prompt into a funnel. Idempotent."""
        if self._planned and self.spec is not None:
            return self.spec
        self.on_planning()
        self.initial_context = await asyncio.to_thread(
            self.target.planner_context, self.options, self.logger
        )
        self.spec = await self._compile(self.prompt)
        self._planned = True
        self.on_planned(self.spec)
        await self._acknowledge(self.prompt, stream_time_ms=None)
        return self.spec

    async def _compile(self, prompt: str) -> FunnelSpec:
        spec = await asyncio.to_thread(
            build_media_spec,
            prompt,
            audio_only_media=self.target.audio_only,
            use_planner=self.options.use_planner,
            planner_model=self.options.planner_model,
            verifier_model=self.options.verifier_model,
            vision_sample_seconds=self.options.vision_sample_seconds,
            verifier_sample_interval_ms=self.options.verifier_sample_interval_ms,
            verifier_concurrent_requests=self.options.verifier_concurrent_requests,
            verifier_min_request_interval_ms=self.options.verifier_min_request_interval_ms,
            verifier_queued_frame_dedupe_window_ms=self.options.verifier_queued_frame_dedupe_window_ms,
            logger=self.logger,
            initial_context=self.initial_context,
            media_has_audio=self.options.include_audio and self.target.has_audio,
        )
        log_plan(self.logger, spec, self.options.planner_model if self.options.use_planner else None)
        self.on_plan(spec)
        return spec

    # -- running ----------------------------------------------------------

    async def start(self) -> StreamRunner:
        """Build the source and runner. Does not consume the stream yet."""
        if self.runner is not None:
            return self.runner
        spec = await self.plan()
        self.playback_event.set()
        self.source = self.target.open(self.options, self.playback_event)
        await self.source.preload()
        self.runner = StreamRunner(
            source=self.source,
            spec=spec,
            responder=self._responder(),
            logger=self.logger,
            initial_state={"initial_visual_context": self.initial_context} if self.initial_context else None,
        )
        # Subscribe before anything can publish, so an alert on the very first
        # frame is still delivered.
        queue = self.runner.bus.subscribe([EventType.MODEL_RESPONSE])
        self._response_task = asyncio.create_task(
            self._forward_responses(queue), name="session:responses"
        )
        return self.runner

    async def run(self) -> List[ModelResponse]:
        """Start if needed, then consume the stream to completion."""
        runner = await self.start()
        try:
            return await runner.run()
        finally:
            await self.close()

    async def restart(self) -> None:
        """Discard the last run so the funnel can be run again, without re-planning.

        Planning costs a model call and, for a file, a frame summary too. Once
        a prompt has been compiled, replaying the same media should not pay
        that again. Everything the previous run accumulated is dropped: the
        source, gates, counts, cooldowns, and the wall clock, so the next run
        is measured exactly like the first.

        This only resets; it does not begin ingesting. The caller starts the
        replay with `start()` when it is ready, which is what keeps a browser's
        playback and the server's ingest on the same clock.
        """
        if self.spec is None:
            return
        await self.close()
        self.runner = None
        self.source = None
        self.playback_event = asyncio.Event()
        self.logger.reset_wall_clock()

    async def close(self) -> None:
        """Stop forwarding responses. Safe to call more than once."""
        task, self._response_task = self._response_task, None
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _forward_responses(self, queue: "asyncio.Queue[StreamEvent]") -> None:
        """Turn every model response into a user-facing message and, if enabled, speech."""
        try:
            while True:
                event = await queue.get()
                response = event.payload.get("model_response")
                if not isinstance(response, ModelResponse):
                    continue
                self.on_assistant_message(
                    response.text,
                    event.stream_time_ms,
                    response.trigger_gate_id or "response",
                    response.evidence,
                )
                await self.speech.speak(response.text, event.stream_time_ms)
        finally:
            if self.runner:
                self.runner.bus.unsubscribe(queue)

    def _responder(self) -> Responder:
        return create_model_responder() if self.options.model_responder else LocalResponder()

    def pause(self) -> None:
        """Stop pulling from the source; queued model work keeps draining."""
        self.playback_event.clear()
        self.logger.pause()

    def resume(self) -> None:
        self.playback_event.set()
        self.logger.resume()

    # -- live prompts -----------------------------------------------------

    async def add_prompt(self, prompt: str) -> FunnelSpec:
        """Accept a new prompt mid-stream and swap in a freshly compiled funnel."""
        if self.runner is None:
            raise RuntimeError("Session has not started.")
        stream_time_ms = self.runner.stream_time_ms
        self.logger.log(stream_time_ms, f"USER PROMPT: {prompt}", bold=True)
        self.on_user_prompt(prompt, stream_time_ms)
        await self._acknowledge(prompt, stream_time_ms=stream_time_ms)
        await self.runner.bus.publish(
            StreamEvent(
                type=EventType.USER_PROMPT,
                source_id="user",
                stream_time_ms=stream_time_ms,
                sequence_id=0,
                payload={"text": prompt},
            )
        )
        self.prompt = prompt
        self.spec = await self._compile(prompt)
        await self.runner.runtime.replace_spec(self.spec)
        self.logger.log(self.runner.stream_time_ms, "Realtime funnel updated from user prompt")
        return self.spec

    async def _acknowledge(self, prompt: str, stream_time_ms: Optional[int]) -> None:
        ack = acknowledgement_text(prompt)
        self.logger.log(stream_time_ms, f"ASSISTANT ACK: {ack}", bold=True)
        self.on_assistant_message(ack, stream_time_ms, "ack", {})
        await self.speech.speak(ack, stream_time_ms)

    # -- front-end hooks --------------------------------------------------
    #
    # Overridden by front ends that need to surface progress. The defaults do
    # nothing, so the core flow stays identical whether or not anyone is
    # watching.

    def on_planning(self) -> None:
        """Planning has begun."""

    def on_plan(self, spec: FunnelSpec) -> None:
        """A funnel was compiled, either initially or from a later prompt."""

    def on_planned(self, spec: FunnelSpec) -> None:
        """The initial funnel is ready and the run can start."""

    def on_user_prompt(self, prompt: str, stream_time_ms: int) -> None:
        """A new prompt was accepted mid-stream."""

    def on_assistant_message(
        self,
        text: str,
        stream_time_ms: Optional[int],
        trigger_gate_id: str,
        evidence: dict,
    ) -> None:
        """A user-facing message was produced, either an acknowledgement or an alert."""
