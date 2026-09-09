from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from meerkat.benchmarks.metrics import Metrics
from meerkat.events import EventType, ModelResponse, StreamEvent
from meerkat.funnel.runtime import FunnelRuntime
from meerkat.funnel.spec import FunnelSpec
from meerkat.ingest.sources import StreamSource
from meerkat.models.responders import Responder
from meerkat.runtime.event_bus import EventBus
from meerkat.runtime.logging import RuntimeLogger


class StreamRunner:
    def __init__(
        self,
        source: StreamSource,
        spec: FunnelSpec,
        responder: Responder,
        logger: Optional[RuntimeLogger] = None,
        initial_state: Optional[dict[str, Any]] = None,
    ) -> None:
        self.source = source
        self.spec = spec
        self.bus = EventBus()
        self.metrics = Metrics()
        self.logger = logger
        #: Stream time of the most recent ingested event, for anything that
        #: needs to stamp "where are we in the media" rather than wall time.
        self.stream_time_ms = 0
        self.runtime = FunnelRuntime(
            spec=spec,
            bus=self.bus,
            responder=responder,
            metrics=self.metrics,
            logger=logger,
            initial_state=initial_state,
        )

    async def run(self) -> List[ModelResponse]:
        responses: List[ModelResponse] = []
        response_queue = self.bus.subscribe([EventType.MODEL_RESPONSE])
        await self.runtime.warmup()
        handle = self.runtime.start()
        response_task = asyncio.create_task(self._collect_responses(response_queue, responses))
        first_event = True
        try:
            async for event in self.source.events():
                if first_event:
                    if self.logger:
                        self.logger.reset_wall_clock()
                        self.logger.log(event.stream_time_ms, f"{_stream_label(event.type)} starts playing")
                    first_event = False
                self.stream_time_ms = event.stream_time_ms
                self.metrics.inc(f"ingest.{event.type.value}")
                await self.bus.publish(event)
            if self.logger:
                self.logger.log(self.stream_time_ms, "Input stream ended")
            await self.bus.publish(
                StreamEvent(
                    type=EventType.STREAM_END,
                    source_id="runner",
                    stream_time_ms=self.stream_time_ms,
                    sequence_id=0,
                    payload={},
                )
            )
            if self.logger:
                self.logger.log(self.stream_time_ms, "Waiting for model queue to drain")
            drained = await self.runtime.drain()
            if self.logger:
                status = "drained" if drained else "drain timed out"
                self.logger.log(self.stream_time_ms, f"Model queue {status}")
        finally:
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
            await handle.stop()
        return responses

    async def _collect_responses(self, queue: asyncio.Queue[StreamEvent], responses: List[ModelResponse]) -> None:
        while True:
            event = await queue.get()
            response = event.payload.get("model_response")
            if isinstance(response, ModelResponse):
                responses.append(response)


def _stream_label(event_type: EventType) -> str:
    if event_type == EventType.AUDIO_CHUNK:
        return "Audio"
    if event_type == EventType.VIDEO_FRAME:
        return "Video"
    return "Input stream"
