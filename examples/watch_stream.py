"""Embed Meerkat in your own program.

This skips the planner and wires the funnel by hand: cheap local motion
detection proposes candidate frames, and a single model verifier decides
whether the requested event actually happened. Responses are consumed from
the event bus as they arrive, so you can forward them anywhere.

    uv run python examples/watch_stream.py examples/media/dog-beach-ball.mp4 \
        "Is the dog holding the beach ball in its mouth?"
"""

from __future__ import annotations

import asyncio
import sys

from meerkat.events import EventType, ModelResponse
from meerkat.funnel.spec import FunnelSpec
from meerkat.ingest.sources import VideoFileSource
from meerkat.models.responders import LocalResponder
from meerkat.runtime.logging import RuntimeLogger
from meerkat.runtime.runner import StreamRunner


async def watch(media: str, question: str) -> None:
    spec = FunnelSpec.video_query(
        goal=question,
        on_match_text="Condition met.",
        sample_interval_ms=500,
    )
    runner = StreamRunner(
        source=VideoFileSource(media),
        spec=spec,
        responder=LocalResponder(),
        logger=RuntimeLogger(),
    )

    responses = runner.bus.subscribe([EventType.MODEL_RESPONSE])
    consumer = asyncio.create_task(_forward(responses))
    try:
        await runner.run()
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)


async def _forward(queue: asyncio.Queue) -> None:
    """Stand in for whatever your application does with an alert."""
    while True:
        event = await queue.get()
        response = event.payload.get("model_response")
        if isinstance(response, ModelResponse):
            print(f"[{event.stream_time_ms} ms] {response.text}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <media-file> <question>")
    asyncio.run(watch(sys.argv[1], sys.argv[2]))
