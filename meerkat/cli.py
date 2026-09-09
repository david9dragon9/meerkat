"""Command-line front end.

This module only parses flags and prints results. All of the actual work --
planning, source setup, running, re-planning on a new prompt -- happens in
`MonitorSession`, which the web UI drives the same way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import List, Optional

from meerkat.env import load_dotenv
from meerkat.events import ModelResponse
from meerkat.runtime.logging import RuntimeLogger
from meerkat.runtime.session import FileTarget, MonitorOptions, MonitorSession
from meerkat.runtime.speech import NullSpeechSink, ModelSpeechSink, SpeechSink


async def _read_prompts_from_stdin(session: MonitorSession) -> None:
    """Feed extra prompts typed at the terminal into the running session."""
    while True:
        try:
            text = await asyncio.to_thread(input, "")
        except EOFError:
            return
        prompt = text.strip()
        if prompt:
            await session.add_prompt(prompt)


async def run(
    prompt: str,
    media: str,
    *,
    options: Optional[MonitorOptions] = None,
    log: bool = True,
    live_prompts: bool = False,
    speak: bool = False,
    speak_play: bool = False,
    tts_model: str = "tts-1",
    tts_voice: str = "alloy",
) -> List[ModelResponse]:
    """Watch a media file for ``prompt`` and print every response the funnel produces."""
    load_dotenv()
    logger = RuntimeLogger(enabled=log)
    speech: SpeechSink = (
        ModelSpeechSink(model=tts_model, voice=tts_voice, play_audio=speak_play, logger=logger)
        if speak
        else NullSpeechSink()
    )
    session = MonitorSession(
        target=FileTarget(media),
        prompt=prompt,
        options=options,
        logger=logger,
        speech=speech,
    )

    await session.start()
    prompt_task = (
        asyncio.create_task(_read_prompts_from_stdin(session), name="cli:prompts") if live_prompts else None
    )
    try:
        responses = await session.run()
    finally:
        if prompt_task:
            prompt_task.cancel()
            await asyncio.gather(prompt_task, return_exceptions=True)

    for response in responses:
        print(f"\033[1m{response.text}\033[0m")
    print(json.dumps(session.runner.metrics.snapshot(), indent=2, sort_keys=True))
    return responses


def _resolve_ms(ms_value: Optional[int], seconds_value: Optional[float]) -> Optional[int]:
    if ms_value is not None:
        return ms_value
    if seconds_value is not None:
        return int(seconds_value * 1000)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meerkat",
        description="Watch a video or audio stream and respond when a plain-English request is satisfied.",
    )
    parser.add_argument("prompt", help='What to watch for, e.g. "let me know when a person walks in".')
    parser.add_argument("--media", required=True, help="Media file to monitor (video, or an audio-only file).")

    planning = parser.add_argument_group("planning")
    planning.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip the planner and send the prompt straight to a single model verifier.",
    )
    planning.add_argument("--planner-model", help="Model used to plan the funnel. Defaults to MEERKAT_PLANNER_MODEL.")
    planning.add_argument("--verifier-model", help="Model used by the realtime verifier gate.")
    planning.add_argument(
        "--no-initial-frame-context",
        action="store_true",
        help="Do not summarize the first video frame before planning.",
    )
    planning.add_argument(
        "--model-responder",
        action="store_true",
        help="Phrase the final response with a model instead of the funnel's response template.",
    )

    sampling = parser.add_argument_group("sampling and queueing")
    sampling.add_argument("--vision-sample-seconds", type=float, default=1.0, help="Default verifier sampling period.")
    sampling.add_argument(
        "--verifier-sample-interval-ms",
        type=int,
        help="Minimum stream-time spacing between verifier samples, in milliseconds.",
    )
    sampling.add_argument("--verifier-sample-interval-seconds", type=float, help="Same, in seconds.")
    sampling.add_argument(
        "--verifier-concurrent-requests",
        type=int,
        help="Number of verifier requests allowed in flight at once.",
    )
    sampling.add_argument(
        "--verifier-min-request-interval-ms",
        type=int,
        help="Minimum wall-clock spacing between verifier requests, in milliseconds.",
    )
    sampling.add_argument("--verifier-min-request-interval-seconds", type=float, help="Same, in seconds.")
    sampling.add_argument(
        "--verifier-queued-frame-dedupe-window-ms",
        type=int,
        help="Drop a queued verifier frame this close to an already-sent frame, in milliseconds.",
    )
    sampling.add_argument("--verifier-queued-frame-dedupe-seconds", type=float, help="Same, in seconds.")

    source = parser.add_argument_group("source")
    source.add_argument("--no-audio", action="store_true", help="Ignore the audio track of the media file.")
    source.add_argument("--audio-chunk-ms", type=int, default=500, help="Audio chunk size emitted by the source.")
    source.add_argument(
        "--no-realtime",
        action="store_true",
        help="Process as fast as possible instead of pacing playback to wall time.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--quiet-log", action="store_true", help="Disable timestamped runtime logs.")
    output.add_argument(
        "--live-prompts",
        action="store_true",
        help="Read extra prompts from stdin while the stream plays and re-plan the funnel.",
    )
    output.add_argument("--speak", action="store_true", help="Verbalize responses with text-to-speech.")
    output.add_argument("--speak-play", action="store_true", help="Also play the generated speech locally.")
    output.add_argument("--tts-model", default="tts-1", help="Text-to-speech model.")
    output.add_argument("--tts-voice", default="alloy", help="Text-to-speech voice.")
    return parser


def options_from_args(args: argparse.Namespace) -> MonitorOptions:
    """Translate parsed flags into the same options object the web UI builds."""
    return MonitorOptions(
        use_planner=not args.no_planner,
        planner_model=args.planner_model,
        verifier_model=args.verifier_model,
        initial_frame_context=not args.no_initial_frame_context,
        model_responder=args.model_responder,
        vision_sample_seconds=args.vision_sample_seconds,
        verifier_sample_interval_ms=_resolve_ms(
            args.verifier_sample_interval_ms, args.verifier_sample_interval_seconds
        ),
        verifier_concurrent_requests=args.verifier_concurrent_requests,
        verifier_min_request_interval_ms=_resolve_ms(
            args.verifier_min_request_interval_ms, args.verifier_min_request_interval_seconds
        ),
        verifier_queued_frame_dedupe_window_ms=_resolve_ms(
            args.verifier_queued_frame_dedupe_window_ms, args.verifier_queued_frame_dedupe_seconds
        ),
        include_audio=not args.no_audio,
        audio_chunk_ms=args.audio_chunk_ms,
        realtime=not args.no_realtime,
    )


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(
        run(
            args.prompt,
            args.media,
            options=options_from_args(args),
            log=not args.quiet_log,
            live_prompts=args.live_prompts,
            speak=args.speak or args.speak_play,
            speak_play=args.speak_play,
            tts_model=args.tts_model,
            tts_voice=args.tts_voice,
        )
    )


if __name__ == "__main__":
    main()
