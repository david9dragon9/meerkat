"""The CLI, the web UI, and the benchmark must be three doors into one pipeline.

These tests fail if a front end grows its own copy of the flow, or if a flag
name drifts out of sync between them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from meerkat.benchmarks.latency import BenchmarkCase, build_command
from meerkat.cli import _build_parser, options_from_args
from meerkat.runtime.session import FileTarget, MonitorOptions, MonitorSession
from meerkat.web import MediaStore, StartRequest, WebSession

from conftest import SyntheticTarget


def _cli_options(argv: list[str]) -> MonitorOptions:
    return options_from_args(_build_parser().parse_args(argv))


def test_web_session_is_a_monitor_session() -> None:
    """The UI must inherit the shared flow rather than reimplement it."""
    assert issubclass(WebSession, MonitorSession)


def test_start_request_maps_onto_the_same_options_the_cli_builds() -> None:
    cli = _cli_options(
        [
            "let me know when a person appears",
            "--media",
            "clip.mp4",
            "--verifier-concurrent-requests",
            "4",
            "--verifier-min-request-interval-ms",
            "200",
            "--verifier-queued-frame-dedupe-window-ms",
            "300",
        ]
    )
    web = StartRequest(media="clip.mp4", prompt="let me know when a person appears").to_options()

    assert web == cli


# Options the web request names differently, mirroring the CLI's flag names.
_WEB_ALIASES = {"use_planner": "no_planner"}

# Options fixed by the browser itself: a media element always plays at 1x with
# its own audio, so there is nothing for the UI to choose.
_BROWSER_FIXED = {"include_audio", "audio_chunk_ms", "realtime"}


def test_start_request_covers_every_shared_option() -> None:
    """A new option must be reachable from the UI, not just the CLI."""
    fields = set(StartRequest.model_fields)
    missing = {
        field
        for field in MonitorOptions.__dataclass_fields__
        if field not in _BROWSER_FIXED and _WEB_ALIASES.get(field, field) not in fields
    }

    assert not missing, f"MonitorOptions fields the web request cannot express: {sorted(missing)}"


@pytest.mark.parametrize(
    "argv",
    [
        ["watch for a person", "--media", "clip.mp4"],
        ["watch for a person", "--media", "clip.mp4", "--no-planner", "--model-responder"],
        ["watch for a person", "--media", "clip.mp4", "--verifier-sample-interval-seconds", "0.4"],
    ],
)
def test_cli_options_round_trip_through_a_session(argv: list[str]) -> None:
    """Whatever the CLI parses is exactly what the shared session receives."""
    options = _cli_options(argv)
    session = MonitorSession(target=FileTarget("clip.mp4"), prompt=argv[0], options=options)

    assert session.options is options
    assert session.prompt == argv[0]


async def test_cli_and_web_compile_the_same_funnel(tmp_path: Path) -> None:
    """The same prompt and media must produce an identical funnel either way."""
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"fake")
    prompt = "let me know when an orange is mentioned"

    cli_options = _cli_options([prompt, "--media", str(media), "--no-planner"])
    web_options = StartRequest(media=str(media), prompt=prompt, no_planner=True).to_options()

    cli_spec = await MonitorSession(FileTarget(str(media)), prompt, cli_options)._compile(prompt)
    web_spec = await MonitorSession(FileTarget(str(media)), prompt, web_options)._compile(prompt)

    assert cli_spec == web_spec


def test_benchmark_command_only_uses_flags_the_cli_accepts() -> None:
    """The benchmark shells out to the CLI, so its flags have to stay valid."""
    defaults = argparse.Namespace(
        planner_model="planner-model",
        verifier_model="verifier-model",
        vision_sample_seconds=1.0,
        verifier_sample_interval_seconds=0.3,
        verifier_concurrent_requests=4,
        verifier_min_request_interval_seconds=0.2,
        verifier_queued_frame_dedupe_seconds=0.3,
        no_planner=True,
        no_realtime=True,
    )
    command = build_command(BenchmarkCase("clip", "clip.mp4", "watch for a person", 1.0), defaults)

    assert command[:3] == [sys.executable, "-m", "meerkat.cli"]
    args = _build_parser().parse_args(command[3:])

    assert args.prompt == "watch for a person"
    assert args.media == "clip.mp4"
    assert options_from_args(args) == MonitorOptions(
        use_planner=False,
        planner_model="planner-model",
        verifier_model="verifier-model",
        vision_sample_seconds=1.0,
        verifier_sample_interval_ms=300,
        verifier_concurrent_requests=4,
        verifier_min_request_interval_ms=200,
        verifier_queued_frame_dedupe_window_ms=300,
        realtime=False,
    )


async def test_cli_and_web_produce_the_same_responses(fixed_funnel, tmp_path: Path) -> None:
    """The same funnel over the same stream must yield the same alerts either way."""
    prompt = "let me know when a person appears"

    cli_session = MonitorSession(SyntheticTarget(), prompt, MonitorOptions(realtime=False))
    cli_responses = [response.text for response in await cli_session.run()]

    web_session = WebSession(
        "parity",
        MediaStore(tmp_path),
        None,
        StartRequest(media="clip.mp4", prompt=prompt, speak=False),
    )
    web_session.target = SyntheticTarget()
    await web_session.start()
    await web_session.run_task
    await web_session.close()

    web_responses = []
    while not web_session.queue.empty():
        event = web_session.queue.get_nowait()
        if event["type"] == "user_message" and event["trigger_gate_id"] != "ack":
            web_responses.append(event["message"])

    assert cli_responses == ["A person appeared."]
    assert web_responses == cli_responses


async def test_both_front_ends_acknowledge_the_prompt(fixed_funnel, tmp_path: Path) -> None:
    """A prompt is acknowledged the same way whether it is typed or posted."""
    prompt = "let me know when a person appears"
    expected = "Okay, I will watch for: let me know when a person appears."

    logged: list[str] = []

    class _CapturingSession(MonitorSession):
        def on_assistant_message(self, text, stream_time_ms, trigger_gate_id, evidence) -> None:
            if trigger_gate_id == "ack":
                logged.append(text)

    await _CapturingSession(SyntheticTarget(), prompt, MonitorOptions(realtime=False)).plan()

    web_session = WebSession(
        "ack", MediaStore(tmp_path), None, StartRequest(media="clip.mp4", prompt=prompt, speak=False)
    )
    web_session.target = SyntheticTarget()
    await web_session.plan()
    web_acks = [
        event["message"]
        for event in _drain(web_session.queue)
        if event["type"] == "user_message" and event["trigger_gate_id"] == "ack"
    ]

    assert logged == [expected]
    assert web_acks == [expected]


def _drain(queue) -> list:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events
