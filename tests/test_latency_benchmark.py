from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meerkat.benchmarks.latency import (
    BenchmarkCase,
    build_command,
    evaluate_result,
    load_cases,
    parse_user_alerts,
    summarize,
)


def test_load_cases_resolves_relative_media_paths(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "clip",
                        "media": "clips/clip.mp4",
                        "prompt": "let me know when the hands separate",
                        "event_time_seconds": 3.7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(cases_path)

    assert cases == [
        BenchmarkCase(
            case_id="clip",
            media=str(tmp_path / "clips/clip.mp4"),
            prompt="let me know when the hands separate",
            event_time_seconds=3.7,
            options={},
        )
    ]


def test_load_cases_accepts_audio_only_media(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "conversation",
                        "media": "clips/conversation.mp3",
                        "prompt": "let me know who brought the blue umbrella",
                        "event_time_seconds": 14.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(cases_path)

    assert cases == [
        BenchmarkCase(
            case_id="conversation",
            media=str(tmp_path / "clips/conversation.mp3"),
            prompt="let me know who brought the blue umbrella",
            event_time_seconds=14.0,
            options={},
        )
    ]


def test_parse_user_alerts_handles_bold_ansi_and_stream_time() -> None:
    alerts = parse_user_alerts(
        "   5.37: \033[1mUSER RESPONSE trigger=hands_separating: Your hands have separated.\033[0m stream=3.73s\n"
    )

    assert len(alerts) == 1
    assert alerts[0].wall_seconds == 5.37
    assert alerts[0].stream_seconds == 3.73
    assert alerts[0].trigger == "hands_separating"
    assert alerts[0].text == "Your hands have separated."


def test_parse_user_alerts_handles_missing_stream_time() -> None:
    alerts = parse_user_alerts("   5.37: USER RESPONSE trigger=audio_answer: Sophia brought it.\n")

    assert len(alerts) == 1
    assert alerts[0].wall_seconds == 5.37
    assert alerts[0].stream_seconds is None
    assert alerts[0].trigger == "audio_answer"
    assert alerts[0].text == "Sophia brought it."


def test_evaluate_result_passes_when_alert_fires_after_event() -> None:
    case = BenchmarkCase("clip", "clip.mp4", "prompt", 3.7)

    result = evaluate_result(
        case,
        ["cmd"],
        0,
        "   5.37: USER RESPONSE trigger=gate: Alert. stream=3.73s\n",
        "",
    )

    assert result.status == "pass"
    assert result.latency_seconds == 1.67
    assert result.passed


def test_evaluate_result_marks_early_alert_wrong() -> None:
    case = BenchmarkCase("clip", "clip.mp4", "prompt", 3.7)

    result = evaluate_result(
        case,
        ["cmd"],
        0,
        "   2.20: USER RESPONSE trigger=gate: Alert. stream=2.00s\n",
        "",
    )

    assert result.status == "early"
    assert result.latency_seconds == -1.5
    assert not result.passed


def test_evaluate_result_marks_pre_event_video_frame_wrong() -> None:
    case = BenchmarkCase("clip", "clip.mp4", "prompt", 3.7)

    result = evaluate_result(
        case,
        ["cmd"],
        0,
        "   5.20: USER RESPONSE trigger=gate: Alert. stream=2.26s\n",
        "",
    )

    assert result.status == "stream_early"
    assert result.latency_seconds == 1.5
    assert not result.passed


def test_evaluate_result_marks_missing_alert_as_miss() -> None:
    case = BenchmarkCase("clip", "clip.mp4", "prompt", 3.7)

    result = evaluate_result(case, ["cmd"], 0, "no alert here", "")

    assert result.status == "miss"
    assert result.latency_seconds is None


def test_evaluate_result_marks_late_alert_as_slow() -> None:
    case = BenchmarkCase("clip", "clip.mp4", "prompt", 3.7)

    result = evaluate_result(
        case,
        ["cmd"],
        0,
        "   7.20: USER RESPONSE trigger=gate: Alert. stream=5.50s\n",
        "",
        max_latency_seconds=3.0,
    )

    assert result.status == "slow"
    assert result.latency_seconds == 3.5
    assert not result.passed


def test_evaluate_result_marks_nonzero_return_as_error() -> None:
    case = BenchmarkCase("clip", "clip.mp4", "prompt", 3.7)

    result = evaluate_result(
        case,
        ["cmd"],
        1,
        "   5.37: USER RESPONSE trigger=gate: Alert. stream=3.73s\n",
        "boom",
    )

    assert result.status == "error"
    assert result.latency_seconds is None


def test_build_command_applies_defaults_and_case_options() -> None:
    defaults = argparse.Namespace(
        planner_model="gpt-5.6-luna",
        verifier_model=None,
        vision_sample_seconds=None,
        verifier_sample_interval_seconds=0.3,
        verifier_concurrent_requests=4,
        verifier_min_request_interval_seconds=0.2,
        verifier_queued_frame_dedupe_seconds=0.3,
        no_planner=False,
        no_realtime=False,
    )
    case = BenchmarkCase(
        "clip",
        "clip.mp4",
        "let me know when a person appears",
        3.7,
        options={"verifier_model": "gpt-5.4-mini"},
    )

    command = build_command(case, defaults)

    assert command[:4] == [sys.executable, "-m", "meerkat.cli", "let me know when a person appears"]
    assert command[command.index("--media") + 1] == "clip.mp4"
    assert command[command.index("--verifier-model") + 1] == "gpt-5.4-mini"
    assert command[command.index("--verifier-sample-interval-seconds") + 1] == "0.3"
    assert command[command.index("--verifier-concurrent-requests") + 1] == "4"


def test_build_command_carries_audio_only_media() -> None:
    defaults = argparse.Namespace(
        planner_model="gpt-5.6-luna",
        verifier_model=None,
        vision_sample_seconds=None,
        verifier_sample_interval_seconds=None,
        verifier_concurrent_requests=None,
        verifier_min_request_interval_seconds=None,
        verifier_queued_frame_dedupe_seconds=None,
        no_planner=False,
        no_realtime=False,
    )
    case = BenchmarkCase("conversation", "conversation.mp3", "prompt", 14.0)

    command = build_command(case, defaults)

    assert command[:3] == [sys.executable, "-m", "meerkat.cli"]
    assert command[command.index("--media") + 1] == "conversation.mp3"


def test_summarize_includes_only_pass_latencies() -> None:
    passing = evaluate_result(
        BenchmarkCase("pass", "a.mp4", "prompt", 1.0),
        ["cmd"],
        0,
        "   2.00: USER RESPONSE trigger=gate: Alert. stream=1.50s\n",
        "",
    )
    early = evaluate_result(
        BenchmarkCase("early", "b.mp4", "prompt", 3.0),
        ["cmd"],
        0,
        "   4.00: USER RESPONSE trigger=gate: Alert. stream=1.50s\n",
        "",
    )

    summary = summarize([passing, early])

    assert summary["counts"] == {"pass": 1, "stream_early": 1}
    assert summary["mean_latency_seconds"] == 1.0


def test_benchmark_children_silence_the_duplicate_class_warning() -> None:
    """OpenCV and PyAV each bundle FFmpeg; the macOS ObjC runtime warns about it.

    The runtime reads this switch at process start, so it can only be set for a
    child -- which is exactly what the benchmark spawns.
    """
    from meerkat.benchmarks.latency import _child_env

    env = _child_env()

    assert env["OBJC_DEBUG_DUPLICATE_CLASSES"] == "NO"
    assert "PATH" in env, "the child must inherit the rest of the environment"
