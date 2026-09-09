from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

from meerkat.env import load_dotenv


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
USER_RESPONSE_RE = re.compile(
    r"^\s*(?P<wall>\d+(?:\.\d+)?):\s+USER RESPONSE trigger=(?P<trigger>[^:]+):"
    r"\s*(?P<text>.*?)(?:\s+stream=(?P<stream>\d+(?:\.\d+)?)s)?\s*$"
)



@dataclass(frozen=True)
class BenchmarkCase:
    """One annotated clip: what to watch for, and when the event actually happens."""

    case_id: str
    media: str
    prompt: str
    event_time_seconds: float
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    wall_seconds: float
    trigger: str
    text: str
    stream_seconds: Optional[float]


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    status: str
    latency_seconds: Optional[float]
    alert: Optional[Alert]
    returncode: int
    command: List[str]
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def load_cases(path: Path) -> List[BenchmarkCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["cases"] if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Benchmark file must be a list or an object with a 'cases' list.")
    base_dir = path.parent
    return [_parse_case(item, base_dir) for item in items]


def _parse_case(item: Dict[str, Any], base_dir: Path) -> BenchmarkCase:
    if not isinstance(item, dict):
        raise ValueError("Each benchmark case must be an object.")
    case_id = str(item.get("id") or "").strip()
    media = str(item.get("media") or "").strip()
    prompt = str(item.get("prompt") or "").strip()
    event_time = item.get("event_time_seconds")
    if not case_id:
        raise ValueError("Each benchmark case needs an 'id'.")
    if not media:
        raise ValueError(f"Benchmark case {case_id!r} needs a 'media' path.")
    if not prompt:
        raise ValueError(f"Benchmark case {case_id!r} needs a 'prompt'.")
    if event_time is None:
        raise ValueError(f"Benchmark case {case_id!r} needs 'event_time_seconds'.")
    media_path = Path(media)
    if not media_path.is_absolute():
        media_path = base_dir / media_path
    options = item.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"Benchmark case {case_id!r} options must be an object.")
    return BenchmarkCase(
        case_id=case_id,
        media=str(media_path),
        prompt=prompt,
        event_time_seconds=float(event_time),
        options=options,
    )


def parse_user_alerts(output: str) -> List[Alert]:
    alerts = []
    for raw_line in output.splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = USER_RESPONSE_RE.match(line)
        if not match:
            continue
        stream = match.group("stream")
        alerts.append(
            Alert(
                wall_seconds=float(match.group("wall")),
                trigger=match.group("trigger"),
                text=match.group("text").strip(),
                stream_seconds=float(stream) if stream is not None else None,
            )
        )
    return alerts


def evaluate_result(
    case: BenchmarkCase,
    command: List[str],
    returncode: int,
    stdout: str,
    stderr: str,
    max_latency_seconds: Optional[float] = None,
) -> BenchmarkResult:
    alerts = parse_user_alerts(stdout)
    first_alert = alerts[0] if alerts else None
    if returncode != 0:
        status = "error"
        latency = None
    elif first_alert is None:
        status = "miss"
        latency = None
    else:
        latency = first_alert.wall_seconds - case.event_time_seconds
        if latency < 0:
            status = "early"
        elif first_alert.stream_seconds is not None and first_alert.stream_seconds < case.event_time_seconds:
            status = "stream_early"
        elif max_latency_seconds is not None and latency > max_latency_seconds:
            status = "slow"
        else:
            status = "pass"
    return BenchmarkResult(
        case=case,
        status=status,
        latency_seconds=latency,
        alert=first_alert,
        returncode=returncode,
        command=command,
        stdout=stdout,
        stderr=stderr,
    )


def _child_env() -> Dict[str, str]:
    """Environment for a benchmark subprocess.

    OpenCV and PyAV each bundle their own FFmpeg, so importing both makes the
    macOS Objective-C runtime warn about duplicate `AVFFrameReceiver` /
    `AVFAudioReceiver` classes. Neither library uses FFmpeg's avfoundation
    capture device here, so the warning is noise; silence it so a long
    benchmark's stderr stays readable. The runtime reads this at process start,
    which is why it can only be set for a child.
    """
    return {**os.environ, "OBJC_DEBUG_DUPLICATE_CLASSES": "NO"}


def run_case(case: BenchmarkCase, defaults: argparse.Namespace) -> BenchmarkResult:
    command = build_command(case, defaults)
    completed = subprocess.run(
        command,
        cwd=defaults.cwd,
        env=_child_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=defaults.timeout_seconds,
        check=False,
    )
    return evaluate_result(
        case,
        command,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        max_latency_seconds=defaults.max_latency_seconds,
    )


def build_command(case: BenchmarkCase, defaults: argparse.Namespace) -> List[str]:
    options = {**_default_case_options(defaults), **case.options}
    command = [
        sys.executable,
        "-m",
        "meerkat.cli",
        case.prompt,
        "--media",
        case.media,
    ]
    _append_optional(command, "--planner-model", options.get("planner_model"))
    _append_optional(command, "--verifier-model", options.get("verifier_model"))
    _append_optional(command, "--vision-sample-seconds", options.get("vision_sample_seconds"))
    _append_optional(command, "--verifier-sample-interval-seconds", options.get("verifier_sample_interval_seconds"))
    _append_optional(command, "--verifier-concurrent-requests", options.get("verifier_concurrent_requests"))
    _append_optional(
        command,
        "--verifier-min-request-interval-seconds",
        options.get("verifier_min_request_interval_seconds"),
    )
    _append_optional(
        command,
        "--verifier-queued-frame-dedupe-seconds",
        options.get("verifier_queued_frame_dedupe_seconds"),
    )
    if options.get("no_planner"):
        command.append("--no-planner")
    if options.get("no_realtime"):
        command.append("--no-realtime")
    return command


def _default_case_options(defaults: argparse.Namespace) -> Dict[str, Any]:
    return {
        "planner_model": defaults.planner_model,
        "verifier_model": defaults.verifier_model,
        "vision_sample_seconds": defaults.vision_sample_seconds,
        "verifier_sample_interval_seconds": defaults.verifier_sample_interval_seconds,
        "verifier_concurrent_requests": defaults.verifier_concurrent_requests,
        "verifier_min_request_interval_seconds": defaults.verifier_min_request_interval_seconds,
        "verifier_queued_frame_dedupe_seconds": defaults.verifier_queued_frame_dedupe_seconds,
        "no_planner": defaults.no_planner,
        "no_realtime": defaults.no_realtime,
    }


def _append_optional(command: List[str], flag: str, value: object) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def summarize(results: Sequence[BenchmarkResult]) -> Dict[str, Any]:
    latencies = [result.latency_seconds for result in results if result.status == "pass" and result.latency_seconds is not None]
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary: Dict[str, Any] = {
        "total": len(results),
        "counts": counts,
    }
    if latencies:
        summary.update(
            {
                "mean_latency_seconds": mean(latencies),
                "median_latency_seconds": median(latencies),
                "max_latency_seconds": max(latencies),
                "min_latency_seconds": min(latencies),
            }
        )
    return summary


def write_json_report(path: Path, results: Sequence[BenchmarkResult]) -> None:
    payload = {
        "summary": summarize(results),
        "results": [_result_to_dict(result, include_logs=True) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_report(path: Path, results: Sequence[BenchmarkResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "status",
                "event_time_seconds",
                "alert_wall_seconds",
                "latency_seconds",
                "alert_stream_seconds",
                "trigger",
                "text",
                "media",
                "prompt",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(_result_row(result))


def _result_to_dict(result: BenchmarkResult, include_logs: bool = False) -> Dict[str, Any]:
    row = _result_row(result)
    row["returncode"] = result.returncode
    row["command"] = result.command
    if include_logs:
        row["stdout"] = result.stdout
        row["stderr"] = result.stderr
    return row


def _result_row(result: BenchmarkResult) -> Dict[str, Any]:
    alert = result.alert
    return {
        "id": result.case.case_id,
        "status": result.status,
        "event_time_seconds": result.case.event_time_seconds,
        "alert_wall_seconds": alert.wall_seconds if alert else None,
        "latency_seconds": result.latency_seconds,
        "alert_stream_seconds": alert.stream_seconds if alert else None,
        "trigger": alert.trigger if alert else None,
        "text": alert.text if alert else None,
        "media": result.case.media,
        "prompt": result.case.prompt,
    }


def print_result(result: BenchmarkResult) -> None:
    row = _result_row(result)
    latency = row["latency_seconds"]
    latency_text = "n/a" if latency is None else f"{latency:.2f}s"
    alert_wall = row["alert_wall_seconds"]
    alert_text = "none" if alert_wall is None else f"{alert_wall:.2f}s"
    print(
        f"{row['id']}: {row['status']} latency={latency_text} "
        f"event={row['event_time_seconds']:.2f}s alert_wall={alert_text}"
    )
    if result.status == "early":
        print("  WRONG: alert fired before the annotated event time.")
    if result.status == "stream_early":
        print("  WRONG: the alert was triggered by a frame from before the annotated event time.")
    if result.status == "slow":
        print("  SLOW: alert fired after the latency target.")
    if result.status == "error":
        stderr_tail = "\n".join(result.stderr.splitlines()[-5:])
        if stderr_tail:
            print(stderr_tail)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure alert latency against annotated media/prompt pairs.")
    parser.add_argument("cases", help="JSON file containing benchmark cases.")
    parser.add_argument("--planner-model", help="Planner model override; defaults to MEERKAT_PLANNER_MODEL.")
    parser.add_argument("--verifier-model")
    parser.add_argument("--vision-sample-seconds", type=float)
    parser.add_argument("--verifier-sample-interval-seconds", type=float)
    parser.add_argument("--verifier-concurrent-requests", type=int)
    parser.add_argument("--verifier-min-request-interval-seconds", type=float)
    parser.add_argument("--verifier-queued-frame-dedupe-seconds", type=float)
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--max-latency-seconds",
        type=float,
        default=3.0,
        help="Mark an otherwise-correct alert as slow if it fires after this latency budget.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--cwd", default=str(Path.cwd()))
    args = parser.parse_args(argv)

    load_dotenv()
    cases = load_cases(Path(args.cases))
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}: running {Path(case.media).name}")
        result = run_case(case, args)
        results.append(result)
        print_result(result)

    summary = summarize(results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        write_json_report(args.json_out, results)
        print(f"Wrote JSON report: {args.json_out}")
    if args.csv_out:
        write_csv_report(args.csv_out, results)
        print(f"Wrote CSV report: {args.csv_out}")
    return 1 if any(result.status in {"early", "stream_early", "miss", "error", "slow"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
