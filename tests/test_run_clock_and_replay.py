"""The two clocks a run keeps, replaying a funnel, and reporting a changed value."""

from __future__ import annotations

import asyncio
import time

import pytest

from meerkat.runtime.logging import RuntimeLogger
from meerkat.runtime.session import MonitorOptions, MonitorSession
from meerkat.web import MediaStore, StartRequest, WebSession

from conftest import SyntheticTarget


# --------------------------------------------------------------------------
# wall clock
# --------------------------------------------------------------------------


def test_wall_clock_excludes_paused_time() -> None:
    """Wall time means time spent processing, so a pause must not inflate it."""
    logger = RuntimeLogger(enabled=False)
    logger.reset_wall_clock()
    time.sleep(0.05)

    running = logger.elapsed_ms()
    logger.pause()
    time.sleep(0.15)
    while_paused = logger.elapsed_ms()
    logger.resume()

    assert while_paused == pytest.approx(running, abs=15)
    assert logger.elapsed_ms() < 120, "the 150ms pause leaked into the wall clock"


def test_wall_clock_keeps_advancing_after_resume() -> None:
    logger = RuntimeLogger(enabled=False)
    logger.reset_wall_clock()
    logger.pause()
    time.sleep(0.05)
    logger.resume()
    before = logger.elapsed_ms()
    time.sleep(0.05)

    assert logger.elapsed_ms() > before


def test_pausing_twice_does_not_double_count() -> None:
    logger = RuntimeLogger(enabled=False)
    logger.reset_wall_clock()
    logger.pause()
    logger.pause()
    time.sleep(0.05)
    logger.resume()

    assert logger.elapsed_ms() < 40
    assert not logger.paused


def test_resetting_the_clock_forgets_earlier_pauses() -> None:
    logger = RuntimeLogger(enabled=False)
    logger.pause()
    time.sleep(0.05)
    logger.resume()
    logger.reset_wall_clock()

    assert logger.elapsed_ms() < 20


def test_session_pause_stops_the_wall_clock() -> None:
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    session.logger = RuntimeLogger(enabled=False)
    session.logger.reset_wall_clock()

    session.pause()
    assert session.logger.paused
    assert not session.playback_event.is_set()

    session.resume()
    assert not session.logger.paused
    assert session.playback_event.is_set()


# --------------------------------------------------------------------------
# stream clock
# --------------------------------------------------------------------------


async def test_runner_publishes_the_stream_position(fixed_funnel) -> None:
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    runner = await session.start()

    assert runner.stream_time_ms == 0
    await runner.run()
    await session.close()

    assert runner.stream_time_ms > 0


async def test_a_mid_stream_prompt_is_stamped_with_stream_time_not_wall_time(
    fixed_funnel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These are different clocks; a prompt belongs at the media position."""
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    runner = await session.start()
    runner.stream_time_ms = 4321
    # Make wall and stream disagree, so using the wrong one is visible.
    session.logger.reset_wall_clock()
    session.logger._start -= 30.0

    stamped: list[int] = []
    monkeypatch.setattr(
        type(session), "on_user_prompt", lambda self, prompt, ms: stamped.append(ms)
    )
    await session.add_prompt("also tell me when a dog appears")
    await session.close()

    assert stamped == [4321]


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


async def test_restart_reuses_the_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaying the same media must not pay for planning again."""
    compiles = []

    def compile_once(*args, **kwargs):
        compiles.append(kwargs.get("prompt") or args[0])
        from conftest import FIXED_SPEC

        return FIXED_SPEC

    monkeypatch.setattr("meerkat.runtime.session.build_media_spec", compile_once)

    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    await session.run()
    assert len(compiles) == 1

    await session.restart()
    runner = await session.start()
    await runner.run()
    await session.close()

    assert len(compiles) == 1, "restart re-planned instead of reusing the funnel"


async def test_restart_builds_a_fresh_runner_and_source(fixed_funnel) -> None:
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    first = await session.start()
    await first.run()

    await session.restart()
    second = await session.start()
    await second.run()
    await session.close()

    assert second is not first
    assert session.source is not None


async def test_restart_reports_the_same_alerts_again(fixed_funnel) -> None:
    """A replay is a fresh run: gate counts and cooldowns must not carry over."""
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    first = [response.text for response in await session.run()]

    await session.restart()
    second = [response.text for response in await session.run()]

    assert first == ["A person appeared."]
    assert second == first


async def test_restart_before_planning_does_nothing(fixed_funnel) -> None:
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))

    await session.restart()

    assert session.runner is None
    assert session.spec is None


async def test_restart_does_not_begin_ingesting(fixed_funnel) -> None:
    """The caller starts the replay, so a browser's playback stays in step."""
    session = MonitorSession(SyntheticTarget(), "watch", MonitorOptions(realtime=False))
    await session.run()

    await session.restart()

    assert session.runner is None, "restart started a run on its own"
    assert session.source is None
    assert session.spec is not None, "the plan must survive a restart"


async def test_web_restart_clears_the_duplicate_message_guard(fixed_funnel, tmp_path) -> None:
    """The UI suppresses repeated alerts; a replay must not be suppressed by the last run."""
    session = WebSession(
        "replay", MediaStore(tmp_path), None, StartRequest(media="clip.mp4", prompt="watch", speak=False)
    )
    session.target = SyntheticTarget()

    await session.start()
    await session.run_task
    seen_first = _user_messages(session)

    await session.restart()
    await session.start()
    await session.run_task
    seen_second = _user_messages(session)
    await session.close()

    assert seen_first == ["A person appeared."]
    assert seen_second == seen_first


def _user_messages(session: WebSession) -> list:
    found = []
    while not session.queue.empty():
        event = session.queue.get_nowait()
        if event["type"] == "user_message" and event["trigger_gate_id"] != "ack":
            found.append(event["message"])
    return found


# --------------------------------------------------------------------------
# reporting a value that changes
# --------------------------------------------------------------------------


async def _responses_for(values: list, template: str, cooldown: float) -> list:
    """Fire a terminal gate once per value and collect what the user is told."""
    from meerkat.benchmarks.metrics import Metrics
    from meerkat.events import EventType, GateFire, StreamEvent
    from meerkat.funnel.runtime import FunnelRuntime
    from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec
    from meerkat.models.responders import LocalResponder
    from meerkat.runtime.event_bus import EventBus

    spec = FunnelSpec(
        goal="report the value",
        gates=[GateSpec(id="v", type="object_label")],
        response=ResponseSpec(on_match_text=template, cooldown_seconds=cooldown),
    )
    bus = EventBus()
    runtime = FunnelRuntime(spec=spec, bus=bus, responder=LocalResponder(), metrics=Metrics())
    said: list = []
    queue = bus.subscribe([EventType.MODEL_RESPONSE])

    async def collect() -> None:
        while True:
            said.append((await queue.get()).payload["model_response"].text)

    task = asyncio.create_task(collect())
    handle = runtime.start()
    for index, value in enumerate(values):
        await bus.publish(
            StreamEvent(
                type=EventType.GATE_FIRE,
                source_id="t",
                stream_time_ms=index * 100,
                sequence_id=index,
                payload={"gate_fire": GateFire("v", 1.0, "matched", {"text": value})},
            )
        )
        await asyncio.sleep(0.25)
    await handle.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return said


async def test_a_changed_value_is_reported_even_inside_the_cooldown() -> None:
    """The cooldown throttles repetition; it must never withhold new information."""
    said = await _responses_for(["2", "2", "4", "4", "1"], "{evidence.text} fingers", cooldown=30.0)

    assert said == ["2 fingers", "4 fingers", "1 fingers"]


async def test_an_unchanged_value_is_reported_only_once() -> None:
    said = await _responses_for(["2", "2", "2"], "{evidence.text} fingers", cooldown=0.0)

    assert said == ["2 fingers"]


async def test_a_fixed_message_is_still_rate_limited() -> None:
    said = await _responses_for(["a", "b", "c"], "Event happened.", cooldown=30.0)

    assert said == ["Event happened."]


async def test_the_cooldown_still_throttles_a_flapping_value() -> None:
    """Alternating between two values is repetition, not news."""
    said = await _responses_for(["2", "4", "2"], "{evidence.text} fingers", cooldown=30.0)

    assert said == ["2 fingers", "4 fingers"]


async def test_no_cooldown_reports_every_change() -> None:
    said = await _responses_for(["2", "4", "2"], "{evidence.text} fingers", cooldown=0.0)

    assert said == ["2 fingers", "4 fingers", "2 fingers"]


def test_an_extracting_funnel_puts_the_value_in_the_message() -> None:
    from meerkat.funnel.compiler import _ensure_extracted_value_reaches_the_user
    from meerkat.funnel.spec import GateSpec, ResponseSpec

    gates = [GateSpec(id="v", type="model_vision_query", params={"verification_mode": "extract"})]

    filled = _ensure_extracted_value_reaches_the_user(gates, ResponseSpec(on_match_text="The count is."))
    assert filled.on_match_text == "The count is: {evidence.text}"

    empty = _ensure_extracted_value_reaches_the_user(gates, ResponseSpec(on_match_text=None))
    assert empty.on_match_text == "{evidence.text}"


def test_a_message_that_already_carries_the_value_is_left_alone() -> None:
    from meerkat.funnel.compiler import _ensure_extracted_value_reaches_the_user
    from meerkat.funnel.spec import GateSpec, ResponseSpec

    gates = [GateSpec(id="v", type="model_vision_query", params={"verification_mode": "extract"})]
    response = ResponseSpec(on_match_text="I count {evidence.text}")

    assert _ensure_extracted_value_reaches_the_user(gates, response) is response


def test_a_yes_no_funnel_keeps_its_plain_message() -> None:
    from meerkat.funnel.compiler import _ensure_extracted_value_reaches_the_user
    from meerkat.funnel.spec import GateSpec, ResponseSpec

    gates = [GateSpec(id="v", type="model_vision_query", params={"verification_mode": "binary"})]
    response = ResponseSpec(on_match_text="A person appeared.")

    assert _ensure_extracted_value_reaches_the_user(gates, response) is response


def test_a_request_for_repeated_updates_is_never_capped_at_one() -> None:
    from meerkat.funnel.compiler import _is_one_shot_notification_prompt

    for prompt in (
        "let me know when the number of fingers changes",
        "tell me when the score changes, keep updating me",
        "let me know every time the count changes",
        "tell me when the value updates each time",
    ):
        assert not _is_one_shot_notification_prompt(prompt), prompt

    for prompt in (
        "let me know when a person appears",
        "what number is on the runner's bib? let me know as soon as you find out",
    ):
        assert _is_one_shot_notification_prompt(prompt), prompt
