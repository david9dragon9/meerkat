from __future__ import annotations

from typing import Optional

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.benchmarks.metrics import Metrics
from meerkat.funnel.planner import RuleBasedFunnelPlanner
from meerkat.funnel.runtime import FunnelRuntime
from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec, StateUpdateSpec
from meerkat.gates.base import Gate
from meerkat.gates.object_detection import ObjectLabelGate
from meerkat.gates.temporal import TemporalCountGate
from meerkat.ingest.sources import SyntheticStreamSource
from meerkat.models.responders import LocalResponder, _notification_from_goal
from meerkat.runtime.event_bus import EventBus
from meerkat.runtime.runner import StreamRunner


class _TerminalSyntheticGate(Gate):
    def __init__(self, gate_id: str) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.terminal_response = True

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason="terminal test",
            evidence={"stream_time_ms": event.stream_time_ms},
        )


class _ModelConfirmedSyntheticGate(Gate):
    def __init__(self, gate_id: str) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.terminal_response = True

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason="gpt confirmed test",
            evidence={"stream_time_ms": event.stream_time_ms, "model_confirmed": True},
        )


class _ObservationSyntheticGate(Gate):
    def __init__(self, gate_id: str) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.terminal_response = True

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason="observation test",
            evidence={
                "stream_time_ms": event.stream_time_ms,
                "observation": f"frame {event.sequence_id}: dog visible",
            },
        )


class _StateEchoGate(Gate):
    def __init__(self, gate_id: str) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.terminal_response = True

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason="state echo",
            evidence={
                "stream_time_ms": event.stream_time_ms,
                "state": event.payload.get("state", {}),
            },
        )


class _RetimestampGate(Gate):
    def __init__(self, gate_id: str) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason="retimestamped",
            evidence={"stream_time_ms": event.stream_time_ms + 1000},
        )


class _FingerCountGate(Gate):
    def __init__(self, gate_id: str) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.terminal_response = True

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        count_by_sequence = {0: 2, 1: 2, 2: 4, 3: 4, 4: 0}
        count = count_by_sequence.get(event.sequence_id, 0)
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason=f"{count} fingers",
            evidence={"finger_count": count},
        )


async def test_dog_prompt_triggers_response() -> None:
    spec = RuleBasedFunnelPlanner().plan("let me know when the dog shows up")
    source = SyntheticStreamSource(
        fps=10,
        duration_seconds=1,
        object_schedule={2: ["dog"], 3: ["dog"]},
        realtime=False,
    )
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].text == "The dog showed up"
    assert responses[0].trigger_gate_id == "object_temporal_confirm"


async def test_runtime_skips_consecutive_duplicate_response_states(monkeypatch) -> None:
    import meerkat.funnel.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "build_gate", lambda spec: _FingerCountGate(spec.id))
    spec = FunnelSpec(
        goal="let me know how many fingers are in frame",
        gates=[GateSpec(id="finger_count", type="synthetic")],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=0,
            on_match_text="There are {evidence.finger_count} fingers in frame.",
        ),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=5, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert [response.text for response in responses] == [
        "There are 2 fingers in frame.",
        "There are 4 fingers in frame.",
        "There are 0 fingers in frame.",
    ]


async def test_runtime_publishes_gate_fire_at_evidence_stream_time(monkeypatch) -> None:
    import meerkat.funnel.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "build_gate", lambda _spec: _RetimestampGate("retimestamp"))
    bus = EventBus()
    runtime = FunnelRuntime(
        spec=FunnelSpec(
            goal="retimestamp",
            gates=[GateSpec(id="retimestamp", type="synthetic", params={})],
            response=ResponseSpec(on_match_text="ok"),
        ),
        bus=bus,
        responder=LocalResponder(),
        metrics=Metrics(),
    )
    queue = bus.subscribe([EventType.GATE_FIRE])
    handle = runtime.start()
    try:
        await bus.publish(
            StreamEvent(
                type=EventType.VIDEO_FRAME,
                source_id="test",
                stream_time_ms=500,
                sequence_id=1,
                payload={},
            )
        )
        event = await queue.get()
    finally:
        await handle.stop()

    assert event.stream_time_ms == 1500


async def test_transcript_keyword_prompt_triggers_response() -> None:
    spec = RuleBasedFunnelPlanner().plan('tell me when someone says "urgent"')
    source = SyntheticStreamSource(
        fps=5,
        duration_seconds=1,
        transcript_schedule={1: "this is urgent"},
        realtime=False,
    )
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].trigger_gate_id == "transcript_keyword"


async def test_local_responder_formats_evidence_placeholders() -> None:
    spec = FunnelSpec(
        goal="answer visible value",
        gates=[],
        response=ResponseSpec(on_match_text="The runner's bib number is {evidence.text}."),
    )
    fire = GateFire(
        gate_id="extract_number",
        confidence=1.0,
        reason="extracted",
        evidence={"text": "48"},
    )

    response = await LocalResponder().respond(spec, fire)

    assert response.text == "The runner's bib number is 48."


async def test_non_matching_stream_stays_quiet() -> None:
    spec = RuleBasedFunnelPlanner().plan("let me know when the cat shows up")
    source = SyntheticStreamSource(
        fps=5,
        duration_seconds=1,
        object_schedule={1: ["dog"]},
        realtime=False,
    )
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert responses == []


async def test_terminal_gate_can_trigger_before_last_gate(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "terminal_test":
            return _TerminalSyntheticGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="alert when terminal gate fires",
        gates=[
            GateSpec(id="early_terminal", type="terminal_test"),
            GateSpec(id="later_non_terminal", type="object_label", params={"classes": ["never"]}),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="brief", on_match_text="Terminal fired."),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=1, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].trigger_gate_id == "early_terminal"


async def test_terminal_gate_with_downstream_confirmation_does_not_bypass_leaf(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "terminal_test":
            return _TerminalSyntheticGate(spec.id)
        if spec.type == "temporal_count":
            return TemporalCountGate(
                gate_id=spec.id,
                upstream_gate_id=str(spec.params["upstream_gate_id"]),
                required_count=int(spec.params["required_count"]),
                window_ms=int(spec.params["window_ms"]),
            )
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="alert after confirmation",
        gates=[
            GateSpec(id="early_terminal", type="terminal_test"),
            GateSpec(
                id="confirmed_leaf",
                type="temporal_count",
                params={"upstream_gate_id": "early_terminal", "required_count": 1, "window_ms": 1000},
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="brief", on_match_text="Confirmed."),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=1, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].trigger_gate_id == "confirmed_leaf"


async def test_temporal_count_gate_accepts_multiple_candidate_gates() -> None:
    gate = TemporalCountGate(
        gate_id="candidate_count",
        upstream_gate_ids=["motion", "ocr"],
        required_count=2,
        window_ms=1000,
    )
    first = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"gate_fire": GateFire("motion", 0.5, "motion", {"frame": "frame-a"})},
    )
    second = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1400,
        sequence_id=2,
        payload={"gate_fire": GateFire("ocr", 0.8, "ocr", {"text": "2 outs", "frame": "frame-b"})},
    )

    assert await gate.process(first) is None
    fire = await gate.process(second)

    assert fire is not None
    assert fire.gate_id == "candidate_count"
    assert fire.evidence["upstream_gate_ids"] == ["motion", "ocr"]
    assert fire.evidence["frame"] == "frame-b"


async def test_planned_runtime_requires_model_confirmed_evidence(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "terminal_test":
            return _TerminalSyntheticGate(spec.id)
        if spec.type == "model_vision_query":
            return _ModelConfirmedSyntheticGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    cheap_spec = FunnelSpec(
        goal="alert only with model confirmation",
        gates=[
            GateSpec(id="cheap_terminal", type="terminal_test"),
            GateSpec(id="model_gate", type="model_vision_query", params={"query": "confirm"}),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="brief", on_match_text="Confirmed."),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=1, realtime=False)
    runner = StreamRunner(source=source, spec=cheap_spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].trigger_gate_id == "model_gate"


async def test_response_can_increment_and_read_runtime_state(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "terminal_test":
            return _TerminalSyntheticGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="increment a count every time a dog shows up and say the count",
        gates=[GateSpec(id="dog_seen", type="terminal_test")],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=0,
            style="brief",
            on_match_text="Dog count: {state.dog_count}",
            state_updates=[StateUpdateSpec(key="dog_count", operation="increment", value=1)],
        ),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=3, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert [response.text for response in responses] == ["Dog count: 1", "Dog count: 2", "Dog count: 3"]
    assert responses[-1].evidence["state"] == {"dog_count": 3}


async def test_runner_seeds_initial_state_for_gates(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "state_echo":
            return _StateEchoGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="read seeded state",
        gates=[GateSpec(id="echo", type="state_echo")],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=10,
            on_match_text="{state.initial_visual_context}",
        ),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=1, realtime=False)
    runner = StreamRunner(
        source=source,
        spec=spec,
        responder=LocalResponder(),
        initial_state={"initial_visual_context": "A runner wearing bib #70 is in frame."},
    )

    responses = await runner.run()

    assert responses[0].text == "A runner wearing bib #70 is in frame."


async def test_response_can_increment_unique_values_only_once(monkeypatch) -> None:
    class _FoodWordGate(Gate):
        def __init__(self, gate_id: str) -> None:
            super().__init__(gate_id, [EventType.VIDEO_FRAME])
            self.terminal_response = True

        async def process(self, event: StreamEvent) -> Optional[GateFire]:
            matches_by_sequence = {
                0: ["waffle", "pancake"],
                1: ["pancake"],
                2: ["apple"],
                3: ["apple"],
            }
            return GateFire(
                gate_id=self.gate_id,
                confidence=1.0,
                reason="food words",
                evidence={
                    "stream_time_ms": event.stream_time_ms,
                    "matches": matches_by_sequence.get(event.sequence_id, []),
                },
            )

    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "food_word_test":
            return _FoodWordGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="count unique food words",
        gates=[GateSpec(id="food_words", type="food_word_test")],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=0,
            style="brief",
            on_match_text="Food word count: {state.food_word_count}",
            state_updates=[
                StateUpdateSpec(
                    key="food_word_count",
                    operation="increment_unique",
                    value="{evidence.matches}",
                )
            ],
        ),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=4, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert [response.text for response in responses] == [
        "Food word count: 2",
        "Food word count: 3",
    ]
    assert responses[-1].evidence["state"]["food_word_count"] == 3


async def test_gate_can_write_natural_language_state_before_response(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "observation_test":
            return _ObservationSyntheticGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="remember what is happening and say the notes",
        gates=[
            GateSpec(
                id="scene_observer",
                type="observation_test",
                params={
                    "state_updates": [
                        {
                            "key": "scene_notes",
                            "operation": "append_text",
                            "value": "{evidence.observation}",
                        }
                    ]
                },
            )
        ],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=0,
            style="brief",
            on_match_text="Notes: {state.scene_notes}",
            max_responses=1,
        ),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=2, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].text == "Notes: frame 0: dog visible"
    assert responses[0].evidence["state"]["scene_notes"] == "frame 0: dog visible"


async def test_temporal_join_inputs_are_not_terminal_response_gates(monkeypatch) -> None:
    from meerkat.funnel.factory import build_gate as real_build_gate

    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.id == "model_intermediate":
            return _ModelConfirmedSyntheticGate(spec.id)
        if spec.id == "cheap_intermediate":
            return _TerminalSyntheticGate(spec.id)
        return real_build_gate(spec)

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    spec = FunnelSpec(
        goal="joined event",
        gates=[
            GateSpec(id="model_intermediate", type="synthetic_model"),
            GateSpec(id="cheap_intermediate", type="synthetic_cheap"),
            GateSpec(
                id="joined_terminal",
                type="temporal_join",
                params={"gate_ids": ["model_intermediate", "cheap_intermediate"], "join_window_ms": 1000},
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=0, on_match_text="Joined."),
    )
    source = SyntheticStreamSource(fps=1, duration_seconds=1, realtime=False)
    runner = StreamRunner(source=source, spec=spec, responder=LocalResponder())

    responses = await runner.run()

    assert len(responses) == 1
    assert responses[0].trigger_gate_id == "joined_terminal"
    assert responses[0].text == "Joined."


async def test_runtime_can_replace_active_funnel(monkeypatch) -> None:
    def fake_build_gate(spec: GateSpec) -> Gate:
        if spec.type == "terminal_test":
            return _TerminalSyntheticGate(spec.id)
        return ObjectLabelGate(spec.id, classes=["never"])

    monkeypatch.setattr("meerkat.funnel.runtime.build_gate", fake_build_gate)
    bus = EventBus()
    response_queue = bus.subscribe([EventType.MODEL_RESPONSE])
    runtime = FunnelRuntime(
        spec=FunnelSpec(
            goal="first",
            gates=[GateSpec(id="first_gate", type="terminal_test")],
            response=ResponseSpec(model="local", cooldown_seconds=0, on_match_text="First."),
        ),
        bus=bus,
        responder=LocalResponder(),
        metrics=Metrics(),
    )
    await runtime.warmup()
    handle = runtime.start()
    try:
        await bus.publish(StreamEvent(EventType.VIDEO_FRAME, "test", 0, 0, payload={"frame": None}))
        first = await response_queue.get()
        await runtime.replace_spec(
            FunnelSpec(
                goal="second",
                gates=[GateSpec(id="second_gate", type="terminal_test")],
                response=ResponseSpec(model="local", cooldown_seconds=0, on_match_text="Second."),
            )
        )
        assert bus.subscriber_counts()["video_frame"] == 1
        assert bus.subscriber_counts()["gate_fire"] == 1
        await bus.publish(StreamEvent(EventType.VIDEO_FRAME, "test", 1000, 1, payload={"frame": None}))
        second = await response_queue.get()
    finally:
        await handle.stop()

    assert first.payload["model_response"].text == "First."
    assert second.payload["model_response"].text == "Second."
    assert second.payload["model_response"].trigger_gate_id == "second_gate"


def test_notification_from_goal_is_short_and_user_facing() -> None:
    assert _notification_from_goal("let me know when the dog picks up the beachball") == "The dog picked up the beachball"
    assert _notification_from_goal("tell me when someone says urgent") == "Someone said urgent"
