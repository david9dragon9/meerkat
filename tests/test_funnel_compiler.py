from __future__ import annotations

from meerkat.cli import _resolve_ms
from meerkat.funnel.compiler import (
    _audio_runtime_spec_from_planned_spec,
    _video_runtime_spec_from_planned_spec,
)
from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec, StateUpdateSpec


def test_video_runtime_spec_uses_planned_query_gate() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[
            GateSpec(
                id="cheap_candidates",
                type="local_yolo_object",
                params={"classes": ["dog", "sports ball"], "sample_interval_ms": 100},
            ),
            GateSpec(
                id="check_action",
                type="model_vision_query",
                params={
                    "query": "Is the dog lifting the beach ball with its mouth?",
                    "min_confidence": 0.82,
                    "sample_interval_ms": 500,
                    "concurrent_requests": 4,
                    "min_request_interval_ms": 250,
                    "queued_frame_dedupe_window_ms": 600,
                    "model": "gpt-5.6-luna",
                    "upstream_gate_id": "cheap_candidates",
                    "verification_mode": "binary",
                },
            )
        ],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=10,
            style="Brief alert",
            on_match_text="The dog picked up the beach ball.",
            on_no_match="ignore",
        ),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the beachball",
        planned,
        sample_interval_ms=1000,
    )

    gate = runtime.gates[0]
    expensive_gate = runtime.gates[1]
    assert len(runtime.gates) == 2
    assert gate.type == "local_yolo_object"
    assert gate.params["classes"] == ["dog", "sports ball"]
    assert gate.id == "cheap_candidates"
    assert expensive_gate.type == "model_vision_query"
    assert expensive_gate.id == "check_action"
    assert expensive_gate.params["query"] == "Is the dog lifting the beach ball with its mouth?"
    assert expensive_gate.params["sample_interval_ms"] == 500
    assert expensive_gate.params["min_confidence"] == 0.82
    assert expensive_gate.params["upstream_gate_id"] == "cheap_candidates"
    assert expensive_gate.params["verification_mode"] == "binary"
    assert expensive_gate.params["concurrent_requests"] == 4
    assert expensive_gate.params["min_request_interval_ms"] == 250
    assert expensive_gate.params["queued_frame_dedupe_window_ms"] == 600
    assert runtime.response.cooldown_seconds == 10
    assert runtime.response.on_match_text == "The dog picked up the beach ball."
    assert runtime.response.max_responses == 1


def test_answer_prompt_uses_extract_verifier_mode() -> None:
    planned = FunnelSpec(
        goal="Identify the visible bib number.",
        gates=[
            GateSpec(
                id="person_candidate",
                type="local_yolo_object",
                params={"classes": ["person"], "sample_interval_ms": 200},
            ),
            GateSpec(
                id="confirm_number",
                type="model_vision_query",
                params={
                    "query": "Can you read the bib number?",
                    "verification_mode": "binary",
                    "upstream_gate_id": "person_candidate",
                    "model": "gpt-5.6-luna",
                },
            ),
        ],
        response=ResponseSpec(on_match_text="The runner's bib number is {evidence.text}."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "what number is on the runner's bib? let me know as soon as you find out",
        planned,
        sample_interval_ms=1000,
    )

    expensive_gate = runtime.gates[1]
    assert expensive_gate.params["verification_mode"] == "extract"
    assert expensive_gate.params["query"] == (
        "what number is on the runner's bib? let me know as soon as you find out"
    )
    assert runtime.response.max_responses == 1


def test_answer_prompt_drops_redundant_cheap_terminal_leaf_when_extract_gate_exists() -> None:
    planned = FunnelSpec(
        goal="Identify the visible bib number.",
        gates=[
            GateSpec(
                id="person_candidate",
                type="local_yolo_object",
                params={"classes": ["person"], "sample_interval_ms": 200},
            ),
            GateSpec(
                id="bib_text_candidate",
                type="ocr_text",
                params={"query": "Read visible bib text.", "upstream_gate_id": "person_candidate"},
            ),
            GateSpec(
                id="bib_number_verifier",
                type="model_vision_query",
                params={
                    "query": "what number is on the runner's bib?",
                    "verification_mode": "extract",
                    "upstream_gate_id": "person_candidate",
                    "model": "gpt-5.6-terra",
                },
            ),
        ],
        response=ResponseSpec(on_match_text="The runner's bib number is {evidence.text}."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "what number is on the runner's bib? let me know as soon as you find out",
        planned,
        sample_interval_ms=1000,
    )

    assert [gate.id for gate in runtime.gates] == ["person_candidate", "bib_number_verifier"]
    assert runtime.gates[1].params["verification_mode"] == "extract"


def test_audio_text_terminal_leaf_gets_transcript_model_confirmation() -> None:
    planned = FunnelSpec(
        goal="Alert when someone says urgent.",
        gates=[
            GateSpec(
                id="keyword",
                type="transcript_keyword",
                params={"keywords": ["urgent"]},
            )
        ],
        response=ResponseSpec(on_match_text="Someone said urgent."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when someone says urgent",
        planned,
        sample_interval_ms=1000,
    )

    assert [gate.type for gate in runtime.gates] == [
        "local_realtime_transcription",
        "transcript_keyword",
        "model_transcript_query",
    ]
    assert runtime.gates[1].params["upstream_gate_id"] == "cheap_realtime_transcription"
    assert runtime.gates[2].params["upstream_gate_id"] == "keyword"
    assert runtime.gates[2].params["query"] == "let me know when someone says urgent"


def test_transcript_query_without_upstream_gets_local_transcription() -> None:
    planned = FunnelSpec(
        goal="Semantically check spoken text.",
        gates=[
            GateSpec(
                id="speech_semantic_check",
                type="model_transcript_query",
                params={
                    "query": "Is anyone asking for help?",
                    "verification_mode": "binary",
                    "model": "gpt-5.6-luna",
                },
            )
        ],
        response=ResponseSpec(on_match_text="Someone asked for help."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when someone asks for help",
        planned,
        sample_interval_ms=1000,
    )

    assert [gate.type for gate in runtime.gates] == [
        "local_realtime_transcription",
        "transcript_keyword",
        "model_transcript_query",
    ]
    assert runtime.gates[1].params["upstream_gate_id"] == "cheap_realtime_transcription"
    assert "help" in runtime.gates[1].params["keywords"]
    assert runtime.gates[2].params["upstream_gate_id"] == runtime.gates[1].id


def test_audio_runtime_spec_adds_transcription_and_extract_query_for_who_prompt() -> None:
    planned = FunnelSpec(
        goal="let me know who brings the umbrella as soon as you know",
        gates=[],
        response=ResponseSpec(on_match_text="let me know who brings the umbrella as soon as you know"),
    )

    runtime = _audio_runtime_spec_from_planned_spec(
        "let me know who brings the umbrella as soon as you know",
        planned,
    )

    assert [gate.type for gate in runtime.gates] == [
        "local_realtime_transcription",
        "transcript_keyword",
        "model_transcript_query",
    ]
    assert runtime.gates[0].params["model"] == "small.en"
    assert runtime.gates[1].params["upstream_gate_id"] == "cheap_realtime_transcription"
    assert runtime.gates[1].params["keywords"] == ["umbrella"]
    assert runtime.gates[2].params["verification_mode"] == "extract"
    assert runtime.gates[2].params["upstream_gate_id"] == runtime.gates[1].id
    assert runtime.response.on_match_text == "{evidence.text}"
    assert runtime.response.max_responses == 1


def test_audio_runtime_spec_uses_keyword_trigger_for_blue_umbrella_prompt() -> None:
    planned = FunnelSpec(
        goal="Alert as soon as the person who brings a blue umbrella can be identified.",
        gates=[
            GateSpec(
                id="answer_from_transcript",
                type="model_transcript_query",
                params={
                    "query": "let me know who brings the blue umbrella as soon as you know",
                    "model": "gpt-5.6-luna",
                    "verification_mode": "extract",
                },
            )
        ],
        response=ResponseSpec(on_match_text="{evidence.text} brought the blue umbrella."),
    )

    runtime = _audio_runtime_spec_from_planned_spec(
        "let me know who brings the blue umbrella as soon as you know",
        planned,
    )

    assert [gate.type for gate in runtime.gates] == [
        "local_realtime_transcription",
        "transcript_keyword",
        "model_transcript_query",
    ]
    assert runtime.gates[1].params["keywords"] == ["blue", "umbrella"]
    assert runtime.gates[2].params["upstream_gate_id"] == runtime.gates[1].id


def test_audio_runtime_spec_makes_who_answer_prompt_one_shot_without_asap_phrase() -> None:
    planned = FunnelSpec(
        goal="Identify and notify me about the person who brought the blue umbrella.",
        gates=[
            GateSpec(
                id="answer_from_transcript",
                type="model_transcript_query",
                params={
                    "query": "let me know who brought the blue umbrella",
                    "model": "gpt-5.6-luna",
                    "verification_mode": "extract",
                },
            )
        ],
        response=ResponseSpec(on_match_text="The person bringing the blue umbrella is {evidence.text}."),
    )

    runtime = _audio_runtime_spec_from_planned_spec(
        "let me know who brings the blue umbrella",
        planned,
    )

    assert runtime.response.max_responses == 1


def test_audio_runtime_spec_caps_planner_transcription_buffer_for_latency() -> None:
    planned = FunnelSpec(
        goal="Identify who brought the blue umbrella.",
        gates=[
            GateSpec(
                id="cheap_realtime_transcription",
                type="local_realtime_transcription",
                params={
                    "model": "small.en",
                    "buffer_ms": 4000,
                    "sample_interval_ms": 1500,
                    "compute_type": "int8",
                },
            ),
            GateSpec(
                id="answer_from_transcript",
                type="model_transcript_query",
                params={
                    "query": "let me know who brought the blue umbrella",
                    "upstream_gate_id": "cheap_realtime_transcription",
                    "verification_mode": "extract",
                },
            ),
        ],
        response=ResponseSpec(on_match_text="{evidence.text}"),
    )

    runtime = _audio_runtime_spec_from_planned_spec(
        "let me know who brought the blue umbrella",
        planned,
    )

    assert runtime.gates[0].params["buffer_ms"] == 2000
    assert runtime.gates[0].params["sample_interval_ms"] == 1000


def test_audio_runtime_spec_drops_visual_gates_from_audio_only_plan() -> None:
    planned = FunnelSpec(
        goal="Audio-only prompt.",
        gates=[
            GateSpec(
                id="visual_person",
                type="local_yolo_object",
                params={"classes": ["person"]},
            ),
            GateSpec(
                id="vision_check",
                type="model_vision_query",
                params={"query": "Is there a person?", "upstream_gate_id": "visual_person"},
            ),
            GateSpec(
                id="speech_check",
                type="model_transcript_query",
                params={"query": "Does someone mention an umbrella?", "verification_mode": "binary"},
            ),
        ],
        response=ResponseSpec(on_match_text="Umbrella mentioned."),
    )

    runtime = _audio_runtime_spec_from_planned_spec(
        "let me know when someone mentions an umbrella",
        planned,
    )

    assert [gate.type for gate in runtime.gates] == [
        "local_realtime_transcription",
        "transcript_keyword",
        "model_transcript_query",
    ]
    assert runtime.gates[1].params["keywords"] == ["mentions", "umbrella"]
    assert runtime.gates[2].params["upstream_gate_id"] == runtime.gates[1].id
    assert all("vision" not in gate.type and "yolo" not in gate.type for gate in runtime.gates)


def test_video_runtime_spec_normalizes_temporal_frame_change_gate() -> None:
    planned = FunnelSpec(
        goal="Let the user know when hands separate.",
        gates=[
            GateSpec(
                id="hands_separate_change",
                type="model_frame_change",
                params={
                    "subject": "hands",
                    "change": "hands separate from being together",
                    "baseline_mode": "previous",
                    "model": "gpt-5.6-terra",
                },
            )
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the hands separate",
        planned,
        sample_interval_ms=1000,
        verifier_model="gpt-5.6-luna",
        verifier_concurrent_requests=4,
        verifier_min_request_interval_ms=200,
        verifier_queued_frame_dedupe_window_ms=300,
    )

    gate = runtime.gates[0]
    assert gate.type == "model_frame_change"
    assert gate.params["baseline_mode"] == "initial"
    assert gate.params["model"] == "gpt-5.6-luna"
    assert gate.params["sample_interval_ms"] == 1000
    assert gate.params["query"] == "let me know when the hands separate"
    assert gate.params["concurrent_requests"] == 4
    assert gate.params["min_request_interval_ms"] == 200
    assert gate.params["queued_frame_dedupe_window_ms"] == 300
    assert gate.params["required_count"] == 2
    assert gate.params["confirmation_window_ms"] == 2500
    assert runtime.response.max_responses == 1


def test_video_runtime_spec_does_not_double_confirm_frame_change_with_downstream_gate() -> None:
    planned = FunnelSpec(
        goal="Let the user know when hands separate.",
        gates=[
            GateSpec(
                id="hands_separate_change",
                type="model_frame_change",
                params={"subject": "hands", "change": "hands separate"},
            ),
            GateSpec(
                id="confirm_hands_separate",
                type="temporal_count",
                params={"upstream_gate_id": "hands_separate_change", "required_count": 2, "window_ms": 1500},
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the hands separate",
        planned,
        sample_interval_ms=1000,
    )

    gate = runtime.gates[0]
    assert gate.type == "model_frame_change"
    assert "required_count" not in gate.params
    assert "confirmation_window_ms" not in gate.params


def test_video_runtime_spec_collapses_model_stage_after_model_stage() -> None:
    planned = FunnelSpec(
        goal="Let the user know when hands separate.",
        gates=[
            GateSpec(
                id="hands_separate_change",
                type="model_frame_change",
                params={"subject": "hands", "change": "hands separate"},
            ),
            GateSpec(
                id="verify_hands_separate",
                type="model_vision_query",
                params={
                    "upstream_gate_id": "hands_separate_change",
                    "query": "Are the hands separated now?",
                    "verification_mode": "binary",
                },
            ),
            GateSpec(
                id="response_tree",
                type="temporal_count",
                params={"upstream_gate_id": "verify_hands_separate", "required_count": 1, "window_ms": 1000},
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the hands separate",
        planned,
        sample_interval_ms=1000,
    )

    assert [gate.id for gate in runtime.gates] == ["hands_separate_change", "response_tree"]
    assert runtime.gates[1].params["upstream_gate_id"] == "hands_separate_change"


def test_video_runtime_spec_does_not_append_second_model_stage_after_response_tree() -> None:
    planned = FunnelSpec(
        goal="Let the user know when hands separate.",
        gates=[
            GateSpec(
                id="hands_separate_change",
                type="model_frame_change",
                params={"subject": "hands", "change": "hands separate"},
            ),
            GateSpec(
                id="response_tree",
                type="temporal_count",
                params={"upstream_gate_id": "hands_separate_change", "required_count": 1, "window_ms": 1000},
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the hands separate",
        planned,
        sample_interval_ms=1000,
    )

    assert [gate.type for gate in runtime.gates].count("model_frame_change") == 1
    assert [gate.type for gate in runtime.gates].count("model_vision_query") == 0
    assert runtime.gates[-1].id == "response_tree"


def test_video_runtime_spec_adds_model_confirmation_when_planner_omits_gates() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the beachball",
        planned,
        sample_interval_ms=1000,
    )

    assert len(runtime.gates) == 1
    assert runtime.gates[0].id == "model_user_alert_confirm"
    assert runtime.gates[0].type == "model_vision_query"
    assert runtime.gates[0].params["query"] == "let me know when the dog picks up the beachball"
    assert runtime.response.on_match_text == "The dog picked up the beachball"


def test_video_runtime_spec_preserves_non_object_planner_gates() -> None:
    planned = FunnelSpec(
        goal="Alert when checkout appears.",
        gates=[
            GateSpec(
                id="checkout_text",
                type="ocr_text",
                params={"keywords": ["checkout"], "sample_interval_ms": 500},
            )
        ],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=10,
            style="Brief alert",
            on_match_text="Checkout appeared.",
            on_no_match="ignore",
        ),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when checkout appears",
        planned,
        sample_interval_ms=1000,
    )

    assert len(runtime.gates) == 2
    assert runtime.gates[0].id == "checkout_text"
    assert runtime.gates[0].type == "ocr_text"
    assert runtime.gates[0].params["keywords"] == ["checkout"]
    assert runtime.gates[1].type == "model_vision_query"
    assert runtime.gates[1].params["upstream_gate_id"] == "checkout_text"
    assert runtime.gates[1].params["query"] == "let me know when checkout appears"


def test_video_runtime_spec_appends_model_confirmation_to_cheap_terminal_leaf() -> None:
    planned = FunnelSpec(
        goal="Alert when fruit text appears.",
        gates=[
            GateSpec(id="fruit_ocr", type="ocr_text", params={"classes": ["apple"], "sample_interval_ms": 500}),
            GateSpec(
                id="confirm_fruit_ocr",
                type="temporal_count",
                params={"upstream_gate_id": "fruit_ocr", "required_count": 2, "window_ms": 1500},
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when a fruit word shows up on screen",
        planned,
        sample_interval_ms=1000,
        verifier_concurrent_requests=4,
        verifier_min_request_interval_ms=200,
        verifier_queued_frame_dedupe_window_ms=300,
    )

    assert runtime.gates[-1].id == "confirm_fruit_ocr_model_confirm"
    assert runtime.gates[-1].type == "model_vision_query"
    assert runtime.gates[-1].params["upstream_gate_id"] == "confirm_fruit_ocr"
    assert runtime.gates[-1].params["concurrent_requests"] == 4


def test_video_runtime_spec_can_override_verifier_model() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[
            GateSpec(
                id="check_action",
                type="model_vision_query",
                params={"query": "Is the dog holding the ball?", "model": "gpt-5.6-luna"},
            )
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the beachball",
        planned,
        sample_interval_ms=1000,
        verifier_model="gpt-5.4-mini",
    )

    assert runtime.gates[0].params["model"] == "gpt-5.4-mini"


def test_video_runtime_spec_can_override_verifier_queue_settings() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[
            GateSpec(
                id="check_action",
                type="model_vision_query",
                params={
                    "query": "Is the dog holding the ball?",
                    "concurrent_requests": 2,
                    "min_request_interval_ms": 500,
                    "queued_frame_dedupe_window_ms": 700,
                },
            )
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the beachball",
        planned,
        sample_interval_ms=1000,
        verifier_concurrent_requests=4,
        verifier_min_request_interval_ms=300,
        verifier_queued_frame_dedupe_window_ms=300,
    )

    gate = runtime.gates[0]
    assert gate.params["concurrent_requests"] == 4
    assert gate.params["min_request_interval_ms"] == 300
    assert gate.params["queued_frame_dedupe_window_ms"] == 300


def test_video_runtime_spec_can_override_verifier_sample_interval() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[
            GateSpec(
                id="objects",
                type="local_yolo_object",
                params={"classes": ["dog", "sports ball"], "sample_interval_ms": 200},
            ),
            GateSpec(
                id="check_action",
                type="model_vision_query",
                params={
                    "query": "Is the dog holding the ball?",
                    "sample_interval_ms": 1000,
                    "upstream_gate_id": "objects",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the beachball",
        planned,
        sample_interval_ms=1000,
        verifier_sample_interval_ms=300,
    )

    assert runtime.gates[0].params["sample_interval_ms"] == 200
    assert runtime.gates[1].params["sample_interval_ms"] == 300


def test_video_runtime_spec_defers_ocr_state_updates_to_confirmed_response() -> None:
    planned = FunnelSpec(
        goal="Count food words.",
        gates=[
            GateSpec(
                id="food_word_ocr",
                type="ocr_text",
                params={
                    "classes": ["food"],
                    "use_tesseract": True,
                    "state_updates": [
                        {"key": "food_word_count", "operation": "increment", "value": 1},
                    ],
                },
            ),
            GateSpec(
                id="verify_food_word",
                type="model_vision_query",
                params={
                    "upstream_gate_id": "food_word_ocr",
                    "query": "Is a food word visible?",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(
            model="local",
            cooldown_seconds=0,
            style="Brief alert",
            on_match_text="Food word count: {state.food_word_count}",
            state_updates=[StateUpdateSpec(key="count", operation="increment", value=1)],
        ),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "increment a count and announce it everytime a food word shows up on screen",
        planned,
        sample_interval_ms=1000,
    )

    assert "state_updates" not in runtime.gates[0].params
    assert "food" in runtime.gates[0].params["classes"]
    assert runtime.response.state_updates == [
        StateUpdateSpec(
            key="food_word_count",
            operation="increment_unique",
            value="{evidence.matches}",
        )
    ]


def test_video_runtime_spec_broadens_query_upstream_from_spatial_relation() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["dog", "sports ball"]}),
            GateSpec(
                id="near",
                type="object_spatial_relation",
                params={"upstream_gate_id": "objects", "subject": "dog", "object": "sports ball"},
            ),
            GateSpec(
                id="verify",
                type="model_vision_query",
                params={
                    "upstream_gate_id": "near",
                    "query": "Is the dog picking up the ball?",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the ball",
        planned,
        sample_interval_ms=1000,
    )

    assert runtime.gates[2].params["upstream_gate_id"] == "objects"


def test_video_runtime_spec_broadens_query_upstream_through_local_filters() -> None:
    planned = FunnelSpec(
        goal="Alert when the dog picks up the ball.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["dog", "sports ball"]}),
            GateSpec(
                id="near",
                type="object_spatial_relation",
                params={"upstream_gate_id": "objects", "subject": "dog", "object": "sports ball"},
            ),
            GateSpec(
                id="near_twice",
                type="temporal_count",
                params={"upstream_gate_id": "near", "required_count": 2, "window_ms": 1000},
            ),
            GateSpec(
                id="verify",
                type="model_vision_query",
                params={
                    "upstream_gate_id": "near_twice",
                    "query": "Is the dog picking up the ball?",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", cooldown_seconds=10, style="Brief alert"),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the dog picks up the ball",
        planned,
        sample_interval_ms=1000,
    )

    assert runtime.gates[3].params["upstream_gate_id"] == "objects"


def test_video_runtime_spec_drops_audio_gates_when_video_has_no_audio() -> None:
    planned = FunnelSpec(
        goal="Alert when a visible event happens.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["person"]}),
            GateSpec(id="transcript", type="local_realtime_transcription", params={"model": "small.en"}),
            GateSpec(
                id="spoken_event_words",
                type="transcript_keyword",
                params={"keywords": ["event happened"], "upstream_gate_id": "transcript"},
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="The event happened."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when a visible event happens",
        planned,
        sample_interval_ms=1000,
        media_has_audio=False,
    )

    gate_types = [gate.type for gate in runtime.gates]

    assert "local_realtime_transcription" not in gate_types
    assert "transcript_keyword" not in gate_types
    assert "local_yolo_object" in gate_types


def test_video_runtime_spec_adds_generic_visual_confirmation_to_audio_first_plan() -> None:
    planned = FunnelSpec(
        goal="Alert when the requested visible event happens.",
        gates=[
            GateSpec(id="live_audio_transcript", type="local_realtime_transcription", params={"model": "small.en"}),
            GateSpec(
                id="audio_keyword_candidate",
                type="transcript_keyword",
                params={
                    "keywords": ["done", "finished"],
                    "upstream_gate_id": "live_audio_transcript",
                },
            ),
            GateSpec(
                id="verify_audio_event",
                type="model_transcript_query",
                params={
                    "query": "Did the transcript clearly say the requested event happened?",
                    "upstream_gate_id": "audio_keyword_candidate",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="The requested event happened."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the requested visible event happens",
        planned,
        sample_interval_ms=1000,
        verifier_concurrent_requests=4,
        verifier_min_request_interval_ms=200,
        verifier_queued_frame_dedupe_window_ms=300,
        media_has_audio=True,
    )

    visual_confirm = [gate for gate in runtime.gates if gate.id == "generic_visual_change_confirm"]

    assert len(visual_confirm) == 1
    assert visual_confirm[0].type == "model_frame_change"
    assert visual_confirm[0].params["query"] == "let me know when the requested visible event happens"
    assert visual_confirm[0].params["baseline_mode"] == "previous"
    assert visual_confirm[0].params["concurrent_requests"] == 4
    assert visual_confirm[0].params["queued_frame_dedupe_window_ms"] == 300


def test_temporal_visual_query_gets_recent_frame_window() -> None:
    planned = FunnelSpec(
        goal="Alert when a visual action completes.",
        gates=[
            GateSpec(id="candidate", type="local_yolo_object", params={"classes": ["person"]}),
            GateSpec(
                id="verify_action",
                type="model_vision_query",
                params={
                    "query": "Determine whether the current sequence shows the action completing.",
                    "upstream_gate_id": "candidate",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="The action completed."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the visual action completes",
        planned,
        sample_interval_ms=1000,
    )

    assert runtime.gates[1].params["window_ms"] == 3000
    assert runtime.gates[1].params["required_count"] == 2
    assert runtime.gates[1].params["confirmation_window_ms"] >= 4500


def test_visual_only_join_upstream_is_broadened_for_model_verifier() -> None:
    planned = FunnelSpec(
        goal="Alert from visual candidates.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["person"]}),
            GateSpec(id="motion", type="motion", params={}),
            GateSpec(id="screen_text", type="ocr_text", params={"keywords": ["out"]}),
            GateSpec(
                id="visual_join",
                type="temporal_join",
                params={"gate_ids": ["objects", "motion", "screen_text"], "join_window_ms": 4000},
            ),
            GateSpec(
                id="verify_event",
                type="model_vision_query",
                params={
                    "query": "Did the requested visual event happen?",
                    "upstream_gate_id": "visual_join",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="Event happened."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the visual event happens",
        planned,
        sample_interval_ms=500,
    )

    assert runtime.gates[-1].params["upstream_gate_id"] == "objects"
    assert runtime.gates[-1].params["required_count"] == 2


def test_sports_outcome_verifier_uses_broad_visual_trigger_with_state_transition_guard() -> None:
    planned = FunnelSpec(
        goal="Alert when a pitcher completes a strikeout.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["person", "sports ball"]}),
            GateSpec(id="motion", type="motion", params={}),
            GateSpec(id="scorebug", type="ocr_text", params={"keywords": ["K", "strikeout", "out"]}),
            GateSpec(
                id="strikeout_candidate",
                type="temporal_join",
                params={"gate_ids": ["objects", "motion", "scorebug"], "join_window_ms": 8000},
            ),
            GateSpec(
                id="verify_strikeout",
                type="model_vision_query",
                params={
                    "query": "Return YES only if the current live baseball play shows the pitcher has just struck out the batter.",
                    "upstream_gate_id": "strikeout_candidate",
                    "verification_mode": "binary",
                    "window_ms": 3000,
                },
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="Strikeout."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the pitcher strikes someone out",
        planned,
        sample_interval_ms=500,
    )

    verifier = runtime.gates[-1]
    assert verifier.params["upstream_gate_id"] == "objects"
    assert verifier.params["required_count"] == 2
    assert verifier.params["confirmation_window_ms"] >= 4500
    assert "persistent state display" in verifier.params["query"]
    assert "same relevant state" in verifier.params["query"]


def test_audio_video_join_upstream_is_not_broadened_for_model_verifier() -> None:
    planned = FunnelSpec(
        goal="Alert from audio and video candidates.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["person"]}),
            GateSpec(id="keyword", type="transcript_keyword", params={"keywords": ["go"]}),
            GateSpec(
                id="av_join",
                type="temporal_join",
                params={"gate_ids": ["objects", "keyword"], "join_window_ms": 4000},
            ),
            GateSpec(
                id="verify_event",
                type="model_vision_query",
                params={
                    "query": "Did the requested audiovisual event happen?",
                    "upstream_gate_id": "av_join",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="Event happened."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the audiovisual event happens",
        planned,
        sample_interval_ms=500,
    )

    assert runtime.gates[-1].params["upstream_gate_id"] == "av_join"


def test_ocr_upstream_is_broadened_to_available_visual_candidate() -> None:
    planned = FunnelSpec(
        goal="Alert from visible text and action.",
        gates=[
            GateSpec(id="objects", type="local_yolo_object", params={"classes": ["person"]}),
            GateSpec(id="screen_text", type="ocr_text", params={"keywords": ["out"]}),
            GateSpec(
                id="verify_event",
                type="model_vision_query",
                params={
                    "query": "Did the visible state and action confirm the event?",
                    "upstream_gate_id": "screen_text",
                    "verification_mode": "binary",
                },
            ),
        ],
        response=ResponseSpec(model="local", on_match_text="Event happened."),
    )

    runtime = _video_runtime_spec_from_planned_spec(
        "let me know when the visible event happens",
        planned,
        sample_interval_ms=500,
    )

    assert runtime.gates[-1].params["upstream_gate_id"] == "objects"


def test_resolve_ms_accepts_seconds_alias() -> None:
    assert _resolve_ms(None, 0.3) == 300
    assert _resolve_ms(250, 0.3) == 250
    assert _resolve_ms(None, None) is None
