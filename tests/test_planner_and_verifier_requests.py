from __future__ import annotations

from meerkat.gates.model_vision import OBJECT_DETECTION_SCHEMA, QUERY_MATCH_SCHEMA
from meerkat.funnel.factory import _gate_query
from meerkat.models.planner_client import FUNNEL_SCHEMA
from meerkat.models.planner_client import ModelFunnelPlanner
from meerkat.gates.model_vision import ModelVisionQueryGate
from meerkat.events import EventType, StreamEvent
from meerkat.funnel.spec import GateSpec
from meerkat.models.config import ModelConfig
import asyncio

from conftest import StubProvider


class _PlannerWithoutInit(ModelFunnelPlanner):
    """The planner's normalization logic, without touching a provider."""

    def __init__(self) -> None:
        self.config = ModelConfig(
            planner_model="gpt-5.6-sol",
            cheap_model="gpt-5.6-luna",
            mid_model="gpt-5.6-terra",
            responder_model="gpt-5.6-sol",
        )


def test_funnel_schema_uses_responses_text_format_shape() -> None:
    assert FUNNEL_SCHEMA["type"] == "json_schema"
    assert FUNNEL_SCHEMA["name"] == "funnel_spec"
    assert "schema" in FUNNEL_SCHEMA
    assert "json_schema" not in FUNNEL_SCHEMA


def test_vision_schemas_are_named_json_schemas() -> None:
    assert OBJECT_DETECTION_SCHEMA["type"] == "json_schema"
    assert OBJECT_DETECTION_SCHEMA["name"]
    assert QUERY_MATCH_SCHEMA["type"] == "json_schema"
    assert QUERY_MATCH_SCHEMA["name"]


def test_funnel_schema_allows_additional_cheap_audio_and_vision_tools() -> None:
    gate_types = FUNNEL_SCHEMA["schema"]["properties"]["gates"]["items"]["properties"]["type"]["enum"]

    for gate_type in [
        "ocr_text",
        "local_yolo_segmentation",
        "local_yolo_pose",
        "local_image_classification",
        "motion",
        "color_presence",
        "object_spatial_relation",
        "object_track",
        "audio_volume",
        "audio_pitch",
        "audio_cadence",
        "local_realtime_transcription",
        "hosted_transcription",
        "model_transcript_query",
        "temporal_join",
        "model_frame_change",
        "model_state_monitor",
    ]:
        assert gate_type in gate_types

    params = FUNNEL_SCHEMA["schema"]["properties"]["gates"]["items"]["properties"]["params"]
    for key in [
        "baseline_mode",
        "buffer_ms",
        "change",
        "color",
        "compute_type",
        "gate_ids",
        "join_window_ms",
        "language",
        "max_distance_ratio",
        "max_pitch_hz",
        "max_volume_db",
        "max_words_per_minute",
        "min_area_ratio",
        "min_delta_ratio",
        "min_motion_ratio",
        "min_pitch_hz",
        "min_volume_db",
        "min_words_per_minute",
        "object",
        "pattern",
        "pose",
        "relation",
        "state_updates",
        "state_interval_ms",
        "subject",
        "use_tesseract",
    ]:
        assert key in params["properties"]
        assert key in params["required"]


def test_funnel_schema_allows_response_state_updates() -> None:
    response_schema = FUNNEL_SCHEMA["schema"]["properties"]["response"]

    assert "state_updates" in response_schema["required"]
    update_schema = response_schema["properties"]["state_updates"]["items"]
    assert update_schema["additionalProperties"] is False
    assert update_schema["required"] == ["key", "operation", "value"]
    assert "append_text" in update_schema["properties"]["operation"]["enum"]


def test_funnel_schema_allows_gate_state_updates() -> None:
    params = FUNNEL_SCHEMA["schema"]["properties"]["gates"]["items"]["properties"]["params"]
    update_schema = params["properties"]["state_updates"]["items"]

    assert "state_updates" in params["required"]
    assert update_schema["additionalProperties"] is False
    assert "set_text" in update_schema["properties"]["operation"]["enum"]


def test_planner_includes_initial_context_in_its_request() -> None:
    planner = _PlannerWithoutInit()
    planner.provider = StubProvider(
        payload={
            "goal": "watch",
            "gates": [],
            "response": {
                "model": "local",
                "cooldown_seconds": 1,
                "style": "brief",
                "on_match_text": "ok",
                "on_no_match": "ignore",
                "state_updates": [],
            },
        }
    )

    planner.plan("let me know when the hands separate", context="Initial frame: hands are together.")

    request = planner.provider.request
    assert request.schema is FUNNEL_SCHEMA
    assert request.model == "gpt-5.6-sol"
    assert "let me know when the hands separate" in request.text()
    assert "Initial frame: hands are together." in request.text()


def test_frame_change_gate_prefers_explicit_query() -> None:
    query = _gate_query(
        GateSpec(
            id="hands_separate_change",
            type="model_frame_change",
            params={
                "query": "Did the hands separate compared with the previous frame?",
                "subject": "hands",
                "change": "separate from being together",
                "model": "gpt-5.6-luna",
            },
        )
    )

    assert query == "Did the hands separate compared with the previous frame?"


def test_planner_normalizes_unknown_vision_gate_models() -> None:
    planner = _PlannerWithoutInit()

    spec = planner._parse_gate_spec(
        {
            "id": "check_action",
            "type": "model_vision_query",
            "params": {
                "query": "dog picks up beachball",
                "model": "gpt-4.1-mini",
                "sample_interval_ms": 500,
                "min_confidence": 0.75,
                "classes": None,
                "keywords": None,
                "required_count": None,
                "upstream_gate_id": None,
                "window_ms": None,
            },
        }
    )

    assert spec.params["model"] == "gpt-5.6-terra"
    assert "classes" not in spec.params


def test_planner_downgrades_easy_binary_verification_to_luna() -> None:
    planner = _PlannerWithoutInit()

    spec = planner._parse_gate_spec(
        {
            "id": "check_action",
            "type": "model_vision_query",
            "params": {
                "query": "Is the dog picking up the beachball?",
                "model": "gpt-5.6-terra",
                "verification_mode": "binary",
                "sample_interval_ms": 500,
                "min_confidence": 0.75,
            },
        }
    )

    assert spec.params["model"] == "gpt-5.6-luna"


def test_planner_accepts_standard_service_tier_for_verification() -> None:
    planner = _PlannerWithoutInit()

    spec = planner._parse_gate_spec(
        {
            "id": "check_action",
            "type": "model_vision_query",
            "params": {
                "query": "Is the dog picking up the beachball?",
                "model": "gpt-5.6-luna",
                "service_tier": "default",
                "verification_mode": "binary",
            },
        }
    )

    assert spec.params["service_tier"] == "default"


def test_planner_normalizes_response_model_to_local_or_model() -> None:
    planner = _PlannerWithoutInit()

    response = planner._parse_response_spec(
        {
            "model": "gpt-5.6-terra",
            "cooldown_seconds": 10,
            "style": "brief",
            "on_match_text": "The dog picked up the beach ball.",
            "on_no_match": "ignore",
            "state_updates": [
                {"key": "dog_count", "operation": "increment", "value": 1},
            ],
        }
    )

    assert response.model == "local"
    assert response.on_match_text == "The dog picked up the beach ball."
    assert response.state_updates[0].key == "dog_count"
    assert response.state_updates[0].operation == "increment"
    assert response.state_updates[0].value == 1


def test_binary_verifier_parses_yes_no_without_structured_output() -> None:
    gate = ModelVisionQueryGate.__new__(ModelVisionQueryGate)
    gate.verification_mode = "binary"

    assert gate._parse_verification_response("YES") == (
        True,
        1.0,
        "Binary vision verification returned YES.",
        {},
    )
    assert gate._parse_verification_response("NO") == (
        False,
        0.0,
        "Binary vision verification returned NO.",
        {},
    )


async def test_vision_confirmation_reports_latest_confirming_stream_time() -> None:
    gate = ModelVisionQueryGate.__new__(ModelVisionQueryGate)
    gate.confirmation_window_ms = 4500
    gate.required_count = 2
    gate._matched_request_stream_times_ms = __import__("collections").deque(maxlen=16)
    gate._sample_lock = asyncio.Lock()

    assert await gate._confirmation_reached(14170) is None
    assert await gate._confirmation_reached(13670) == 14170


def test_extract_verifier_parses_visible_value_into_evidence() -> None:
    gate = ModelVisionQueryGate.__new__(ModelVisionQueryGate)
    gate.verification_mode = "extract"
    gate.query = "what number is on the runner's bib?"

    assert gate._parse_verification_response("48") == (
        True,
        1.0,
        "Vision verifier extracted value: 48.",
        {"text": "48", "value": "48", "answer": "48"},
    )
    assert gate._parse_verification_response("NO") == (
        False,
        0.0,
        "Vision verifier extraction returned NO.",
        {},
    )


async def test_queued_frame_dedupe_only_skips_when_backlog_exists() -> None:
    gate = ModelVisionQueryGate.__new__(ModelVisionQueryGate)
    gate.queued_frame_dedupe_window_ms = 500
    gate._sent_request_stream_times_ms = __import__("collections").deque([1000], maxlen=128)
    gate._sample_lock = asyncio.Lock()
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    gate.runtime_queue = queue

    assert await gate._should_skip_due_to_nearby_queued_request(1200) is False

    queue.put_nowait(StreamEvent(type=EventType.GATE_FIRE, source_id="test", stream_time_ms=1300, sequence_id=1))

    assert await gate._should_skip_due_to_nearby_queued_request(1200) is True
    assert await gate._should_skip_due_to_nearby_queued_request(1700) is False


def _verifier(mode: str, query: str, provider: StubProvider) -> ModelVisionQueryGate:
    gate = ModelVisionQueryGate.__new__(ModelVisionQueryGate)
    gate.provider = provider
    gate.model = "gpt-5.6-luna"
    gate.query = query
    gate.verification_mode = mode
    gate.service_tier = "default"
    return gate


async def test_binary_verifier_streams_a_short_capped_request() -> None:
    gate = _verifier("binary", "let me know when the dog picks up the beachball", StubProvider("YES"))

    output_text = await gate._request_verification_text(["data:image/jpeg;base64,abc"])

    request = gate.provider.request
    assert output_text == "YES"
    assert request.service_tier == "default"
    assert request.max_output_tokens == 16
    assert request.effort == "none"
    assert request.schema is None


async def test_extract_verifier_streams_a_value_with_room_for_it() -> None:
    gate = _verifier("extract", "what number is on the runner's bib?", StubProvider("48"))

    output_text = await gate._request_verification_text(["data:image/jpeg;base64,abc"])

    request = gate.provider.request
    assert output_text == "48"
    assert request.max_output_tokens == 24
    assert request.service_tier == "default"
    assert request.schema is None


async def test_verifier_carries_current_state_into_the_request() -> None:
    gate = _verifier("binary", "did the count change?", StubProvider("NO"))

    await gate._request_verification_text(["data:image/jpeg;base64,abc"], {"person_count": 2})

    assert "person_count" in gate.provider.request.text()


async def test_binary_verifier_can_send_an_ordered_recent_frame_window() -> None:
    gate = _verifier("binary", "let me know when the visible event happens", StubProvider("YES"))

    output_text = await gate._request_verification_text(
        ["data:image/jpeg;base64,one", "data:image/jpeg;base64,two"]
    )

    request = gate.provider.request
    assert output_text == "YES"
    assert len(request.images()) == 2
    assert [image.data_url for image in request.images()] == [
        "data:image/jpeg;base64,one",
        "data:image/jpeg;base64,two",
    ]
    assert "ordered from earlier to later" in request.text()


def test_service_tier_standard_alias_maps_to_default() -> None:
    gate = ModelVisionQueryGate.__new__(ModelVisionQueryGate)

    assert gate._normalize_service_tier("standard") == "default"
