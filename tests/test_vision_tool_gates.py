from __future__ import annotations

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.funnel.factory import build_gate
from meerkat.funnel.spec import GateSpec
from meerkat.gates.temporal_vision import ObjectTrackGate
from meerkat.gates.vision_tools import OCRTextGate, ObjectSpatialRelationGate


async def test_ocr_text_gate_matches_payload_text() -> None:
    gate = OCRTextGate(gate_id="ocr", keywords=["checkout"], sample_interval_ms=0)
    event = StreamEvent(
        type=EventType.VIDEO_FRAME,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"ocr_text": "Proceed to checkout"},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.gate_id == "ocr"
    assert fire.evidence["matches"] == ["checkout"]


async def test_ocr_text_gate_matches_whole_food_words_and_expands_food_keyword() -> None:
    gate = OCRTextGate(gate_id="ocr", keywords=["food"], sample_interval_ms=0)
    first = StreamEvent(
        type=EventType.VIDEO_FRAME,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"ocr_text": "WAFFLE PANCAKE"},
    )
    second = StreamEvent(
        type=EventType.VIDEO_FRAME,
        source_id="test",
        stream_time_ms=1001,
        sequence_id=2,
        payload={"ocr_text": "PANCAKE"},
    )

    first_fire = await gate.process(first)
    second_fire = await gate.process(second)

    assert first_fire is not None
    assert first_fire.evidence["matches"] == ["pancake", "waffle"]
    assert second_fire is not None
    assert second_fire.evidence["matches"] == ["pancake"]


async def test_ocr_text_gate_does_not_match_keyword_inside_larger_word() -> None:
    gate = OCRTextGate(gate_id="ocr", keywords=["cake"], sample_interval_ms=0)
    event = StreamEvent(
        type=EventType.VIDEO_FRAME,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"ocr_text": "PANCAKE"},
    )

    fire = await gate.process(event)

    assert fire is None


async def test_ocr_text_gate_without_keywords_emits_text_candidate() -> None:
    gate = OCRTextGate(gate_id="ocr", keywords=[], sample_interval_ms=0)
    frame = object()
    event = StreamEvent(
        type=EventType.VIDEO_FRAME,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"ocr_text": "apple", "frame": frame},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.gate_id == "ocr"
    assert fire.evidence["matches"] == ["text"]
    assert fire.evidence["frame"] is frame


async def test_ocr_text_gate_uses_classes_as_keywords_from_factory() -> None:
    gate = build_gate(
        GateSpec(
            id="fruit_word_ocr",
            type="ocr_text",
            params={"classes": ["apple", "banana"], "sample_interval_ms": 0},
        )
    )
    event = StreamEvent(
        type=EventType.VIDEO_FRAME,
        source_id="test",
        stream_time_ms=1000,
        sequence_id=1,
        payload={"ocr_text": "apple"},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.evidence["matches"] == ["apple"]


async def test_object_spatial_relation_gate_matches_nearby_boxes() -> None:
    gate = ObjectSpatialRelationGate(
        gate_id="dog_near_ball",
        upstream_gate_id="objects",
        subject="dog",
        object_="beach ball",
        relation="near",
    )
    upstream = GateFire(
        gate_id="objects",
        confidence=0.9,
        reason="objects",
        evidence={
            "stream_time_ms": 1200,
            "detections": [
                {"label": "dog", "confidence": 0.9, "bbox": [10, 10, 60, 60]},
                {"label": "sports ball", "confidence": 0.8, "bbox": [65, 20, 95, 50]},
            ],
        },
    )
    event = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=1200,
        sequence_id=2,
        payload={"gate_fire": upstream},
    )

    fire = await gate.process(event)

    assert fire is not None
    assert fire.gate_id == "dog_near_ball"
    assert fire.evidence["relation"] == "near"
    assert [item["label"] for item in fire.evidence["detections"]] == ["dog", "sports ball"]


async def test_object_track_gate_detects_moving_apart() -> None:
    gate = ObjectTrackGate(
        gate_id="track_apart",
        upstream_gate_id="objects",
        labels=["left_hand", "right_hand"],
        change="moving_apart",
        window_ms=2000,
        min_delta_ratio=0.5,
    )
    first = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=0,
        sequence_id=1,
        payload={
            "gate_fire": GateFire(
                gate_id="objects",
                confidence=0.9,
                reason="objects",
                evidence={
                    "stream_time_ms": 0,
                    "detections": [
                        {"label": "left_hand", "bbox": [0, 0, 20, 20]},
                        {"label": "right_hand", "bbox": [25, 0, 45, 20]},
                    ],
                },
            )
        },
    )
    second = StreamEvent(
        type=EventType.GATE_FIRE,
        source_id="test",
        stream_time_ms=500,
        sequence_id=2,
        payload={
            "gate_fire": GateFire(
                gate_id="objects",
                confidence=0.9,
                reason="objects",
                evidence={
                    "stream_time_ms": 500,
                    "detections": [
                        {"label": "left_hand", "bbox": [0, 0, 20, 20]},
                        {"label": "right_hand", "bbox": [80, 0, 100, 20]},
                    ],
                },
            )
        },
    )

    assert await gate.process(first) is None
    fire = await gate.process(second)

    assert fire is not None
    assert fire.gate_id == "track_apart"
    assert fire.evidence["change"] == "moving_apart"
