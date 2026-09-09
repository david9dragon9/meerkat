from __future__ import annotations

from meerkat.funnel.spec import GateSpec
from meerkat.gates.base import Gate
from meerkat.gates.audio import (
    AudioPitchGate,
    AudioVolumeGate,
    LocalRealtimeTranscriptionGate,
    HostedTranscriptionGate,
    ModelTranscriptQueryGate,
    SpeakingCadenceGate,
    TemporalJoinGate,
)
from meerkat.gates.local_vision import LocalYoloObjectGate
from meerkat.gates.object_detection import ObjectLabelGate
from meerkat.gates.model_vision import ModelVisionObjectGate, ModelVisionQueryGate
from meerkat.gates.temporal_vision import ObjectTrackGate, ModelFrameChangeGate, ModelStateMonitorGate
from meerkat.gates.temporal import TemporalCountGate
from meerkat.gates.transcript import TranscriptKeywordGate
from meerkat.gates.vision_tools import (
    ColorPresenceGate,
    LocalImageClassificationGate,
    LocalYoloPoseGate,
    LocalYoloSegmentationGate,
    MotionGate,
    OCRTextGate,
    ObjectSpatialRelationGate,
)


def _gate_query(spec: GateSpec) -> str:
    query = spec.params.get("query")
    if query:
        return str(query)
    subject = spec.params.get("subject")
    change = spec.params.get("change")
    if subject and change:
        return f"{subject}: {change}"
    if change:
        return str(change)
    if subject:
        return str(subject)
    return spec.id.replace("_", " ")


def build_gate(spec: GateSpec) -> Gate:
    if spec.type == "object_label":
        return ObjectLabelGate(gate_id=spec.id, classes=spec.params.get("classes", []))
    if spec.type == "local_yolo_object":
        return LocalYoloObjectGate(
            gate_id=spec.id,
            classes=spec.params.get("classes", []),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 200)),
            min_confidence=float(spec.params.get("min_confidence", 0.35)),
            model_path=str(spec.params.get("model_path", "yolov8n.pt")),
            trigger_on_any_detection=bool(spec.params.get("trigger_on_any_detection", False)),
        )
    if spec.type == "model_vision_object":
        return ModelVisionObjectGate(
            gate_id=spec.id,
            classes=spec.params.get("classes", []),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 1000)),
            min_confidence=float(spec.params.get("min_confidence", 0.6)),
            model=spec.params.get("model"),
        )
    if spec.type == "model_vision_query":
        return ModelVisionQueryGate(
            gate_id=spec.id,
            query=spec.params["query"],
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 1000)),
            min_confidence=float(spec.params.get("min_confidence", 0.75)),
            model=spec.params.get("model"),
            upstream_gate_id=spec.params.get("upstream_gate_id"),
            verification_mode=str(spec.params.get("verification_mode", "binary")),
            concurrent_requests=int(spec.params.get("concurrent_requests", 3)),
            min_request_interval_ms=int(spec.params.get("min_request_interval_ms", 200)),
            queued_frame_dedupe_window_ms=int(spec.params.get("queued_frame_dedupe_window_ms", 500)),
            service_tier=str(spec.params.get("service_tier", "default")),
            window_ms=int(spec.params.get("window_ms", 0)),
            required_count=int(spec.params.get("required_count", 1)),
            confirmation_window_ms=int(spec.params.get("confirmation_window_ms", 1500)),
        )
    if spec.type == "model_frame_change":
        return ModelFrameChangeGate(
            gate_id=spec.id,
            query=_gate_query(spec),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 500)),
            min_confidence=float(spec.params.get("min_confidence", 0.75)),
            model=spec.params.get("model"),
            baseline_mode=str(spec.params.get("baseline_mode", "initial")),
            service_tier=str(spec.params.get("service_tier", "default")),
            state_interval_ms=int(spec.params.get("state_interval_ms", 3000)),
            concurrent_requests=int(spec.params.get("concurrent_requests", 1)),
            min_request_interval_ms=int(spec.params.get("min_request_interval_ms", 200)),
            queued_frame_dedupe_window_ms=int(spec.params.get("queued_frame_dedupe_window_ms", 500)),
            required_count=int(spec.params.get("required_count", 1)),
            confirmation_window_ms=int(spec.params.get("confirmation_window_ms", 1500)),
        )
    if spec.type == "model_state_monitor":
        return ModelStateMonitorGate(
            gate_id=spec.id,
            query=spec.params["query"],
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 3000)),
            model=spec.params.get("model"),
            service_tier=str(spec.params.get("service_tier", "default")),
        )
    if spec.type == "object_track":
        return ObjectTrackGate(
            gate_id=spec.id,
            upstream_gate_id=str(spec.params["upstream_gate_id"]),
            labels=spec.params.get("labels") or spec.params.get("classes") or [],
            change=str(spec.params.get("change", "moving_apart")),
            window_ms=int(spec.params.get("window_ms", 2000)),
            min_delta_ratio=float(spec.params.get("min_delta_ratio", 0.25)),
        )
    if spec.type == "ocr_text":
        return OCRTextGate(
            gate_id=spec.id,
            keywords=spec.params.get("keywords") or spec.params.get("classes") or [],
            pattern=spec.params.get("pattern"),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 1000)),
            min_confidence=float(spec.params.get("min_confidence", 0.5)),
            use_tesseract=bool(spec.params.get("use_tesseract", False)),
        )
    if spec.type == "local_yolo_segmentation":
        return LocalYoloSegmentationGate(
            gate_id=spec.id,
            classes=spec.params.get("classes", []),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 500)),
            min_confidence=float(spec.params.get("min_confidence", 0.35)),
            min_area_ratio=float(spec.params.get("min_area_ratio", 0.0)),
            model_path=str(spec.params.get("model_path", "yolo11n-seg.pt")),
        )
    if spec.type == "local_yolo_pose":
        return LocalYoloPoseGate(
            gate_id=spec.id,
            pose=str(spec.params.get("pose", "person")),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 500)),
            min_confidence=float(spec.params.get("min_confidence", 0.35)),
            model_path=str(spec.params.get("model_path", "yolo11n-pose.pt")),
        )
    if spec.type == "local_image_classification":
        return LocalImageClassificationGate(
            gate_id=spec.id,
            classes=spec.params.get("classes", []),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 1000)),
            min_confidence=float(spec.params.get("min_confidence", 0.35)),
            model_path=str(spec.params.get("model_path", "yolo11n-cls.pt")),
        )
    if spec.type == "motion":
        return MotionGate(
            gate_id=spec.id,
            min_motion_ratio=float(spec.params.get("min_motion_ratio", 0.02)),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 200)),
        )
    if spec.type == "color_presence":
        return ColorPresenceGate(
            gate_id=spec.id,
            color=str(spec.params["color"]),
            min_area_ratio=float(spec.params.get("min_area_ratio", 0.02)),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 200)),
        )
    if spec.type == "object_spatial_relation":
        return ObjectSpatialRelationGate(
            gate_id=spec.id,
            upstream_gate_id=str(spec.params["upstream_gate_id"]),
            subject=str(spec.params["subject"]),
            object_=str(spec.params["object"]),
            relation=str(spec.params.get("relation", "near")),
            max_distance_ratio=float(spec.params.get("max_distance_ratio", 0.25)),
        )
    if spec.type == "transcript_keyword":
        return TranscriptKeywordGate(
            gate_id=spec.id,
            keywords=spec.params.get("keywords", []),
            upstream_gate_id=spec.params.get("upstream_gate_id"),
        )
    if spec.type == "audio_volume":
        return AudioVolumeGate(
            gate_id=spec.id,
            min_volume_db=float(spec.params.get("min_volume_db", -35.0)),
            max_volume_db=(
                float(spec.params["max_volume_db"])
                if spec.params.get("max_volume_db") is not None
                else None
            ),
        )
    if spec.type == "audio_pitch":
        return AudioPitchGate(
            gate_id=spec.id,
            min_pitch_hz=(
                float(spec.params["min_pitch_hz"])
                if spec.params.get("min_pitch_hz") is not None
                else None
            ),
            max_pitch_hz=(
                float(spec.params["max_pitch_hz"])
                if spec.params.get("max_pitch_hz") is not None
                else None
            ),
            min_confidence=float(spec.params.get("min_confidence", 0.4)),
        )
    if spec.type == "audio_cadence":
        return SpeakingCadenceGate(
            gate_id=spec.id,
            min_words_per_minute=(
                float(spec.params["min_words_per_minute"])
                if spec.params.get("min_words_per_minute") is not None
                else None
            ),
            max_words_per_minute=(
                float(spec.params["max_words_per_minute"])
                if spec.params.get("max_words_per_minute") is not None
                else None
            ),
            window_ms=int(spec.params.get("window_ms", 5000)),
            upstream_gate_id=spec.params.get("upstream_gate_id"),
        )
    if spec.type == "local_realtime_transcription":
        return LocalRealtimeTranscriptionGate(
            gate_id=spec.id,
            model=str(spec.params.get("model", "tiny.en")),
            model_path=(
                str(spec.params["model_path"])
                if spec.params.get("model_path") is not None
                else None
            ),
            compute_type=str(spec.params.get("compute_type", "int8")),
            language=(
                str(spec.params["language"])
                if spec.params.get("language") is not None
                else None
            ),
            buffer_ms=min(int(spec.params.get("buffer_ms", 2000)), 2000),
            sample_interval_ms=max(250, min(int(spec.params.get("sample_interval_ms", 1000)), 1000)),
            min_volume_db=float(spec.params.get("min_volume_db", -45.0)),
            text_window_ms=int(spec.params.get("text_window_ms", 12000)),
        )
    if spec.type == "hosted_transcription":
        return HostedTranscriptionGate(
            gate_id=spec.id,
            model=str(spec.params.get("model", "gpt-4o-mini-transcribe")),
            sample_interval_ms=int(spec.params.get("sample_interval_ms", 1000)),
            min_volume_db=float(spec.params.get("min_volume_db", -45.0)),
        )
    if spec.type == "model_transcript_query":
        return ModelTranscriptQueryGate(
            gate_id=spec.id,
            query=str(spec.params["query"]),
            model=spec.params.get("model"),
            upstream_gate_id=spec.params.get("upstream_gate_id"),
            verification_mode=str(spec.params.get("verification_mode", "binary")),
            service_tier=str(spec.params.get("service_tier", "default")),
        )
    if spec.type == "temporal_join":
        return TemporalJoinGate(
            gate_id=spec.id,
            gate_ids=spec.params.get("gate_ids", []),
            join_window_ms=int(spec.params.get("join_window_ms", spec.params.get("window_ms", 1000))),
        )
    if spec.type == "temporal_count":
        return TemporalCountGate(
            gate_id=spec.id,
            upstream_gate_id=spec.params.get("upstream_gate_id"),
            upstream_gate_ids=spec.params.get("gate_ids"),
            required_count=int(spec.params["required_count"]),
            window_ms=int(spec.params["window_ms"]),
        )
    raise ValueError(f"Unknown gate type: {spec.type}")
