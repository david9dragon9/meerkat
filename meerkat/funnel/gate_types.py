"""Gate type names, and which of them cost a model call.

Gate types are part of the funnel wire format: the planner emits them, the
factory builds from them, and the UI displays them. They are named for what a
gate *does*, never for a vendor, so a plan reads the same whichever provider is
active.
"""

from __future__ import annotations

from typing import FrozenSet

#: Gates that reach a model provider. Everything else runs locally.
MODEL_BACKED_GATE_TYPES: FrozenSet[str] = frozenset(
    {
        "model_vision_object",
        "model_vision_query",
        "model_frame_change",
        "model_state_monitor",
        "model_transcript_query",
        "hosted_transcription",
    }
)

#: Model-backed gates that can confirm a condition well enough to alert a user.
#: `model_state_monitor` only describes state, and `hosted_transcription` only
#: produces text, so neither can stand as the final confirmation.
MODEL_CONFIRMATION_GATE_TYPES: FrozenSet[str] = frozenset(
    {
        "model_vision_query",
        "model_frame_change",
        "model_vision_object",
        "model_transcript_query",
    }
)

#: Gates that need a transcript to exist before they can fire.
TRANSCRIPT_CONSUMER_GATE_TYPES: FrozenSet[str] = frozenset(
    {"transcript_keyword", "audio_cadence", "model_transcript_query"}
)

#: Gates that produce a transcript.
TRANSCRIPT_PRODUCER_GATE_TYPES: FrozenSet[str] = frozenset(
    {"local_realtime_transcription", "hosted_transcription"}
)

#: Gates that read audio rather than video.
AUDIO_GATE_TYPES: FrozenSet[str] = (
    TRANSCRIPT_CONSUMER_GATE_TYPES | TRANSCRIPT_PRODUCER_GATE_TYPES | {"audio_volume", "audio_pitch"}
)

#: Gates that read video frames.
VISUAL_GATE_TYPES: FrozenSet[str] = frozenset(
    {
        "object_label",
        "local_yolo_object",
        "local_yolo_segmentation",
        "local_yolo_pose",
        "local_image_classification",
        "model_vision_object",
        "model_vision_query",
        "model_frame_change",
        "model_state_monitor",
        "object_track",
        "ocr_text",
        "motion",
        "color_presence",
        "object_spatial_relation",
    }
)


#: Verification modes where the gate returns a value rather than a yes/no.
#: The value lands in ``evidence["text"]``.
VALUE_EXTRACTION_MODES: FrozenSet[str] = frozenset({"extract", "value", "text"})
