from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic_ns
from typing import Any, Dict, Optional
from uuid import uuid4


def now_ms() -> int:
    return monotonic_ns() // 1_000_000


class EventType(str, Enum):
    VIDEO_FRAME = "video_frame"
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPT_DELTA = "transcript_delta"
    USER_PROMPT = "user_prompt"
    GATE_FIRE = "gate_fire"
    MODEL_RESPONSE = "model_response"
    FUNNEL_UPDATE = "funnel_update"
    STREAM_END = "stream_end"


@dataclass(frozen=True)
class StreamEvent:
    type: EventType
    source_id: str
    stream_time_ms: int
    sequence_id: int
    wall_time_ms: int = field(default_factory=now_ms)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateFire:
    gate_id: str
    confidence: float
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    trigger_gate_id: Optional[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
