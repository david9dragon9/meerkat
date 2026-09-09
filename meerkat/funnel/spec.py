from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from meerkat.models.config import ModelConfig


@dataclass(frozen=True)
class GateSpec:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateUpdateSpec:
    key: str
    operation: str = "increment"
    value: Any = 1


@dataclass(frozen=True)
class ResponseSpec:
    """How to answer once a funnel fires. ``model="local"`` fills the template without a model call."""

    model: str = "local"
    cooldown_seconds: float = 30.0
    style: str = "brief"
    on_match_text: Optional[str] = None
    on_no_match: str = "ignore"
    max_responses: Optional[int] = None
    state_updates: List[StateUpdateSpec] = field(default_factory=list)


@dataclass(frozen=True)
class FunnelSpec:
    """A monitoring funnel: a goal, the gates that test for it, and how to respond."""

    goal: str
    gates: List[GateSpec]
    response: ResponseSpec = field(default_factory=ResponseSpec)
    version: int = 1

    @classmethod
    def video_query(
        cls,
        goal: str,
        sample_interval_ms: int = 1000,
        query: Optional[str] = None,
        min_confidence: float = 0.75,
        cooldown_seconds: float = 5.0,
        style: str = "brief",
        candidate_classes: Optional[List[str]] = None,
        on_match_text: Optional[str] = None,
        concurrent_requests: int = 3,
        min_request_interval_ms: int = 200,
        queued_frame_dedupe_window_ms: int = 500,
        service_tier: str = "default",
        verifier_model: Optional[str] = None,
    ) -> "FunnelSpec":
        """Build the default funnel: cheap local candidates feeding one model verifier."""
        verifier_model = verifier_model or ModelConfig.from_env().cheap_model
        candidate_gate_id = "cheap_object_candidates"
        if candidate_classes:
            candidate_gate = GateSpec(
                id=candidate_gate_id,
                type="local_yolo_object",
                params={
                    "classes": candidate_classes,
                    "sample_interval_ms": 200,
                    "min_confidence": 0.35,
                    "model_path": "yolov8n.pt",
                    "trigger_on_any_detection": True,
                },
            )
        else:
            candidate_gate_id = "generic_visual_motion_candidate"
            candidate_gate = GateSpec(
                id=candidate_gate_id,
                type="motion",
                params={
                    "sample_interval_ms": 500,
                    "min_motion_ratio": 0.005,
                },
            )
        return cls(
            goal=goal,
            gates=[
                candidate_gate,
                GateSpec(
                    id="vision_query_confirm",
                    type="model_vision_query",
                    params={
                        "query": query or goal,
                        "sample_interval_ms": sample_interval_ms,
                        "min_confidence": min_confidence,
                        "model": verifier_model,
                        "upstream_gate_id": candidate_gate_id,
                        "verification_mode": "binary",
                        "concurrent_requests": concurrent_requests,
                        "min_request_interval_ms": min_request_interval_ms,
                        "queued_frame_dedupe_window_ms": queued_frame_dedupe_window_ms,
                        "service_tier": service_tier,
                    },
                )
            ],
            response=ResponseSpec(
                model="local",
                cooldown_seconds=cooldown_seconds,
                style=style,
                on_match_text=on_match_text,
                on_no_match="ignore",
            ),
        )
