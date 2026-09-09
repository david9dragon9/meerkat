from __future__ import annotations

import re
from typing import List, Optional

from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec


COMMON_OBJECTS = {
    "dog",
    "cat",
    "person",
    "car",
    "bicycle",
    "bird",
    "laptop",
    "phone",
}


class RuleBasedFunnelPlanner:
    """Offline keyword planner used when no model-backed planner is available."""

    def plan(self, prompt: str, context: Optional[str] = None) -> FunnelSpec:
        prompt_lower = prompt.lower()
        objects = self._find_objects(prompt_lower)
        gates: List[GateSpec] = []

        if objects:
            gates.append(GateSpec(id="object_detector", type="object_label", params={"classes": objects}))
            gates.append(
                GateSpec(
                    id="object_temporal_confirm",
                    type="temporal_count",
                    params={"upstream_gate_id": "object_detector", "required_count": 2, "window_ms": 1500},
                )
            )

        quoted_keywords = re.findall(r'"([^"]+)"|' + r"'([^']+)'", prompt)
        keywords = [left or right for left, right in quoted_keywords]
        if keywords:
            gates.append(GateSpec(id="transcript_keyword", type="transcript_keyword", params={"keywords": keywords}))

        if not gates:
            gates.append(GateSpec(id="user_prompt_gate", type="transcript_keyword", params={"keywords": ["now", "help"]}))

        return FunnelSpec(goal=prompt, gates=gates, response=ResponseSpec(cooldown_seconds=2.0))

    def _find_objects(self, text: str) -> List[str]:
        return sorted(obj for obj in COMMON_OBJECTS if re.search(rf"\b{re.escape(obj)}s?\b", text))


def create_model_planner(config: object = None) -> object:
    """Build the model-backed planner, importing the OpenAI client lazily."""
    from meerkat.models.planner_client import ModelFunnelPlanner

    return ModelFunnelPlanner(config=config)
