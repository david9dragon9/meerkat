from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Optional

from meerkat.events import GateFire, ModelResponse
from meerkat.funnel.spec import FunnelSpec
from meerkat.runtime.logging import RuntimeLogger


class Responder(ABC):
    def __init__(self) -> None:
        self.logger: Optional[RuntimeLogger] = None

    def set_logger(self, logger: RuntimeLogger) -> None:
        self.logger = logger

    @abstractmethod
    async def respond(self, spec: FunnelSpec, fire: GateFire) -> ModelResponse:
        raise NotImplementedError


class LocalResponder(Responder):
    async def respond(self, spec: FunnelSpec, fire: GateFire) -> ModelResponse:
        template = spec.response.on_match_text or _notification_from_goal(spec.goal)
        text = _format_response_template(template, fire.evidence)
        return ModelResponse(
            text=text,
            trigger_gate_id=fire.gate_id,
            evidence={"reason": fire.reason, **fire.evidence},
        )


def create_model_responder() -> Responder:
    """Build the model-backed responder, importing the OpenAI client lazily."""
    from meerkat.models.planner_client import ModelResponder

    return ModelResponder()


def _notification_from_goal(goal: str) -> str:
    phrase = goal.strip().rstrip(".!?")
    phrase = re.sub(r"^(please\s+)?(let me know|tell me|notify me|alert me)\s+when\s+", "", phrase, flags=re.I)
    phrase = re.sub(r"^(please\s+)?(let me know|tell me|notify me|alert me)\s+if\s+", "", phrase, flags=re.I)
    phrase = re.sub(r"^when\s+", "", phrase, flags=re.I)
    phrase = phrase.strip()
    if not phrase:
        return "The requested event happened."

    replacements = [
        (r"\bpicks up\b", "picked up"),
        (r"\bshows up\b", "showed up"),
        (r"\bappears\b", "appeared"),
        (r"\benters\b", "entered"),
        (r"\bleaves\b", "left"),
        (r"\bstarts\b", "started"),
        (r"\bstops\b", "stopped"),
        (r"\bsays\b", "said"),
        (r"\bis holding\b", "is holding"),
        (r"\bholds\b", "held"),
    ]
    lowered = phrase.lower()
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, lowered, count=1)
        if updated != lowered:
            return _capitalize(updated)
    return _capitalize(lowered)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _format_response_template(template: str, evidence: dict[str, object]) -> str:
    def replace_evidence(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in evidence:
            return match.group(0)
        return str(evidence[key])

    text = re.sub(r"\{evidence\.([A-Za-z_][A-Za-z0-9_]*)\}", replace_evidence, template)

    state = evidence.get("state")
    if not isinstance(state, dict):
        return text

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in state:
            return match.group(0)
        return str(state[key])

    return re.sub(r"\{state\.([A-Za-z_][A-Za-z0-9_]*)\}", replace, text)
