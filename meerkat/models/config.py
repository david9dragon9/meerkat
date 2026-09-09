from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from typing import Optional

from meerkat.models.provider import Provider, active_provider


@dataclass(frozen=True)
class ModelConfig:
    """The model tiers a funnel can draw on.

    Defaults come from the active provider, so the same funnel works on OpenAI,
    Anthropic, or Fireworks without edits. Any tier can be pinned with the
    matching ``MEERKAT_*_MODEL`` variable.
    """

    planner_model: str
    cheap_model: str
    mid_model: str
    responder_model: str

    @property
    def tiers(self) -> frozenset:
        """Every model name this configuration is allowed to select."""
        return frozenset({self.planner_model, self.cheap_model, self.mid_model, self.responder_model})

    @classmethod
    def for_provider(cls, provider: Provider) -> "ModelConfig":
        return cls(
            planner_model=getenv("MEERKAT_PLANNER_MODEL", provider.planner_model),
            cheap_model=getenv("MEERKAT_CHEAP_MODEL", provider.cheap_model),
            mid_model=getenv("MEERKAT_MID_MODEL", provider.mid_model),
            responder_model=getenv("MEERKAT_RESPONDER_MODEL", provider.responder_model),
        )

    @classmethod
    def from_env(cls, provider: Optional[Provider] = None) -> "ModelConfig":
        return cls.for_provider(provider or active_provider())
