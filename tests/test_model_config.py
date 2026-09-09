from __future__ import annotations

from meerkat.models.config import ModelConfig
from meerkat.models.provider import OpenAIProvider


def test_openai_tiers_span_cheap_to_strong() -> None:
    config = ModelConfig.for_provider(OpenAIProvider(client=object()))

    assert config.planner_model == "gpt-5.6-sol"
    assert config.cheap_model == "gpt-5.6-luna"
    assert config.mid_model == "gpt-5.6-terra"
    assert config.responder_model == "gpt-5.6-sol"
    assert config.tiers == {"gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"}
