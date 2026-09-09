"""Provider selection, and the per-vendor translation of a `ModelRequest`.

The gates speak only `ModelRequest`/`Provider`, so this is the one place that
knows what each vendor's wire format looks like.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from meerkat.funnel.factory import build_gate
from meerkat.funnel.spec import FunnelSpec, GateSpec
from meerkat.models.config import ModelConfig
from meerkat.models.provider import (
    AnthropicProvider,
    FireworksProvider,
    ImagePart,
    ModelBacked,
    ModelRequest,
    OpenAIProvider,
    Provider,
    TextPart,
    UnsupportedCapability,
    available_provider_names,
    create_provider,
    resolve_provider_name,
)

DATA_URL = "data:image/jpeg;base64,QUJD"

SCHEMA = {
    "type": "json_schema",
    "name": "verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["matched"],
        "properties": {"matched": {"type": "boolean"}},
    },
}


def _request(**overrides: Any) -> ModelRequest:
    defaults: Dict[str, Any] = dict(
        model="a-model",
        system="be strict",
        content=[TextPart("is it open?"), ImagePart(DATA_URL)],
        max_output_tokens=16,
        effort="none",
    )
    defaults.update(overrides)
    return ModelRequest(**defaults)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_provider_preference_order_is_openai_then_anthropic_then_fireworks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "f")
    assert resolve_provider_name() == "fireworks"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert resolve_provider_name() == "anthropic"

    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert resolve_provider_name() == "openai"

    assert available_provider_names() == ["openai", "anthropic", "fireworks"]


def test_explicit_override_wins_over_available_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("MEERKAT_PROVIDER", "anthropic")

    assert resolve_provider_name() == "anthropic"


def test_unknown_override_is_rejected_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEERKAT_PROVIDER", "hal9000")

    with pytest.raises(RuntimeError, match="hal9000"):
        resolve_provider_name()


def test_missing_every_key_names_the_variables_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError) as error:
        resolve_provider_name()

    message = str(error.value)
    assert "OPENAI_API_KEY" in message
    assert "ANTHROPIC_API_KEY" in message
    assert "FIREWORKS_API_KEY" in message


def test_create_provider_builds_the_named_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "f")

    assert create_provider("fireworks").name == "fireworks"


# --------------------------------------------------------------------------
# model tiers
# --------------------------------------------------------------------------


def test_model_tiers_follow_the_active_provider() -> None:
    openai_tiers = ModelConfig.for_provider(OpenAIProvider(client=object()))
    anthropic_tiers = ModelConfig.for_provider(AnthropicProvider(client=object()))

    assert openai_tiers.planner_model.startswith("gpt-")
    assert anthropic_tiers.planner_model.startswith("claude-")
    assert anthropic_tiers.cheap_model != anthropic_tiers.planner_model


def test_env_overrides_beat_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEERKAT_CHEAP_MODEL", "pinned-cheap-model")

    assert ModelConfig.for_provider(AnthropicProvider(client=object())).cheap_model == "pinned-cheap-model"


def test_every_provider_defines_all_four_tiers() -> None:
    for provider_class in (OpenAIProvider, AnthropicProvider, FireworksProvider):
        provider = provider_class(client=object())
        config = ModelConfig.for_provider(provider)
        assert all(
            [config.planner_model, config.cheap_model, config.mid_model, config.responder_model]
        ), provider.name


# --------------------------------------------------------------------------
# OpenAI: Responses API
# --------------------------------------------------------------------------


class _RecordingResponses:
    def __init__(self, output_text: str = "YES", events: Optional[List[Any]] = None) -> None:
        self.output_text = output_text
        self.events = events or []
        self.kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self.events)
        return type("Response", (), {"output_text": self.output_text})()


class _OpenAIClient:
    def __init__(self, **kwargs: Any) -> None:
        self.responses = _RecordingResponses(**kwargs)


def test_openai_sends_responses_api_input_with_images() -> None:
    client = _OpenAIClient()
    OpenAIProvider(client=client).complete_text(_request(service_tier="default"))

    kwargs = client.responses.kwargs
    assert kwargs["model"] == "a-model"
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["max_output_tokens"] == 16
    assert kwargs["service_tier"] == "default"
    assert kwargs["input"][0] == {"role": "system", "content": "be strict"}
    assert kwargs["input"][1]["content"] == [
        {"type": "input_text", "text": "is it open?"},
        {"type": "input_image", "image_url": DATA_URL},
    ]
    assert "text" not in kwargs


def test_openai_passes_a_schema_as_a_text_format() -> None:
    client = _OpenAIClient(output_text='{"matched": true}')
    result = OpenAIProvider(client=client).complete_json(_request(schema=SCHEMA))

    assert client.responses.kwargs["text"] == {"format": SCHEMA}
    assert result == {"matched": True}


def test_openai_streams_only_output_text_deltas() -> None:
    events = [
        {"type": "response.created"},
        {"type": "response.output_text.delta", "delta": "Y"},
        {"type": "response.output_text.delta", "delta": "ES"},
        {"type": "response.completed"},
    ]
    client = _OpenAIClient(events=events)

    deltas = list(OpenAIProvider(client=client).stream_text(_request()))

    assert client.responses.kwargs["stream"] is True
    assert deltas == ["Y", "ES"]


# --------------------------------------------------------------------------
# Anthropic: Messages API
# --------------------------------------------------------------------------


class _Block:
    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class _AnthropicMessages:
    def __init__(self, blocks: Optional[List[_Block]] = None, deltas: Optional[List[str]] = None) -> None:
        self.blocks = blocks or [_Block("text", text="YES")]
        self.deltas = deltas or []
        self.kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return type("Message", (), {"content": self.blocks})()

    def stream(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        deltas = self.deltas

        class _Stream:
            text_stream = deltas

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        return _Stream()


class _AnthropicClient:
    def __init__(self, **kwargs: Any) -> None:
        self.messages = _AnthropicMessages(**kwargs)


def test_anthropic_sends_a_system_prompt_and_base64_image_blocks() -> None:
    client = _AnthropicClient()
    text = AnthropicProvider(client=client).complete_text(_request())

    kwargs = client.messages.kwargs
    assert kwargs["model"] == "a-model"
    assert kwargs["system"] == "be strict"
    assert kwargs["max_tokens"] == 16
    assert kwargs["messages"][0]["content"] == [
        {"type": "text", "text": "is it open?"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
        },
    ]
    assert text == "YES"


def test_anthropic_always_supplies_max_tokens() -> None:
    """The Messages API rejects a request without it, so a ceiling is always set."""
    client = _AnthropicClient()
    AnthropicProvider(client=client).complete_text(_request(max_output_tokens=None))

    assert client.messages.kwargs["max_tokens"] > 0


def test_anthropic_uses_a_forced_tool_call_for_schema_output() -> None:
    client = _AnthropicClient(blocks=[_Block("tool_use", input={"matched": True})])

    result = AnthropicProvider(client=client).complete_json(_request(schema=SCHEMA))

    kwargs = client.messages.kwargs
    assert kwargs["tools"][0]["name"] == "verdict"
    assert kwargs["tools"][0]["input_schema"] == SCHEMA["schema"]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "verdict"}
    assert result == {"matched": True}


def test_anthropic_schema_requests_get_a_larger_token_ceiling() -> None:
    client = _AnthropicClient(blocks=[_Block("tool_use", input={"matched": True})])
    AnthropicProvider(client=client).complete_json(_request(max_output_tokens=None, schema=SCHEMA))

    assert client.messages.kwargs["max_tokens"] >= 4096


def test_anthropic_streams_text_deltas() -> None:
    client = _AnthropicClient(deltas=["Y", "ES"])

    assert list(AnthropicProvider(client=client).stream_text(_request())) == ["Y", "ES"]


def test_anthropic_streaming_a_schema_request_falls_back_to_one_call() -> None:
    """Forced tool use cannot be read incrementally, so it is issued whole."""
    client = _AnthropicClient(blocks=[_Block("tool_use", input={"matched": True})])

    chunks = list(AnthropicProvider(client=client).stream_text(_request(schema=SCHEMA)))

    assert json.loads("".join(chunks)) == {"matched": True}


def test_anthropic_reports_no_audio_capabilities() -> None:
    provider = AnthropicProvider(client=object())

    assert not provider.supports_transcription
    assert not provider.supports_speech
    with pytest.raises(UnsupportedCapability):
        provider.transcribe(b"")
    with pytest.raises(UnsupportedCapability):
        provider.synthesize_speech("hello")


# --------------------------------------------------------------------------
# Fireworks: OpenAI-compatible chat completions
# --------------------------------------------------------------------------


class _FireworksCompletions:
    def __init__(self, content: str = "YES", deltas: Optional[List[str]] = None) -> None:
        self.content = content
        self.deltas = deltas or []
        self.kwargs: Dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(
                [
                    type(
                        "Chunk",
                        (),
                        {"choices": [type("Choice", (), {"delta": type("Delta", (), {"content": delta})()})()]},
                    )()
                    for delta in self.deltas
                ]
            )
        message = type("Message", (), {"content": self.content})()
        return type("Completion", (), {"choices": [type("Choice", (), {"message": message})()]})()


class _FireworksClient:
    def __init__(self, **kwargs: Any) -> None:
        completions = _FireworksCompletions(**kwargs)
        self.chat = type("Chat", (), {"completions": completions})()
        self.completions = completions


def test_fireworks_sends_chat_completions_with_image_urls() -> None:
    client = _FireworksClient()
    text = FireworksProvider(client=client).complete_text(_request())

    kwargs = client.completions.kwargs
    assert kwargs["model"] == "a-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "be strict"}
    assert kwargs["messages"][1]["content"] == [
        {"type": "text", "text": "is it open?"},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]
    assert text == "YES"


def test_fireworks_requests_json_schema_response_format() -> None:
    client = _FireworksClient(content='{"matched": false}')

    result = FireworksProvider(client=client).complete_json(_request(schema=SCHEMA))

    assert client.completions.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": SCHEMA["schema"]},
    }
    assert result == {"matched": False}


def test_fireworks_streams_chat_deltas() -> None:
    client = _FireworksClient(deltas=["N", "O"])

    assert list(FireworksProvider(client=client).stream_text(_request())) == ["N", "O"]


def test_fireworks_passes_effort_through_and_keeps_the_caller_cap() -> None:
    """Fireworks shares this codebase's effort vocabulary; "none" disables thinking.

    With thinking off there is no hidden scratchpad to pay for, so the caller's
    token ceiling is sent unchanged and short answers stay short.
    """
    client = _FireworksClient()
    FireworksProvider(client=client).complete_text(_request(effort="none", max_output_tokens=16))

    assert client.completions.kwargs["reasoning_effort"] == "none"
    assert client.completions.kwargs["max_tokens"] == 16


def test_fireworks_budgets_headroom_when_thinking_stays_on() -> None:
    """Thinking is billed against max_tokens while producing no content.

    A cap sized only for the visible answer returns finish_reason="length" and
    a null content, so any request that reasons needs room for the scratchpad.
    """
    client = _FireworksClient()
    FireworksProvider(client=client).complete_text(_request(effort="medium", max_output_tokens=16))

    assert client.completions.kwargs["reasoning_effort"] == "medium"
    assert client.completions.kwargs["max_tokens"] > 16 + 100


def test_fireworks_sends_no_token_cap_when_the_caller_sets_none() -> None:
    client = _FireworksClient()
    FireworksProvider(client=client).complete_text(_request(max_output_tokens=None))

    assert "max_tokens" not in client.completions.kwargs


class _ThinkingOnlyClient(_FireworksClient):
    """Rejects reasoning_effort="none" the way a thinking-only model does."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.attempts: List[str] = []
        inner = self.completions.create

        def create(**call: Any) -> Any:
            self.attempts.append(call.get("reasoning_effort"))
            if call.get("reasoning_effort") == "none":
                raise RuntimeError(
                    "Error code: 400 - GLM-5.3 is a thinking-only model; disabling "
                    "thinking (reasoning_effort='none') is not supported."
                )
            return inner(**call)

        self.completions.create = create  # type: ignore[method-assign]
        self.chat.completions = self.completions


def test_fireworks_retries_thinking_only_models_with_minimal_effort() -> None:
    """Some models refuse to stop thinking; minimise it rather than failing."""
    client = _ThinkingOnlyClient()
    provider = FireworksProvider(client=client)

    assert provider.complete_text(_request(effort="none", max_output_tokens=16)) == "YES"
    assert client.attempts == ["none", "low"]
    assert client.completions.kwargs["max_tokens"] > 16 + 100


def test_fireworks_remembers_a_thinking_only_model_and_stops_retrying() -> None:
    client = _ThinkingOnlyClient()
    provider = FireworksProvider(client=client)

    provider.complete_text(_request(effort="none", max_output_tokens=16))
    client.attempts.clear()
    provider.complete_text(_request(effort="none", max_output_tokens=16))

    assert client.attempts == ["low"]


def test_fireworks_does_not_swallow_unrelated_errors() -> None:
    class _BrokenClient(_FireworksClient):
        def __init__(self) -> None:
            super().__init__()
            def create(**call: Any) -> Any:
                raise RuntimeError("Error code: 429 - rate limited")
            self.completions.create = create  # type: ignore[method-assign]
            self.chat.completions = self.completions

    with pytest.raises(RuntimeError, match="rate limited"):
        FireworksProvider(client=_BrokenClient()).complete_text(_request(effort="none"))


def test_fireworks_transcribes_from_a_separate_audio_host() -> None:
    """The inference host answers 401 for audio; transcription has its own base URL."""

    class _Transcriptions:
        def __init__(self) -> None:
            self.kwargs: Dict[str, Any] = {}

        def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return type("Transcript", (), {"text": "hello there"})()

    transcriptions = _Transcriptions()
    audio_client = type(
        "AudioClient", (), {"audio": type("Audio", (), {"transcriptions": transcriptions})()}
    )()
    chat_client = _FireworksClient()

    provider = FireworksProvider(client=chat_client, audio_client=audio_client)
    text = provider.transcribe(b"RIFF")

    assert text == "hello there"
    assert transcriptions.kwargs["model"] == provider.transcription_model
    assert provider.audio_base_url != provider.base_url


def test_fireworks_cannot_speak() -> None:
    provider = FireworksProvider(client=object())

    assert provider.supports_transcription
    assert not provider.supports_speech
    with pytest.raises(UnsupportedCapability):
        provider.synthesize_speech("hello")


def test_fireworks_default_tiers_are_all_model_paths() -> None:
    """The cheap tier serves the frame verifier, so every tier must accept images."""
    provider = FireworksProvider(client=object())

    for model in (provider.planner_model, provider.mid_model, provider.cheap_model, provider.responder_model):
        assert model.startswith("accounts/fireworks/models/"), model


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_gates_build_without_credentials() -> None:
    """A funnel must be constructible offline; a provider is only needed per request."""
    gate = build_gate(GateSpec(id="verify", type="model_vision_query", params={"query": "is it open?"}))

    assert gate.gate_id == "verify"


def test_injected_provider_is_returned_unchanged() -> None:
    class Holder(ModelBacked):
        pass

    holder = Holder()
    sentinel = OpenAIProvider(client=object())
    holder.provider = sentinel

    assert holder.provider is sentinel


def test_video_query_funnel_uses_the_configured_cheap_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEERKAT_CHEAP_MODEL", "some-other-model")

    spec = FunnelSpec.video_query("is the door open?")

    verifier = next(gate for gate in spec.gates if gate.type == "model_vision_query")
    assert verifier.params["model"] == "some-other-model"


# --------------------------------------------------------------------------
# degradation when a provider lacks a capability
# --------------------------------------------------------------------------


def test_hosted_transcription_falls_back_to_local_without_provider_stt() -> None:
    from meerkat.funnel.compiler import _swap_unsupported_hosted_transcription
    from meerkat.models.provider import set_active_provider

    gates = [
        GateSpec(id="stt", type="hosted_transcription", params={"model": "whisper"}),
        GateSpec(id="kw", type="transcript_keyword", params={"keywords": ["orange"], "upstream_gate_id": "stt"}),
    ]

    set_active_provider(AnthropicProvider(client=object()))
    assert [gate.type for gate in _swap_unsupported_hosted_transcription(gates)] == [
        "local_realtime_transcription",
        "transcript_keyword",
    ]

    set_active_provider(FireworksProvider(client=object()))
    assert [gate.type for gate in _swap_unsupported_hosted_transcription(gates)] == [
        "hosted_transcription",
        "transcript_keyword",
    ]


async def test_speech_sink_disables_itself_when_the_provider_cannot_speak(tmp_path) -> None:
    from meerkat.runtime.speech import ModelSpeechSink

    sink = ModelSpeechSink(output_dir=str(tmp_path))
    sink.provider = AnthropicProvider(client=object())

    assert await sink.speak("a person appeared") is None
    assert list(tmp_path.iterdir()) == []


async def test_speech_sink_writes_audio_when_the_provider_can_speak(tmp_path) -> None:
    from conftest import StubProvider
    from meerkat.runtime.speech import ModelSpeechSink

    sink = ModelSpeechSink(output_dir=str(tmp_path))
    sink.provider = StubProvider()

    path = await sink.speak("a person appeared")

    assert path is not None and path.read_bytes() == b"audio"


def test_every_provider_implements_the_whole_text_interface() -> None:
    """A new provider cannot silently omit a method the gates depend on."""
    for provider_class in (OpenAIProvider, AnthropicProvider, FireworksProvider):
        for method in ("complete_text", "stream_text", "complete_json"):
            assert getattr(provider_class, method) is not getattr(Provider, method, None), (
                f"{provider_class.__name__} does not implement {method}"
            )
