"""Provider-neutral model access.

Every model call in Meerkat is one of five shapes:

* text out, from a system prompt plus text and/or images
* the same, streamed, so a verifier can stop at the first decisive token
* JSON out, matching a supplied schema
* audio in, text out
* text in, speech out

`Provider` is exactly those five, so gates never name a vendor. Which provider
is used is decided once, from whichever API key is present, in this order:
OpenAI, then Anthropic, then Fireworks. `MEERKAT_PROVIDER` overrides the choice.

Providers differ in what they can do. Anthropic offers no transcription or
speech synthesis, and Fireworks offers no speech synthesis; those calls raise
`UnsupportedCapability`, and callers degrade rather than crash (audio falls back
to local transcription, and spoken output is simply disabled).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Union


class UnsupportedCapability(RuntimeError):
    """Raised when the active provider cannot serve a request at all."""


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """A frame, carried as a ``data:image/jpeg;base64,...`` URL."""

    data_url: str

    @property
    def media_type(self) -> str:
        if self.data_url.startswith("data:") and ";" in self.data_url:
            return self.data_url[len("data:") : self.data_url.index(";")]
        return "image/jpeg"

    @property
    def base64_data(self) -> str:
        _, _, encoded = self.data_url.partition("base64,")
        return encoded or self.data_url


ContentPart = Union[TextPart, ImagePart]


@dataclass(frozen=True)
class ModelRequest:
    """One model call, described without reference to any vendor's API."""

    model: str
    system: str
    content: Sequence[ContentPart] = field(default_factory=tuple)
    max_output_tokens: Optional[int] = None
    #: Relative thinking budget: "none" for latency-critical checks, "low" for
    #: ordinary calls, "medium" for planning. Providers map or ignore it.
    effort: str = "low"
    #: ``{"name": ..., "schema": {...}}`` when JSON output is required.
    schema: Optional[Dict[str, Any]] = None
    #: OpenAI request tier. Ignored by providers that have no equivalent.
    service_tier: str = "default"

    def text(self) -> str:
        """The request's text parts, joined. Used by text-only providers/paths."""
        return "\n".join(part.text for part in self.content if isinstance(part, TextPart))

    def images(self) -> List[ImagePart]:
        return [part for part in self.content if isinstance(part, ImagePart)]


class Provider(ABC):
    """What Meerkat needs from a model vendor."""

    name: str = "provider"
    #: Model tier defaults, overridable with the MEERKAT_*_MODEL variables.
    planner_model: str = ""
    mid_model: str = ""
    cheap_model: str = ""
    responder_model: str = ""
    transcription_model: str = ""
    speech_model: str = ""

    @abstractmethod
    def complete_text(self, request: ModelRequest) -> str:
        """Return the model's full text response."""

    @abstractmethod
    def stream_text(self, request: ModelRequest) -> Iterator[str]:
        """Yield text deltas as they arrive."""

    @abstractmethod
    def complete_json(self, request: ModelRequest) -> Dict[str, Any]:
        """Return a JSON object matching ``request.schema``."""

    def transcribe(self, wav_bytes: bytes, model: Optional[str] = None) -> str:
        raise UnsupportedCapability(f"{self.name} does not provide speech-to-text.")

    def synthesize_speech(self, text: str, model: Optional[str] = None, voice: str = "alloy") -> bytes:
        raise UnsupportedCapability(f"{self.name} does not provide text-to-speech.")

    def stream_speech(
        self, text: str, model: Optional[str] = None, voice: str = "alloy"
    ) -> Iterator[bytes]:
        """Yield audio chunks. Defaults to one chunk from `synthesize_speech`."""
        yield self.synthesize_speech(text, model=model, voice=voice)

    @property
    def supports_transcription(self) -> bool:
        return type(self).transcribe is not Provider.transcribe

    @property
    def supports_speech(self) -> bool:
        return type(self).synthesize_speech is not Provider.synthesize_speech


def _openai_sdk() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - declared in pyproject
        raise RuntimeError("The 'openai' package is required.") from exc
    return OpenAI


class OpenAIProvider(Provider):
    """OpenAI, through the Responses API."""

    name = "openai"
    planner_model = "gpt-5.6-sol"
    mid_model = "gpt-5.6-terra"
    cheap_model = "gpt-5.6-luna"
    responder_model = "gpt-5.6-sol"
    transcription_model = "gpt-4o-mini-transcribe"
    speech_model = "tts-1"

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client or _openai_sdk()()

    def _input(self, request: ModelRequest) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for part in request.content:
            if isinstance(part, TextPart):
                content.append({"type": "input_text", "text": part.text})
            else:
                content.append({"type": "input_image", "image_url": part.data_url})
        return [
            {"role": "system", "content": request.system},
            {"role": "user", "content": content},
        ]

    def _kwargs(self, request: ModelRequest) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "reasoning": {"effort": request.effort},
            "input": self._input(request),
        }
        if request.service_tier:
            kwargs["service_tier"] = request.service_tier
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.schema is not None:
            kwargs["text"] = {"format": request.schema}
        return kwargs

    def complete_text(self, request: ModelRequest) -> str:
        response = self._client.responses.create(**self._kwargs(request))
        return str(response.output_text)

    def complete_json(self, request: ModelRequest) -> Dict[str, Any]:
        response = self._client.responses.create(**self._kwargs(request))
        return json.loads(response.output_text)

    def stream_text(self, request: ModelRequest) -> Iterator[str]:
        stream = self._client.responses.create(stream=True, **self._kwargs(request))
        try:
            for event in stream:
                if _event_value(event, "type") == "response.output_text.delta":
                    delta = _event_value(event, "delta")
                    if delta is not None:
                        yield str(delta)
        finally:
            _close(stream)

    def transcribe(self, wav_bytes: bytes, model: Optional[str] = None) -> str:
        import io

        response = self._client.audio.transcriptions.create(
            model=model or self.transcription_model,
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
        )
        return str(getattr(response, "text", ""))

    def synthesize_speech(self, text: str, model: Optional[str] = None, voice: str = "alloy") -> bytes:
        chunks = list(self.stream_speech(text, model=model, voice=voice))
        return b"".join(chunks)

    def stream_speech(
        self, text: str, model: Optional[str] = None, voice: str = "alloy"
    ) -> Iterator[bytes]:
        with self._client.audio.speech.with_streaming_response.create(
            model=model or self.speech_model,
            voice=voice,
            input=text,
            response_format="mp3",
        ) as response:
            for chunk in response.iter_bytes(4096):
                if chunk:
                    yield chunk


class AnthropicProvider(Provider):
    """Anthropic, through the Messages API.

    Structured output uses a single forced tool call, which is how the Messages
    API guarantees a schema-shaped result. Extended thinking is left off: these
    are latency-critical realtime checks, so `effort` is deliberately ignored.
    """

    name = "anthropic"
    planner_model = "claude-opus-5"
    mid_model = "claude-sonnet-5"
    cheap_model = "claude-haiku-4-5-20251001"
    responder_model = "claude-sonnet-5"

    #: The Messages API requires max_tokens, so every request needs a ceiling.
    default_max_tokens = 1024
    schema_max_tokens = 8192

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client or _anthropic_sdk()()

    def _messages(self, request: ModelRequest) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for part in request.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.base64_data,
                        },
                    }
                )
        return [{"role": "user", "content": content or [{"type": "text", "text": ""}]}]

    def _kwargs(self, request: ModelRequest) -> Dict[str, Any]:
        max_tokens = request.max_output_tokens
        if max_tokens is None:
            max_tokens = self.schema_max_tokens if request.schema else self.default_max_tokens
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "system": request.system,
            "messages": self._messages(request),
        }
        if request.schema is not None:
            name = str(request.schema.get("name", "result"))
            kwargs["tools"] = [
                {
                    "name": name,
                    "description": "Return the result in this exact shape.",
                    "input_schema": request.schema["schema"],
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": name}
        return kwargs

    def complete_text(self, request: ModelRequest) -> str:
        response = self._client.messages.create(**self._kwargs(request))
        return "".join(
            str(getattr(block, "text", "")) for block in response.content if _block_type(block) == "text"
        )

    def complete_json(self, request: ModelRequest) -> Dict[str, Any]:
        response = self._client.messages.create(**self._kwargs(request))
        for block in response.content:
            if _block_type(block) == "tool_use":
                return dict(getattr(block, "input", {}) or {})
        raise RuntimeError("Anthropic returned no structured result for a schema request.")

    def stream_text(self, request: ModelRequest) -> Iterator[str]:
        # Forced tool use cannot be consumed incrementally as text, so a schema
        # request falls back to a single call.
        if request.schema is not None:
            yield json.dumps(self.complete_json(request))
            return
        with self._client.messages.stream(**self._kwargs(request)) as stream:
            for text in stream.text_stream:
                if text:
                    yield str(text)


def _anthropic_sdk() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Using the Anthropic provider requires the 'anthropic' package. Run `uv sync`."
        ) from exc
    return Anthropic


class FireworksProvider(Provider):
    """Fireworks AI, through its OpenAI-compatible chat completions endpoint.

    Fireworks' serverless catalogue moves; verify ids against
    ``GET /inference/v1/models`` and override with the MEERKAT_*_MODEL
    variables if a default has gone stale. The tiers below are all
    vision-capable, because the frame verifier and the initial-frame summary
    both run on the cheap tier.

    Most serverless models here reason before answering, and that thinking is
    billed against ``max_tokens`` while producing no ``content`` -- a cap sized
    only for the visible answer comes back empty. Fireworks happens to use the
    same effort vocabulary this codebase does, so ``effort`` is passed straight
    through and ``"none"`` genuinely turns thinking off on the default tiers.
    """

    name = "fireworks"
    base_url = "https://api.fireworks.ai/inference/v1"
    #: Transcription is not served from the inference host; the general endpoint
    #: answers 401 for audio regardless of the key.
    audio_base_url = "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
    planner_model = "accounts/fireworks/models/kimi-k3"
    mid_model = "accounts/fireworks/models/kimi-k3"
    cheap_model = "accounts/fireworks/models/kimi-k2p6"
    responder_model = "accounts/fireworks/models/kimi-k2p6"
    transcription_model = "whisper-v3-turbo"

    #: Room for the scratchpad, added only when thinking is actually on.
    reasoning_headroom_tokens = 768

    def __init__(self, client: Optional[Any] = None, audio_client: Optional[Any] = None) -> None:
        self._client = client or _openai_sdk()(
            api_key=getenv("FIREWORKS_API_KEY"),
            base_url=self.base_url,
        )
        self._audio_client = audio_client
        #: Models that rejected effort="none"; they cannot stop thinking, so the
        #: best available option is to minimise it.
        self._thinking_only: Set[str] = set()

    def _audio(self) -> Any:
        if self._audio_client is None:
            self._audio_client = _openai_sdk()(
                api_key=getenv("FIREWORKS_API_KEY"),
                base_url=self.audio_base_url,
            )
        return self._audio_client

    def _messages(self, request: ModelRequest) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for part in request.content:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                content.append({"type": "image_url", "image_url": {"url": part.data_url}})
        return [
            {"role": "system", "content": request.system},
            {"role": "user", "content": content or [{"type": "text", "text": ""}]},
        ]

    def _kwargs(self, request: ModelRequest, effort: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": self._messages(request),
            "reasoning_effort": effort,
        }
        if request.max_output_tokens is not None:
            budget = request.max_output_tokens
            if effort != "none":
                budget += self.reasoning_headroom_tokens
            kwargs["max_tokens"] = budget
        if request.schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": str(request.schema.get("name", "result")),
                    "schema": request.schema["schema"],
                },
            }
        return kwargs

    def _create(self, request: ModelRequest, **overrides: Any) -> Any:
        """Issue the call, retrying once if the model refuses to stop thinking."""
        effort = "low" if request.model in self._thinking_only and request.effort == "none" else request.effort
        try:
            return self._client.chat.completions.create(**self._kwargs(request, effort), **overrides)
        except Exception as exc:
            if effort != "none" or not _rejects_disabled_thinking(exc):
                raise
            self._thinking_only.add(request.model)
            return self._client.chat.completions.create(**self._kwargs(request, "low"), **overrides)

    def complete_text(self, request: ModelRequest) -> str:
        response = self._create(request)
        return str(response.choices[0].message.content or "")

    def complete_json(self, request: ModelRequest) -> Dict[str, Any]:
        return json.loads(self.complete_text(request))

    def stream_text(self, request: ModelRequest) -> Iterator[str]:
        stream = self._create(request, stream=True)
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta else None
                if text:
                    yield str(text)
        finally:
            _close(stream)

    def transcribe(self, wav_bytes: bytes, model: Optional[str] = None) -> str:
        import io

        response = self._audio().audio.transcriptions.create(
            model=model or self.transcription_model,
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
        )
        return str(getattr(response, "text", ""))


def _rejects_disabled_thinking(exc: Exception) -> bool:
    """Whether an error says this model cannot have its thinking turned off."""
    message = str(exc).lower()
    return "thinking-only" in message or (
        "reasoning_effort" in message and "not supported" in message
    )


#: Checked in order; the first provider whose key is present wins.
PROVIDERS: Dict[str, Any] = {
    "openai": (OpenAIProvider, "OPENAI_API_KEY"),
    "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY"),
    "fireworks": (FireworksProvider, "FIREWORKS_API_KEY"),
}


def available_provider_names() -> List[str]:
    """Provider names with a usable key, in preference order."""
    return [name for name, (_, env_var) in PROVIDERS.items() if getenv(env_var)]


def resolve_provider_name() -> str:
    """Pick a provider from the environment, honouring an explicit override."""
    override = (getenv("MEERKAT_PROVIDER") or "").strip().lower()
    if override:
        if override not in PROVIDERS:
            raise RuntimeError(
                f"Unknown MEERKAT_PROVIDER={override!r}. Choose one of: {', '.join(PROVIDERS)}."
            )
        return override
    available = available_provider_names()
    if not available:
        raise RuntimeError(
            "No model provider is configured. Set one of "
            + ", ".join(env_var for _, env_var in PROVIDERS.values())
            + " (checked in that order), or set MEERKAT_PROVIDER."
        )
    return available[0]


def create_provider(name: Optional[str] = None) -> Provider:
    """Build a provider by name, or by whichever key is available."""
    provider_class, _ = PROVIDERS[name or resolve_provider_name()]
    return provider_class()


_active: Optional[Provider] = None


def active_provider() -> Provider:
    """The process-wide provider, created on first use."""
    global _active
    if _active is None:
        _active = create_provider()
    return _active


def set_active_provider(provider: Optional[Provider]) -> None:
    """Override the process-wide provider. Pass None to re-resolve on next use."""
    global _active
    _active = provider


class ModelBacked:
    """Mixin giving a gate or client `self.provider`, resolved on first use.

    Resolving lazily keeps funnels constructible with no credentials at all, and
    assigning to `self.provider` lets callers and tests inject a stub.
    """

    _provider_instance: Optional[Provider] = None

    @property
    def provider(self) -> Provider:
        if self._provider_instance is None:
            self._provider_instance = active_provider()
        return self._provider_instance

    @provider.setter
    def provider(self, provider: Optional[Provider]) -> None:
        self._provider_instance = provider


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _block_type(block: Any) -> str:
    return str(_event_value(block, "type") or "")


def _close(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()
