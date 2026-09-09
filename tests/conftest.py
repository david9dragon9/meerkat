"""Test-wide defaults.

The suite must never depend on which API keys happen to be in the developer's
environment, and must never make a real request. Every test therefore runs
against a pinned OpenAI provider holding a client that raises if touched, so an
accidental network call fails loudly instead of silently succeeding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec
from meerkat.ingest.sources import StreamSource, SyntheticStreamSource
from meerkat.models.provider import (
    ModelRequest,
    OpenAIProvider,
    PROVIDERS,
    Provider,
    set_active_provider,
)
from meerkat.runtime.session import MonitorTarget


class UnusableClient:
    """Stands in for a vendor SDK client; any attribute access is a test bug."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"A test reached the model client (.{name}). Inject a stub provider instead."
        )


class StubProvider(Provider):
    """Records the requests a gate or client makes, and returns canned answers.

    Tests assert against `ModelRequest` rather than any vendor's wire format;
    the translation to each vendor is covered in `test_provider.py`.
    """

    name = "stub"
    planner_model = "stub-planner"
    mid_model = "stub-mid"
    cheap_model = "stub-cheap"
    responder_model = "stub-responder"

    def __init__(
        self,
        text: str = "YES",
        deltas: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        transcript: str = "",
    ) -> None:
        self.text = text
        self.deltas = deltas if deltas is not None else [text]
        self.payload = payload or {}
        self.transcript = transcript
        self.requests: List[ModelRequest] = []

    @property
    def request(self) -> ModelRequest:
        assert self.requests, "no model request was made"
        return self.requests[-1]

    def complete_text(self, request: ModelRequest) -> str:
        self.requests.append(request)
        return self.text

    def stream_text(self, request: ModelRequest):
        self.requests.append(request)
        return iter(self.deltas)

    def complete_json(self, request: ModelRequest) -> Dict[str, Any]:
        self.requests.append(request)
        return dict(self.payload)

    def transcribe(self, wav_bytes: bytes, model: Optional[str] = None) -> str:
        return self.transcript

    def synthesize_speech(self, text: str, model: Optional[str] = None, voice: str = "alloy") -> bytes:
        return b"audio"


@pytest.fixture(autouse=True)
def pinned_provider(monkeypatch: pytest.MonkeyPatch):
    for _, env_var in PROVIDERS.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("MEERKAT_PROVIDER", raising=False)
    set_active_provider(OpenAIProvider(client=UnusableClient()))
    yield
    set_active_provider(None)


class SyntheticTarget(MonitorTarget):
    """A deterministic stand-in for real media, so a run needs no model calls."""

    @property
    def audio_only(self) -> bool:
        return False

    @property
    def has_audio(self) -> bool:
        return False

    def open(self, options, playback_event) -> StreamSource:
        return SyntheticStreamSource(
            fps=4,
            duration_seconds=1.0,
            object_schedule={1: ["person"], 2: ["person"]},
            realtime=False,
        )


#: A funnel with no model-backed gate, so a run exercises the flow end to end
#: without a provider.
FIXED_SPEC = FunnelSpec(
    goal="let me know when a person appears",
    gates=[GateSpec(id="person_detector", type="object_label", params={"classes": ["person"]})],
    response=ResponseSpec(on_match_text="A person appeared.", cooldown_seconds=0.0),
)


@pytest.fixture
def fixed_funnel(monkeypatch: pytest.MonkeyPatch) -> FunnelSpec:
    """Pin the compiled funnel so a run exercises the flow, not the planner."""
    monkeypatch.setattr("meerkat.runtime.session.build_media_spec", lambda *a, **k: FIXED_SPEC)
    return FIXED_SPEC
