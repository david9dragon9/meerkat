"""Compile a plain-English request into a runnable funnel.

The planner returns a rough funnel; the normalization passes here turn it into
something that can actually keep up with a live stream: cheap gates ahead of
model-backed ones, exactly one model verifier on the path to any alert, audio
gates dropped for silent media, and verifier queue settings applied uniformly.

Every front end (CLI, web UI, benchmark) compiles through this module, so a
prompt produces the same funnel no matter where it was typed.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, replace
from typing import Optional

from meerkat.funnel.gate_types import (
    MODEL_CONFIRMATION_GATE_TYPES,
    TRANSCRIPT_CONSUMER_GATE_TYPES,
    TRANSCRIPT_PRODUCER_GATE_TYPES,
    VALUE_EXTRACTION_MODES,
    VISUAL_GATE_TYPES,
)
from meerkat.funnel.planner import create_model_planner
from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec, StateUpdateSpec
from meerkat.ingest.sources import load_initial_video_frame
from meerkat.models.config import ModelConfig
from meerkat.models.provider import (
    ImagePart,
    ModelRequest,
    TextPart,
    active_provider,
)
from meerkat.models.responders import _notification_from_goal
from meerkat.runtime.logging import RuntimeLogger


_INITIAL_BASELINE_CHANGE_TERMS = (
    "separate",
    "separating",
    "apart",
    "come together",
    "coming together",
    "together",
    "open",
    "close",
    "change",
    "from ",
    "become",
    "turn into",
)

_AUDIO_TEXT_GATE_TYPES = TRANSCRIPT_CONSUMER_GATE_TYPES | TRANSCRIPT_PRODUCER_GATE_TYPES
_AUDIO_KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "blue",  # keep colors through explicit prompt handling below
    "brings",
    "bring",
    "brought",
    "can",
    "know",
    "let",
    "me",
    "soon",
    "someone",
    "tell",
    "the",
    "when",
    "who",
    "you",
}


def _video_runtime_spec_from_planned_spec(
    prompt: str,
    planned_spec: FunnelSpec,
    sample_interval_ms: int,
    verifier_model: Optional[str] = None,
    verifier_sample_interval_ms: Optional[int] = None,
    verifier_concurrent_requests: Optional[int] = None,
    verifier_min_request_interval_ms: Optional[int] = None,
    verifier_queued_frame_dedupe_window_ms: Optional[int] = None,
    media_has_audio: bool = True,
) -> FunnelSpec:
    planned_gates = planned_spec.gates if media_has_audio else _drop_audio_dependent_gates(planned_spec.gates)
    downstream_gate_ids = {
        str(gate.params["upstream_gate_id"])
        for gate in planned_gates
        if gate.params.get("upstream_gate_id")
    }
    planned_gates_by_id = {gate.id: gate for gate in planned_gates}
    gates = [
        _apply_video_verifier_overrides(
            gate,
            sample_interval_ms=sample_interval_ms,
            verifier_model=verifier_model,
            verifier_sample_interval_ms=verifier_sample_interval_ms,
            verifier_concurrent_requests=verifier_concurrent_requests,
            verifier_min_request_interval_ms=verifier_min_request_interval_ms,
            verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
            has_downstream=gate.id in downstream_gate_ids,
            prompt=prompt,
            planned_gates_by_id=planned_gates_by_id,
        )
        for gate in planned_gates
    ]
    gates = _collapse_redundant_model_stages(gates)
    gates = _drop_unconfirmed_terminal_leaves_for_answer_extraction(prompt, gates)
    gates = _swap_unsupported_hosted_transcription(gates)
    gates = _ensure_local_transcription_for_transcript_consumers(gates)
    gates = _normalize_local_realtime_transcription_gates(gates)
    gates = _ensure_audio_keyword_trigger_for_transcript_model(prompt, gates)
    gates, deferred_state_updates = _defer_nonterminal_state_updates(gates)
    gates = _ensure_model_confirmation_for_terminal_leaves(
        prompt=prompt,
        gates=gates,
        verifier_model=verifier_model,
        verifier_sample_interval_ms=verifier_sample_interval_ms,
        verifier_concurrent_requests=verifier_concurrent_requests,
        verifier_min_request_interval_ms=verifier_min_request_interval_ms,
        verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
        sample_interval_ms=sample_interval_ms,
    )
    gates = _ensure_generic_visual_confirmation_for_video_prompt(
        prompt=prompt,
        gates=gates,
        verifier_model=verifier_model,
        verifier_sample_interval_ms=verifier_sample_interval_ms,
        verifier_concurrent_requests=verifier_concurrent_requests,
        verifier_min_request_interval_ms=verifier_min_request_interval_ms,
        verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
        sample_interval_ms=sample_interval_ms,
    )
    gates, deferred_after_confirmation = _defer_nonterminal_state_updates(gates)
    deferred_state_updates.extend(deferred_after_confirmation)
    response = planned_spec.response
    if deferred_state_updates:
        response = _merge_deferred_state_updates(response, deferred_state_updates)
    response = _ensure_state_tracking_response(prompt, response)
    if not response.on_match_text and response.on_no_match == "ignore":
        response = replace(response, on_match_text=_notification_from_goal(prompt), model="local")
    response = _ensure_extracted_value_reaches_the_user(gates, response)
    if _is_one_shot_notification_prompt(prompt):
        response = replace(response, max_responses=1)
    return FunnelSpec(
        goal=planned_spec.goal or prompt,
        gates=gates,
        response=response,
        version=planned_spec.version,
    )


def _audio_runtime_spec_from_planned_spec(prompt: str, planned_spec: FunnelSpec) -> FunnelSpec:
    gates = [
        gate
        for gate in planned_spec.gates
        if gate.type not in VISUAL_GATE_TYPES
        and not (
            gate.params.get("upstream_gate_id")
            and _upstream_depends_on_visual_gate(str(gate.params["upstream_gate_id"]), planned_spec.gates)
        )
    ]
    if not any(gate.type in TRANSCRIPT_CONSUMER_GATE_TYPES for gate in gates):
        gates.append(
            GateSpec(
                id="answer_from_transcript",
                type="model_transcript_query",
                params={
                    "query": prompt,
                    "model": ModelConfig.from_env().cheap_model,
                    "verification_mode": "extract" if _prompt_requests_extracted_answer(prompt) else "binary",
                    "service_tier": "default",
                },
            )
        )
    gates = _swap_unsupported_hosted_transcription(gates)
    gates = _ensure_local_transcription_for_transcript_consumers(gates)
    gates = _normalize_local_realtime_transcription_gates(gates)
    gates = _ensure_audio_keyword_trigger_for_transcript_model(prompt, gates)
    gates = _collapse_redundant_model_stages(gates)
    gates, deferred_state_updates = _defer_nonterminal_state_updates(gates)
    response = planned_spec.response
    if deferred_state_updates:
        response = _merge_deferred_state_updates(response, deferred_state_updates)
    if not response.on_match_text or response.on_match_text.strip().lower() == prompt.strip().lower():
        response = replace(response, model="local", on_match_text=_audio_answer_template(prompt), on_no_match="ignore")
    response = _ensure_extracted_value_reaches_the_user(gates, response)
    if _is_one_shot_notification_prompt(prompt):
        response = replace(response, max_responses=1)
    return FunnelSpec(
        goal=planned_spec.goal or prompt,
        gates=gates,
        response=response,
        version=planned_spec.version,
    )


def _drop_audio_dependent_gates(gates: list[GateSpec]) -> list[GateSpec]:
    output: list[GateSpec] = []
    dropped_ids: set[str] = set()
    for gate in gates:
        upstream_id = gate.params.get("upstream_gate_id")
        if gate.type in _AUDIO_TEXT_GATE_TYPES or (upstream_id and str(upstream_id) in dropped_ids):
            dropped_ids.add(gate.id)
            continue
        output.append(gate)
    return output


def _upstream_depends_on_visual_gate(upstream_id: str, gates: list[GateSpec]) -> bool:
    by_id = {gate.id: gate for gate in gates}
    current = by_id.get(upstream_id)
    seen: set[str] = set()
    while current is not None and current.id not in seen:
        seen.add(current.id)
        if current.type in VISUAL_GATE_TYPES:
            return True
        next_id = current.params.get("upstream_gate_id")
        current = by_id.get(str(next_id)) if next_id else None
    return False


def _audio_query_fallback_spec(prompt: str) -> FunnelSpec:
    return _audio_runtime_spec_from_planned_spec(
        prompt,
        FunnelSpec(
            goal=prompt,
            gates=[],
            response=ResponseSpec(model="local", cooldown_seconds=2.0, on_match_text=_audio_answer_template(prompt)),
        ),
    )


def _audio_answer_template(prompt: str) -> str:
    text = prompt.lower()
    if "who" in text:
        return "{evidence.text}"
    if _prompt_requests_extracted_answer(prompt):
        return "{evidence.text}"
    return _notification_from_goal(prompt)


def _ensure_audio_keyword_trigger_for_transcript_model(prompt: str, gates: list[GateSpec]) -> list[GateSpec]:
    keywords = _audio_trigger_keywords_from_prompt(prompt)
    if not keywords:
        return gates
    producer_ids = {gate.id for gate in gates if gate.type in TRANSCRIPT_PRODUCER_GATE_TYPES}
    if not producer_ids:
        return gates
    existing_ids = {gate.id for gate in gates}
    output: list[GateSpec] = []
    inserted_by_producer: dict[str, str] = {}
    for gate in gates:
        if gate.type == "model_transcript_query" and gate.params.get("upstream_gate_id") in producer_ids:
            producer_id = str(gate.params["upstream_gate_id"])
            trigger_id = inserted_by_producer.get(producer_id)
            if trigger_id is None:
                trigger_id = _unique_gate_id("cheap_transcript_keyword_trigger", existing_ids)
                existing_ids.add(trigger_id)
                inserted_by_producer[producer_id] = trigger_id
                output.append(
                    GateSpec(
                        id=trigger_id,
                        type="transcript_keyword",
                        params={"keywords": keywords, "upstream_gate_id": producer_id},
                    )
                )
            params = dict(gate.params)
            params["upstream_gate_id"] = trigger_id
            output.append(GateSpec(id=gate.id, type=gate.type, params=params))
        else:
            output.append(gate)
    return output


def _audio_trigger_keywords_from_prompt(prompt: str) -> list[str]:
    text = prompt.lower()
    quoted = [left or right for left, right in re.findall(r'"([^"]+)"|' + r"'([^']+)'", prompt)]
    words = re.findall(r"[a-z][a-z0-9'-]*", text)
    keywords = [
        word
        for word in words
        if len(word) >= 4 and word not in _AUDIO_KEYWORD_STOPWORDS
    ]
    for color in ("blue", "red", "green", "yellow", "black", "white", "purple", "orange"):
        if re.search(rf"\b{color}\b", text):
            keywords.append(color)
    keywords.extend(keyword.lower() for keyword in quoted)
    return sorted(set(keywords))


def _defer_nonterminal_state_updates(gates: list[GateSpec]) -> tuple[list[GateSpec], list[StateUpdateSpec]]:
    downstream_gate_ids = {
        str(gate.params["upstream_gate_id"])
        for gate in gates
        if gate.params.get("upstream_gate_id")
    }
    deferred: list[StateUpdateSpec] = []
    output: list[GateSpec] = []
    for gate in gates:
        raw_updates = gate.params.get("state_updates")
        if gate.id not in downstream_gate_ids or not isinstance(raw_updates, list):
            output.append(gate)
            continue
        params = dict(gate.params)
        params.pop("state_updates", None)
        output.append(GateSpec(id=gate.id, type=gate.type, params=params))
        for update in raw_updates:
            parsed = _parse_state_update(update)
            if parsed is None:
                continue
            if gate.type == "ocr_text" and parsed.operation == "increment":
                parsed = StateUpdateSpec(
                    key=parsed.key,
                    operation="increment_unique",
                    value="{evidence.matches}",
                )
            deferred.append(parsed)
    return output, deferred


def _parse_state_update(raw_update: object) -> Optional[StateUpdateSpec]:
    if isinstance(raw_update, StateUpdateSpec):
        return raw_update
    if not isinstance(raw_update, dict) or not raw_update.get("key"):
        return None
    return StateUpdateSpec(
        key=str(raw_update["key"]),
        operation=str(raw_update.get("operation", "increment")),
        value=raw_update.get("value", 1),
    )


def _merge_deferred_state_updates(response, deferred_state_updates: list[StateUpdateSpec]):
    referenced_keys = set(re.findall(r"\{state\.([A-Za-z_][A-Za-z0-9_]*)\}", response.on_match_text or ""))
    deferred_keys = {update.key for update in deferred_state_updates}
    existing_updates = [
        update
        for update in response.state_updates
        if update.key in referenced_keys
        or (update.key not in deferred_keys and not (update.key == "count" and update.operation == "increment"))
    ]
    for update in deferred_state_updates:
        if not any(existing.key == update.key for existing in existing_updates):
            existing_updates.append(update)
    return replace(response, model="local", state_updates=existing_updates)


def _ensure_state_tracking_response(prompt: str, response):
    text = prompt.lower()
    if response.state_updates or not ("count" in text and ("increment" in text or "every time" in text)):
        return response
    key = _count_state_key_from_prompt(prompt)
    on_match_text = response.on_match_text or f"Count: {{state.{key}}}"
    return replace(
        response,
        model="local",
        on_match_text=on_match_text,
        state_updates=[StateUpdateSpec(key=key, operation="increment", value=1)],
    )


def _count_state_key_from_prompt(prompt: str) -> str:
    text = prompt.lower()
    match = re.search(r"\bevery\s+time\s+(?:a|an|the)?\s*([a-z][a-z0-9_-]*)", text)
    if not match:
        match = re.search(r"\bcount\s+(?:a|an|the)?\s*([a-z][a-z0-9_-]*)", text)
    if match:
        key = re.sub(r"[^a-z0-9_]+", "_", match.group(1).strip())
        if key and key not in {"count", "time", "every"}:
            return f"{key}_count"
    return "count"


def _collapse_redundant_model_stages(gates: list[GateSpec]) -> list[GateSpec]:
    output: list[GateSpec] = []
    replacements: dict[str, str] = {}

    for gate in gates:
        params = dict(gate.params)
        upstream_id = params.get("upstream_gate_id")
        if upstream_id:
            params["upstream_gate_id"] = _resolve_replacement(str(upstream_id), replacements)

        if gate.type in MODEL_CONFIRMATION_GATE_TYPES and params.get("upstream_gate_id"):
            by_id = {existing.id: existing for existing in output}
            if _gate_has_model_confirmation_path(str(params["upstream_gate_id"]), by_id):
                replacements[gate.id] = str(params["upstream_gate_id"])
                continue

        output.append(GateSpec(id=gate.id, type=gate.type, params=params))

    if not replacements:
        return output

    rewritten: list[GateSpec] = []
    for gate in output:
        params = dict(gate.params)
        upstream_id = params.get("upstream_gate_id")
        if upstream_id:
            params["upstream_gate_id"] = _resolve_replacement(str(upstream_id), replacements)
        rewritten.append(GateSpec(id=gate.id, type=gate.type, params=params))
    return rewritten


def _drop_unconfirmed_terminal_leaves_for_answer_extraction(prompt: str, gates: list[GateSpec]) -> list[GateSpec]:
    if not _prompt_requests_extracted_answer(prompt):
        return gates
    if not any(
        gate.type == "model_vision_query" and gate.params.get("verification_mode") in {"extract", "value", "text"}
        for gate in gates
    ):
        return gates
    downstream_ids = {
        str(gate.params["upstream_gate_id"])
        for gate in gates
        if gate.params.get("upstream_gate_id")
    }
    return [
        gate
        for gate in gates
        if gate.id in downstream_ids
        or gate.type in MODEL_CONFIRMATION_GATE_TYPES
        or gate.params.get("state_updates")
    ]


def _swap_unsupported_hosted_transcription(gates: list[GateSpec]) -> list[GateSpec]:
    """Fall back to local transcription when the provider has no speech-to-text.

    Anthropic offers no transcription endpoint, so a planner that reaches for
    `hosted_transcription` would otherwise produce a funnel that cannot run.
    Local transcription covers the same job, just on this machine.
    """
    if not any(gate.type == "hosted_transcription" for gate in gates):
        return gates
    try:
        supported = active_provider().supports_transcription
    except Exception:
        supported = True
    if supported:
        return gates
    return [
        replace(gate, type="local_realtime_transcription", params={})
        if gate.type == "hosted_transcription"
        else gate
        for gate in gates
    ]


def _ensure_local_transcription_for_transcript_consumers(gates: list[GateSpec]) -> list[GateSpec]:
    producer_ids = [gate.id for gate in gates if gate.type in TRANSCRIPT_PRODUCER_GATE_TYPES]
    consumers_needing_transcript = [
        gate
        for gate in gates
        if gate.type in TRANSCRIPT_CONSUMER_GATE_TYPES and not gate.params.get("upstream_gate_id")
    ]
    if not consumers_needing_transcript:
        return gates
    producer_id = producer_ids[0] if producer_ids else _unique_gate_id("cheap_realtime_transcription", {gate.id for gate in gates})
    output: list[GateSpec] = []
    if not producer_ids:
        output.append(
            GateSpec(
                id=producer_id,
                type="local_realtime_transcription",
                params={
                    "model": "small.en",
                    "model_path": None,
                    "compute_type": "int8",
                    "language": "en",
                    "buffer_ms": 2000,
                    "sample_interval_ms": 1000,
                    "min_volume_db": -45.0,
                    "text_window_ms": 12000,
                },
            )
        )
    for gate in gates:
        if gate.type in TRANSCRIPT_CONSUMER_GATE_TYPES and not gate.params.get("upstream_gate_id"):
            params = dict(gate.params)
            params["upstream_gate_id"] = producer_id
            output.append(GateSpec(id=gate.id, type=gate.type, params=params))
        else:
            output.append(gate)
    return output


def _normalize_local_realtime_transcription_gates(gates: list[GateSpec]) -> list[GateSpec]:
    output: list[GateSpec] = []
    for gate in gates:
        if gate.type != "local_realtime_transcription":
            output.append(gate)
            continue
        params = dict(gate.params)
        params["buffer_ms"] = min(int(params.get("buffer_ms", 2000) or 2000), 2000)
        params["sample_interval_ms"] = max(250, min(int(params.get("sample_interval_ms", 1000) or 1000), 1000))
        params.setdefault("compute_type", "int8")
        params.setdefault("min_volume_db", -45.0)
        output.append(GateSpec(id=gate.id, type=gate.type, params=params))
    return output


def _resolve_replacement(gate_id: str, replacements: dict[str, str]) -> str:
    seen: set[str] = set()
    current = gate_id
    while current in replacements and current not in seen:
        seen.add(current)
        current = replacements[current]
    return current


def _ensure_model_confirmation_for_terminal_leaves(
    prompt: str,
    gates: list[GateSpec],
    verifier_model: Optional[str],
    verifier_sample_interval_ms: Optional[int],
    verifier_concurrent_requests: Optional[int],
    verifier_min_request_interval_ms: Optional[int],
    verifier_queued_frame_dedupe_window_ms: Optional[int],
    sample_interval_ms: int,
) -> list[GateSpec]:
    if not gates:
        return [
            GateSpec(
                id="model_user_alert_confirm",
                type="model_vision_query",
                params=_model_confirmation_params(
                    prompt,
                    upstream_gate_id=None,
                    verifier_model=verifier_model,
                    verifier_sample_interval_ms=verifier_sample_interval_ms,
                    verifier_concurrent_requests=verifier_concurrent_requests,
                    verifier_min_request_interval_ms=verifier_min_request_interval_ms,
                    verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
                    sample_interval_ms=sample_interval_ms,
                ),
            )
        ]

    by_id = {gate.id: gate for gate in gates}
    if any(gate.type in MODEL_CONFIRMATION_GATE_TYPES for gate in gates):
        return gates
    downstream_ids = {
        str(gate.params["upstream_gate_id"])
        for gate in gates
        if gate.params.get("upstream_gate_id")
    }
    leaves = [gate for gate in gates if gate.id not in downstream_ids]
    output = list(gates)
    for leaf in leaves:
        if _gate_has_model_confirmation_path(leaf.id, by_id):
            continue
        confirm_id = _unique_gate_id(f"{leaf.id}_model_confirm", {gate.id for gate in output})
        confirmation_type = "model_transcript_query" if _is_audio_text_leaf(leaf) else "model_vision_query"
        output.append(
            GateSpec(
                id=confirm_id,
                type=confirmation_type,
                params=_model_confirmation_params(
                    prompt,
                    upstream_gate_id=leaf.id,
                    verifier_model=verifier_model,
                    verifier_sample_interval_ms=verifier_sample_interval_ms,
                    verifier_concurrent_requests=verifier_concurrent_requests,
                    verifier_min_request_interval_ms=verifier_min_request_interval_ms,
                    verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
                    sample_interval_ms=sample_interval_ms,
                ),
            )
        )
    return output


def _ensure_generic_visual_confirmation_for_video_prompt(
    prompt: str,
    gates: list[GateSpec],
    verifier_model: Optional[str],
    verifier_sample_interval_ms: Optional[int],
    verifier_concurrent_requests: Optional[int],
    verifier_min_request_interval_ms: Optional[int],
    verifier_queued_frame_dedupe_window_ms: Optional[int],
    sample_interval_ms: int,
) -> list[GateSpec]:
    if _prompt_is_audio_or_speech_focused(prompt):
        return gates
    existing_ids = {gate.id for gate in gates}
    output = list(gates)
    added_generic = any(gate.type in {"model_vision_query", "model_frame_change"} for gate in gates)
    if added_generic:
        return output
    for gate in gates:
        if gate.type != "local_yolo_object":
            continue
        if not _has_downstream_model_confirmation(gate.id, gates):
            continue
        if _has_generic_confirmation_for_upstream(prompt, gate.id, gates):
            continue
        confirm_id = _unique_gate_id(f"{gate.id}_model_confirm", existing_ids)
        existing_ids.add(confirm_id)
        output.append(
            GateSpec(
                id=confirm_id,
                type="model_vision_query",
                params=_model_confirmation_params(
                    prompt,
                    upstream_gate_id=gate.id,
                    verifier_model=verifier_model,
                    verifier_sample_interval_ms=verifier_sample_interval_ms,
                    verifier_concurrent_requests=verifier_concurrent_requests,
                    verifier_min_request_interval_ms=verifier_min_request_interval_ms,
                    verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
                    sample_interval_ms=sample_interval_ms,
                ),
            )
        )
        return output
    change_id = _unique_gate_id("generic_visual_change_confirm", existing_ids)
    output.append(
        GateSpec(
            id=change_id,
            type="model_frame_change",
            params={
                "query": prompt,
                "baseline_mode": "previous",
                "sample_interval_ms": verifier_sample_interval_ms
                if verifier_sample_interval_ms is not None
                else min(sample_interval_ms, 500),
                "min_confidence": 0.8,
                "model": verifier_model or ModelConfig.from_env().cheap_model,
                "concurrent_requests": verifier_concurrent_requests if verifier_concurrent_requests is not None else 3,
                "min_request_interval_ms": max(200, verifier_min_request_interval_ms or 200),
                "queued_frame_dedupe_window_ms": verifier_queued_frame_dedupe_window_ms
                if verifier_queued_frame_dedupe_window_ms is not None
                else 500,
                "service_tier": "default",
                "required_count": 1,
                "confirmation_window_ms": 2500,
                "state_interval_ms": 3000,
            },
        )
    )
    return output


def _has_downstream_model_confirmation(gate_id: str, gates: list[GateSpec]) -> bool:
    children = [gate for gate in gates if gate.params.get("upstream_gate_id") == gate_id]
    for child in children:
        if child.type in MODEL_CONFIRMATION_GATE_TYPES:
            return True
        if _has_downstream_model_confirmation(child.id, gates):
            return True
    return False


def _has_generic_confirmation_for_upstream(prompt: str, upstream_gate_id: str, gates: list[GateSpec]) -> bool:
    normalized_prompt = prompt.strip().lower()
    for gate in gates:
        if gate.type != "model_vision_query":
            continue
        if gate.params.get("upstream_gate_id") != upstream_gate_id:
            continue
        if str(gate.params.get("query", "")).strip().lower() == normalized_prompt:
            return True
    return False


def _is_audio_text_leaf(gate: GateSpec) -> bool:
    return gate.type in _AUDIO_TEXT_GATE_TYPES


def _gate_has_model_confirmation_path(gate_id: str, gates_by_id: dict[str, GateSpec]) -> bool:
    seen: set[str] = set()
    current = gates_by_id.get(gate_id)
    while current is not None and current.id not in seen:
        seen.add(current.id)
        if current.type in MODEL_CONFIRMATION_GATE_TYPES:
            return True
        upstream_id = current.params.get("upstream_gate_id")
        current = gates_by_id.get(str(upstream_id)) if upstream_id else None
    return False


def _model_confirmation_params(
    prompt: str,
    upstream_gate_id: Optional[str],
    verifier_model: Optional[str],
    verifier_sample_interval_ms: Optional[int],
    verifier_concurrent_requests: Optional[int],
    verifier_min_request_interval_ms: Optional[int],
    verifier_queued_frame_dedupe_window_ms: Optional[int],
    sample_interval_ms: int,
) -> dict[str, object]:
    params: dict[str, object] = {
        "query": prompt,
        "sample_interval_ms": verifier_sample_interval_ms
        if verifier_sample_interval_ms is not None
        else sample_interval_ms,
        "min_confidence": 0.8,
        "model": verifier_model or ModelConfig.from_env().cheap_model,
        "verification_mode": "binary",
        "concurrent_requests": verifier_concurrent_requests if verifier_concurrent_requests is not None else 3,
        "min_request_interval_ms": max(200, verifier_min_request_interval_ms or 200),
        "queued_frame_dedupe_window_ms": verifier_queued_frame_dedupe_window_ms
        if verifier_queued_frame_dedupe_window_ms is not None
        else 500,
        "service_tier": "default",
    }
    if upstream_gate_id:
        params["upstream_gate_id"] = upstream_gate_id
    return params


def _unique_gate_id(base_id: str, existing_ids: set[str]) -> str:
    if base_id not in existing_ids:
        return base_id
    index = 2
    while f"{base_id}_{index}" in existing_ids:
        index += 1
    return f"{base_id}_{index}"


def _ensure_extracted_value_reaches_the_user(gates: list[GateSpec], response: ResponseSpec) -> ResponseSpec:
    """Put the extracted value into the message when a gate produces one.

    A funnel that extracts a value but answers with a fixed sentence says the
    same words every time. The deduplication that keeps a monitor from
    repeating itself would then also stop it ever reporting a *new* value, so
    the user would hear the first answer and nothing after it.
    """
    extracts = any(gate.params.get("verification_mode") in VALUE_EXTRACTION_MODES for gate in gates)
    if not extracts:
        return response
    template = (response.on_match_text or "").strip()
    if "{evidence." in template:
        return response
    if not template:
        return replace(response, on_match_text="{evidence.text}", model="local")
    return replace(response, on_match_text=f"{template.rstrip('.')}: {{evidence.text}}", model="local")


def _is_one_shot_notification_prompt(prompt: str) -> bool:
    """Whether the prompt asks about a single occurrence rather than an ongoing watch.

    A request for repeated updates always wins: capping such a funnel at one
    response would answer once and then go quiet, which is the opposite of what
    was asked.
    """
    text = prompt.lower().strip()
    if _prompt_requests_continuous_updates(prompt):
        return False
    if _prompt_requests_extracted_answer(prompt):
        return True
    if "as soon as you find out" in text or "as soon as you know" in text:
        return True
    return text.startswith(
        (
            "let me know when ",
            "tell me when ",
            "notify me when ",
            "alert me when ",
        )
    )


def _prompt_requests_continuous_updates(prompt: str) -> bool:
    text = prompt.lower()
    continuous_terms = (
        "every time",
        "each time",
        "whenever",
        "keep telling",
        "keep updating",
        "keep me",
        "update me",
        "updates",
        "count",
        "increment",
        "track",
        "continue",
        "continuously",
        # A prompt about something changing is asking to hear about each change,
        # not only the first one.
        "changes",
        "changes to",
    )
    return any(term in text for term in continuous_terms)


def _apply_video_verifier_overrides(
    gate: GateSpec,
    sample_interval_ms: int,
    verifier_model: Optional[str],
    verifier_sample_interval_ms: Optional[int],
    verifier_concurrent_requests: Optional[int],
    verifier_min_request_interval_ms: Optional[int],
    verifier_queued_frame_dedupe_window_ms: Optional[int],
    has_downstream: bool = False,
    prompt: Optional[str] = None,
    planned_gates_by_id: Optional[dict[str, GateSpec]] = None,
) -> GateSpec:
    if gate.type not in {"model_vision_query", "model_vision_object", "model_frame_change"}:
        return gate
    params = dict(gate.params)
    if verifier_sample_interval_ms is not None:
        params["sample_interval_ms"] = verifier_sample_interval_ms
    elif "sample_interval_ms" not in params:
        params["sample_interval_ms"] = sample_interval_ms
    if verifier_model is not None:
        params["model"] = verifier_model
    if gate.type == "model_frame_change":
        if prompt:
            params["query"] = prompt
        _normalize_temporal_change_params(params)
        if not has_downstream:
            params.setdefault("required_count", 2)
            params.setdefault("confirmation_window_ms", 2500)
        if verifier_concurrent_requests is not None:
            params["concurrent_requests"] = verifier_concurrent_requests
        if verifier_min_request_interval_ms is not None:
            params["min_request_interval_ms"] = max(200, verifier_min_request_interval_ms)
        if verifier_queued_frame_dedupe_window_ms is not None:
            params["queued_frame_dedupe_window_ms"] = verifier_queued_frame_dedupe_window_ms
        return GateSpec(id=gate.id, type=gate.type, params=params)
    if verifier_concurrent_requests is not None:
        params["concurrent_requests"] = verifier_concurrent_requests
    if verifier_min_request_interval_ms is not None:
        params["min_request_interval_ms"] = max(200, verifier_min_request_interval_ms)
    if verifier_queued_frame_dedupe_window_ms is not None:
        params["queued_frame_dedupe_window_ms"] = verifier_queued_frame_dedupe_window_ms
    if gate.type == "model_vision_query":
        query_text = " ".join([str(params.get("query", "")), prompt or ""])
        needs_event_confirmation = _prompt_needs_visual_window(query_text)
        _broaden_query_upstream(
            params,
            planned_gates_by_id or {},
            preserve_temporal_join=False,
        )
        if verifier_sample_interval_ms is None:
            current_interval = int(params.get("sample_interval_ms", sample_interval_ms))
            params["sample_interval_ms"] = min(current_interval, 500)
        if "window_ms" not in params and needs_event_confirmation:
            params["window_ms"] = 3000
        if int(params.get("window_ms") or 0) > 0:
            params["required_count"] = 2 if needs_event_confirmation else 1
            params["confirmation_window_ms"] = max(int(params.get("confirmation_window_ms", 2500) or 2500), 4500)
        if needs_event_confirmation and _prompt_likely_uses_persistent_state_display(query_text):
            params["query"] = _event_outcome_transition_query(str(params.get("query", prompt or "")))
        if prompt and _prompt_requests_extracted_answer(prompt):
            params["query"] = prompt
            params["verification_mode"] = "extract"
    return GateSpec(id=gate.id, type=gate.type, params=params)


def _broaden_query_upstream(
    params: dict[str, object],
    planned_gates_by_id: dict[str, GateSpec],
    preserve_temporal_join: bool = False,
) -> None:
    upstream_id = params.get("upstream_gate_id")
    if not upstream_id:
        return
    params["upstream_gate_id"] = _broadest_candidate_upstream(
        str(upstream_id),
        planned_gates_by_id,
        preserve_temporal_join=preserve_temporal_join,
    )


def _broadest_candidate_upstream(
    upstream_id: str,
    planned_gates_by_id: dict[str, GateSpec],
    preserve_temporal_join: bool = False,
) -> str:
    current_id = upstream_id
    seen: set[str] = set()
    while current_id not in seen:
        seen.add(current_id)
        gate = planned_gates_by_id.get(current_id)
        if gate is None:
            return current_id
        if gate.type == "temporal_join":
            if preserve_temporal_join:
                return current_id
            next_id = _visual_candidate_from_join(gate, planned_gates_by_id)
        elif gate.type in {"object_spatial_relation", "object_track", "temporal_count"}:
            next_id = gate.params.get("upstream_gate_id") or _visual_candidate_from_join(gate, planned_gates_by_id)
        elif gate.type == "ocr_text":
            next_id = _best_broad_visual_candidate(planned_gates_by_id, exclude_id=current_id)
        else:
            return current_id
        if not next_id:
            return current_id
        current_id = str(next_id)
    return upstream_id


def _best_broad_visual_candidate(
    planned_gates_by_id: dict[str, GateSpec],
    exclude_id: Optional[str] = None,
) -> Optional[str]:
    priority = {
        "local_yolo_object": 0,
        "local_yolo_pose": 1,
        "local_yolo_segmentation": 2,
        "motion": 3,
        "local_image_classification": 4,
        "color_presence": 5,
        "object_label": 6,
    }
    candidates = [
        gate
        for gate in planned_gates_by_id.values()
        if gate.id != exclude_id and gate.type in priority
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda gate: priority[gate.type]).id


def _visual_candidate_from_join(gate: GateSpec, planned_gates_by_id: dict[str, GateSpec]) -> Optional[str]:
    gate_ids = [str(item) for item in gate.params.get("gate_ids", []) if str(item)]
    if not gate_ids:
        return None
    joined_gates = [planned_gates_by_id.get(gate_id) for gate_id in gate_ids]
    if any(item is None or item.type not in VISUAL_GATE_TYPES for item in joined_gates):
        return None
    priority = {
        "local_yolo_object": 0,
        "local_yolo_pose": 1,
        "local_yolo_segmentation": 2,
        "motion": 3,
        "ocr_text": 4,
        "local_image_classification": 5,
        "color_presence": 6,
        "object_label": 7,
    }
    return min(
        gate_ids,
        key=lambda gate_id: priority.get(planned_gates_by_id[gate_id].type, 50),
    )


def _prompt_requests_extracted_answer(prompt: str) -> bool:
    text = prompt.lower()
    answer_terms = (
        "what number",
        "which number",
        "what text",
        "what word",
        "who ",
        "read the",
        "find out",
        "tell me the number",
    )
    return any(term in text for term in answer_terms)


def _prompt_is_audio_or_speech_focused(prompt: str) -> bool:
    text = prompt.lower()
    audio_terms = (
        "audio",
        "hear",
        "heard",
        "listen",
        "listening",
        "sound",
        "noise",
        "music",
        "volume",
        "pitch",
        "say",
        "says",
        "said",
        "saying",
        "speak",
        "speaks",
        "speaking",
        "spoken",
        "transcript",
        "commentary",
        "announcer",
        "voice",
        "asks",
        "asked",
    )
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in audio_terms)


def _prompt_needs_visual_window(prompt: str) -> bool:
    text = prompt.lower()
    window_terms = (
        "recent",
        "sequence",
        "just",
        "event",
        "action",
        "happened",
        "happens",
        "completed",
        "complete",
        "completing",
        "finishing",
        "finished",
        "finishes",
        "result",
        "resulted",
        "resulting",
        "transition",
        "change",
        "changes",
        "moving",
        "motion",
        "before",
        "after",
        "starts",
        "stops",
        "picks up",
        "puts down",
        "opens",
        "closes",
    )
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in window_terms)


def _event_outcome_transition_query(query: str) -> str:
    guard = (
        " If a persistent state display is visible, such as a scorebug, count, outs, score, clock, "
        "status, or UI label, answer YES only when the latest/current frame shows a terminal state "
        "that newly changed versus earlier context frames, or an explicit current result label. "
        "Answer NO if the latest frame and earlier context frames show the same relevant state, "
        "even if the action looks like a swing, miss, throw, attempt, or ordinary play result."
    )
    return query if guard in query else f"{query}{guard}"


def _prompt_likely_uses_persistent_state_display(prompt: str) -> bool:
    text = prompt.lower()
    terms = (
        "scorebug",
        "scoreboard",
        "score",
        "count",
        "outs",
        "out ",
        "inning",
        "strike",
        "strikeout",
        "struck",
        "pitcher",
        "batter",
        "game",
        "clock",
        "timer",
        "status",
        "label",
        "graphic",
        "dashboard",
        "ui",
        "goal",
        "touchdown",
        "point",
        "period",
        "quarter",
    )
    return any(term in text for term in terms)


def _normalize_temporal_change_params(params: dict[str, object]) -> None:
    text = " ".join(
        str(params.get(key, ""))
        for key in ("query", "change", "subject")
        if params.get(key) is not None
    ).lower()
    if any(term in text for term in _INITIAL_BASELINE_CHANGE_TERMS):
        params["baseline_mode"] = "initial"


def log_plan(logger: RuntimeLogger, spec: FunnelSpec, planner_model: Optional[str]) -> None:
    logger.log(None, f"Plan goal: {spec.goal}")
    if planner_model:
        logger.log(None, f"Planner model: {planner_model}")
    logger.log(
        None,
        (
            "Response tree: "
            f"yes -> {spec.response.on_match_text or _notification_from_goal(spec.goal)!r}; "
            f"no -> {spec.response.on_no_match}"
        ),
    )
    for index, gate in enumerate(spec.gates, start=1):
        details = ", ".join(f"{key}={value}" for key, value in sorted(gate.params.items()) if key != "query")
        if gate.params.get("query"):
            details = f"query={gate.params['query']!r}" + (f", {details}" if details else "")
        logger.log(None, f"Gate {index}: id={gate.id} type={gate.type} {details}".rstrip())


@dataclass(frozen=True)
class _VerifierTuning:
    """Queue and sampling knobs applied to the model verifier gate."""

    sample_interval_ms: int
    model: Optional[str] = None
    concurrent_requests: int = 3
    min_request_interval_ms: int = 200
    queued_frame_dedupe_window_ms: int = 500


def _builtin_video_spec(prompt: str, tuning: _VerifierTuning) -> FunnelSpec:
    """Funnel used when the planner is disabled or unavailable: prompt straight to the verifier."""
    return FunnelSpec.video_query(
        prompt,
        sample_interval_ms=tuning.sample_interval_ms,
        verifier_model=tuning.model,
        concurrent_requests=tuning.concurrent_requests,
        min_request_interval_ms=tuning.min_request_interval_ms,
        queued_frame_dedupe_window_ms=tuning.queued_frame_dedupe_window_ms,
    )


def build_media_spec(
    prompt: str,
    *,
    audio_only_media: bool,
    use_planner: bool,
    planner_model: Optional[str],
    verifier_model: Optional[str],
    vision_sample_seconds: float,
    verifier_sample_interval_ms: Optional[int],
    verifier_concurrent_requests: Optional[int],
    verifier_min_request_interval_ms: Optional[int],
    verifier_queued_frame_dedupe_window_ms: Optional[int],
    logger: RuntimeLogger,
    initial_context: Optional[str] = None,
    media_has_audio: bool = True,
) -> FunnelSpec:
    tuning = _VerifierTuning(
        sample_interval_ms=verifier_sample_interval_ms
        if verifier_sample_interval_ms is not None
        else int(vision_sample_seconds * 1000),
        model=verifier_model,
        **{
            key: value
            for key, value in (
                ("concurrent_requests", verifier_concurrent_requests),
                ("min_request_interval_ms", verifier_min_request_interval_ms),
                ("queued_frame_dedupe_window_ms", verifier_queued_frame_dedupe_window_ms),
            )
            if value is not None
        },
    )
    if use_planner:
        config = ModelConfig.from_env()
        if planner_model:
            config = ModelConfig(
                planner_model=planner_model,
                cheap_model=config.cheap_model,
                mid_model=config.mid_model,
                responder_model=config.responder_model,
            )
        logger.log(None, f"Sending funnel planner request model={config.planner_model}")
        planner = create_model_planner(config=config)
        try:
            planner_context = (
                initial_context
                if audio_only_media
                else _source_context_for_planner(initial_context, media_has_audio)
            )
            planned_spec = planner.plan(prompt, context=planner_context)
            logger.log(None, "funnel planner returned")
            if audio_only_media:
                spec = _audio_runtime_spec_from_planned_spec(prompt, planned_spec)
                logger.log(None, "Built realtime audio query funnel from planner output")
            else:
                spec = _video_runtime_spec_from_planned_spec(
                    prompt,
                    planned_spec,
                    sample_interval_ms=int(vision_sample_seconds * 1000),
                    verifier_model=verifier_model,
                    verifier_sample_interval_ms=verifier_sample_interval_ms,
                    verifier_concurrent_requests=verifier_concurrent_requests,
                    verifier_min_request_interval_ms=verifier_min_request_interval_ms,
                    verifier_queued_frame_dedupe_window_ms=verifier_queued_frame_dedupe_window_ms,
                    media_has_audio=media_has_audio,
                )
                logger.log(None, "Built realtime video query funnel from planner output")
        except Exception as exc:
            if audio_only_media:
                logger.log(None, f"funnel planner failed; using realtime audio query fallback error={exc}")
                spec = _audio_query_fallback_spec(prompt)
            else:
                logger.log(None, f"funnel planner failed; using original prompt as realtime query error={exc}")
                spec = _builtin_video_spec(prompt, tuning)
    else:
        spec = _audio_query_fallback_spec(prompt) if audio_only_media else _builtin_video_spec(prompt, tuning)
    tracked_response = _ensure_state_tracking_response(prompt, spec.response)
    if tracked_response is not spec.response:
        spec = replace(spec, response=tracked_response)
    return spec


def _source_context_for_planner(initial_context: Optional[str], media_has_audio: bool) -> Optional[str]:
    source_note = (
        "Source type: video with decodable audio."
        if media_has_audio
        else "Source type: video-only; no decodable audio stream is available, so do not rely on audio or transcript gates."
    )
    return f"{source_note}\n{initial_context}" if initial_context else source_note


def initial_frame_context(media: str, model: str, logger: RuntimeLogger) -> Optional[str]:
    frame = load_initial_video_frame(media)
    if frame is None:
        return None
    image_url = _encode_frame_data_url(frame)
    logger.log(None, f"Sending initial frame context request model={model}")
    context = active_provider().complete_text(
        ModelRequest(
            model=model,
            effort="low",
            max_output_tokens=160,
            system=(
                "Describe the first video frame for a realtime monitoring funnel planner. "
                "Be concise. Mention visible people, objects, text, setting, and any starting state "
                "that could clarify later user prompts."
            ),
            content=[TextPart("Initial video frame:"), ImagePart(image_url)],
        )
    ).strip()
    logger.log(None, f"Initial frame context: {context}")
    return context


def _encode_frame_data_url(frame: object) -> str:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Frame encoding requires the uv-managed OpenCV dependency.") from exc
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("Could not encode initial frame as JPEG.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
