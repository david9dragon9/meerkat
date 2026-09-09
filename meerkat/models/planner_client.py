from __future__ import annotations

import json
import asyncio
from time import perf_counter
from typing import Any, Dict, Optional

from meerkat.events import GateFire, ModelResponse
from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec, StateUpdateSpec
from meerkat.models.config import ModelConfig
from meerkat.models.provider import ModelBacked, ModelRequest, TextPart
from meerkat.runtime.logging import RuntimeLogger


FUNNEL_SCHEMA: Dict[str, Any] = {
    "name": "funnel_spec",
    "type": "json_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["goal", "gates", "response"],
        "properties": {
            "goal": {"type": "string"},
            "gates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "type", "params"],
                    "properties": {
                        "id": {"type": "string"},
                        "type": {
                            "enum": [
                                "object_label",
                                "local_yolo_object",
                                "local_yolo_segmentation",
                                "local_yolo_pose",
                                "local_image_classification",
                                "model_vision_object",
                                "model_vision_query",
                                "model_frame_change",
                                "model_state_monitor",
                                "object_track",
                                "ocr_text",
                                "motion",
                                "color_presence",
                                "object_spatial_relation",
                                "transcript_keyword",
                                "audio_volume",
                                "audio_pitch",
                                "audio_cadence",
                                "local_realtime_transcription",
                                "hosted_transcription",
                                "model_transcript_query",
                                "temporal_join",
                                "temporal_count",
                            ]
                        },
                        "params": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "classes",
                                "baseline_mode",
                                "change",
                                "color",
                                "buffer_ms",
                                "compute_type",
                                "concurrent_requests",
                                "keywords",
                                "language",
                                "gate_ids",
                                "join_window_ms",
                                "max_distance_ratio",
                                "max_pitch_hz",
                                "max_volume_db",
                                "max_words_per_minute",
                                "min_confidence",
                                "min_area_ratio",
                                "min_delta_ratio",
                                "min_motion_ratio",
                                "min_pitch_hz",
                                "min_request_interval_ms",
                                "min_volume_db",
                                "min_words_per_minute",
                                "model",
                                "model_path",
                                "object",
                                "pattern",
                                "pose",
                                "query",
                                "queued_frame_dedupe_window_ms",
                                "relation",
                                "required_count",
                                "sample_interval_ms",
                                "service_tier",
                                "state_updates",
                                "state_interval_ms",
                                "subject",
                                "trigger_on_any_detection",
                                "upstream_gate_id",
                                "use_tesseract",
                                "verification_mode",
                                "window_ms",
                            ],
                            "properties": {
                                "classes": {"type": ["array", "null"], "items": {"type": "string"}},
                                "baseline_mode": {"type": ["string", "null"]},
                                "buffer_ms": {"type": ["integer", "null"]},
                                "change": {"type": ["string", "null"]},
                                "color": {"type": ["string", "null"]},
                                "compute_type": {"type": ["string", "null"]},
                                "concurrent_requests": {"type": ["integer", "null"]},
                                "keywords": {"type": ["array", "null"], "items": {"type": "string"}},
                                "language": {"type": ["string", "null"]},
                                "gate_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                                "join_window_ms": {"type": ["integer", "null"]},
                                "max_distance_ratio": {"type": ["number", "null"]},
                                "max_pitch_hz": {"type": ["number", "null"]},
                                "max_volume_db": {"type": ["number", "null"]},
                                "max_words_per_minute": {"type": ["number", "null"]},
                                "min_confidence": {"type": ["number", "null"]},
                                "min_area_ratio": {"type": ["number", "null"]},
                                "min_delta_ratio": {"type": ["number", "null"]},
                                "min_motion_ratio": {"type": ["number", "null"]},
                                "min_pitch_hz": {"type": ["number", "null"]},
                                "min_request_interval_ms": {"type": ["integer", "null"]},
                                "min_volume_db": {"type": ["number", "null"]},
                                "min_words_per_minute": {"type": ["number", "null"]},
                                "model": {"type": ["string", "null"]},
                                "model_path": {"type": ["string", "null"]},
                                "object": {"type": ["string", "null"]},
                                "pattern": {"type": ["string", "null"]},
                                "pose": {"type": ["string", "null"]},
                                "query": {"type": ["string", "null"]},
                                "queued_frame_dedupe_window_ms": {"type": ["integer", "null"]},
                                "relation": {"type": ["string", "null"]},
                                "required_count": {"type": ["integer", "null"]},
                                "sample_interval_ms": {"type": ["integer", "null"]},
                                "service_tier": {
                                    "type": ["string", "null"],
                                    "enum": ["priority", "auto", "default", "flex", "scale", None],
                                },
                                "state_updates": {
                                    "type": ["array", "null"],
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["key", "operation", "value"],
                                        "properties": {
                                            "key": {"type": "string"},
                                            "operation": {
                                                "enum": [
                                                    "increment",
                                                    "increment_unique",
                                                    "set",
                                                    "append",
                                                    "set_text",
                                                    "write_text",
                                                    "append_text",
                                                    "note",
                                                ]
                                            },
                                            "value": {"type": ["number", "integer", "string", "boolean", "null"]},
                                        },
                                    },
                                },
                                "subject": {"type": ["string", "null"]},
                                "state_interval_ms": {"type": ["integer", "null"]},
                                "trigger_on_any_detection": {"type": ["boolean", "null"]},
                                "upstream_gate_id": {"type": ["string", "null"]},
                                "use_tesseract": {"type": ["boolean", "null"]},
                                "verification_mode": {"type": ["string", "null"]},
                                "window_ms": {"type": ["integer", "null"]},
                            },
                        },
                    },
                },
            },
            "response": {
                "type": "object",
                "additionalProperties": False,
                "required": ["model", "cooldown_seconds", "style", "on_match_text", "on_no_match", "state_updates"],
                "properties": {
                    "model": {"type": "string"},
                    "cooldown_seconds": {"type": "number"},
                    "style": {"type": "string"},
                    "on_match_text": {"type": ["string", "null"]},
                    "on_no_match": {"type": "string"},
                    "state_updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["key", "operation", "value"],
                            "properties": {
                                "key": {"type": "string"},
                                "operation": {
                                    "enum": [
                                        "increment",
                                        "increment_unique",
                                        "set",
                                        "append",
                                        "set_text",
                                        "write_text",
                                        "append_text",
                                        "note",
                                    ]
                                },
                                "value": {"type": ["number", "integer", "string", "boolean", "null"]},
                            },
                        },
                    },
                },
            },
        },
    },
    "strict": True,
}


class ModelFunnelPlanner(ModelBacked):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig.from_env()

    def plan(self, prompt: str, context: Optional[str] = None) -> FunnelSpec:
        user_content = prompt
        if context:
            user_content = (
                f"User prompt: {prompt}\n\n"
                "Additional realtime source context available before planning:\n"
                f"{context}\n\n"
                "Use this context only to choose better cheap gates and clarify ambiguous references. "
                "Do not alert solely because something is visible in the initial context unless the user asked for that."
            )
        data = self.provider.complete_json(
            ModelRequest(
                model=self.config.planner_model,
                effort="medium",
                schema=FUNNEL_SCHEMA,
                system=(
                        "Create a realtime monitoring funnel. Prefer cheap parallel gates first. "
                        "When the source has video frames and the user asks about a visual event, action, "
                        "state, object relationship, gesture, sport play, or on-screen text, include visual "
                        "gates in the alert path. Do not build a transcript-only funnel for a visual event "
                        "unless the user explicitly asks about spoken commentary, dialogue, or audio mentions. "
                        "For video with audio, audio gates may be supporting evidence, but visual gates should "
                        "remain available for visually observable prompts. "
                        "Use local_yolo_object as the first visual gate for common COCO objects "
                        "such as person, dog, cat, car, and sports ball. Use transcript_keyword "
                        "for quoted speech, known phrases, keyword mentions, or user-input phrases. "
                        "For audio/video sources, use local_realtime_transcription as the default cheap "
                        "realtime speech-to-text gate before transcript_keyword, audio_cadence, or "
                        "model_transcript_query. Configure it with model='small.en' for name/value "
                        "extraction or model='base.en' for simpler keyword-only tasks, "
                        "compute_type='int8', buffer_ms around 2000, sample_interval_ms around 1000, "
                        "and min_volume_db around -45. "
                        "Use audio_volume for cheap loudness/silence/gunshot/clap/shout candidates, "
                        "audio_pitch for rough high/low pitch candidates, and audio_cadence for fast/slow "
                        "speaking based on transcript deltas. Use hosted_transcription only when "
                        "local transcription is not accurate enough or a higher-quality paid fallback "
                        "is explicitly useful; wire transcript_keyword or model_transcript_query after "
                        "local/openai transcription with upstream_gate_id. "
                        "Use model_transcript_query when transcript text needs semantic verification, "
                        "speaker intent, extraction, or a higher quality text-only model check. "
                        "Use temporal_join to combine audio and video gates, for example a person visible "
                        "and someone says a phrase within join_window_ms, or a loud sound while an object moves. "
                        "Use temporal_count to confirm noisy signals. "
                        "Use ocr_text for visible text, captions, UI labels, signs, scoreboards, "
                        "receipts, or web pages. Use local_yolo_segmentation when mask area, region "
                        "coverage, boundaries, or foreground/background shape matters. Use local_yolo_pose "
                        "for people, body position, arms up, falls, sitting/standing, or gesture candidates. "
                        "Use local_image_classification for whole-scene/category cues such as beach, indoor, "
                        "food, fire, screenshot type, or broad visual style. Use motion for movement, "
                        "stillness, scene changes, or activity onset. Use color_presence for visual state "
                        "cues like red warning banners, green lights, blue screens, or dominant color alerts. "
                        "Use object_spatial_relation after local_yolo_object when two detected objects need "
                        "a cheap relation check such as near, overlap, left_of, right_of, above, or below. "
                        "Use object_track after local_yolo_object for tracked object changes over time, "
                        "including moving_apart, moving_together, moving, appearing, disappearing, or crossing. "
                        "Use model_frame_change for temporal visual changes that require before/after evidence, "
                        "such as hands separating, objects opening or closing, pick-up/drop events, posture changes, "
                        "state transitions, or any event where a single frame is ambiguous. Set baseline_mode='initial' "
                        "when the user's language depends on the starting state, and baseline_mode='previous' for "
                        "short local changes. Use model_state_monitor to periodically summarize current visual state "
                        "when the funnel needs ongoing context about what is currently true in the video. "
                        "For broadcast, sports, games, dashboards, websites, or any stream with persistent on-screen "
                        "state graphics, treat visible UI/state changes as high-quality cheap evidence. Include OCR "
                        "or a frame-change/vision verifier that looks at the relevant scorebug, count, outs, score, "
                        "clock, status label, selected item, notification, or other persistent state. For sports "
                        "events, reason from the rules and the visible broadcast state: a requested event may be "
                        "best confirmed by a count, out, score, possession, or clock transition rather than by player "
                        "motion alone. The verifier query should name the concrete state transition when it can be "
                        "inferred from the prompt and initial context, and should fire as soon as that transition is "
                        "visible, not later when a replay, recap, lower-third, or next-player graphic appears. "
                        "When you add an OCR/state-graphics gate because visible text or persistent state is relevant "
                        "to the user alert, wire that gate into the alert path using temporal_join, temporal_count, "
                        "or as the direct upstream candidate for the single model verifier. Do not leave OCR/state "
                        "gates as isolated terminal leaves unless the user only asked to update internal state. "
                        "Use model_vision_query when the user asks for a visual action, state, "
                        "or relationship that cannot be confirmed by cheap object presence alone, "
                        "but wire it after cheap candidate gates with upstream_gate_id. "
                        "Prefer building a funnel of multiple cheap gates before model_vision_query. "
                        "Never chain one model-backed gate after another on a user-alert path: "
                        "there must be at most one model verifier between a video frame candidate "
                        "or audio/transcript candidate and the final user alert. Use cheap gates before the single model verifier, "
                        "keep the model verifier wired to a broad cheap candidate gate rather than a "
                        "brittle local relation/temporal filter, and use only local/response-tree "
                        "gates after it. "
                        "For model_vision_query params, include query, sample_interval_ms, "
                        "min_confidence, model, upstream_gate_id, and service_tier='default'. "
                        "Use verification_mode='binary' for standard yes/no realtime verification. "
                        "Use verification_mode='extract' when the user asks for a visible value, "
                        "number, word, or text; the verifier will return NO when "
                        "unclear or the extracted value as evidence.text when clear. "
                        "For model_transcript_query, use verification_mode='binary' for yes/no transcript "
                        "conditions and verification_mode='extract' for spoken values, names, counts, or text. "
                        "Choose the realtime verification model by difficulty: use "
                        f"{self.config.cheap_model} for simple yes/no visual checks, "
                        f"{self.config.mid_model} for moderate ambiguity, and "
                        f"{self.config.planner_model} only for hard reasoning. "
                        "Set concurrent_requests between 1 and 5 and min_request_interval_ms >= 200 "
                        "for expensive realtime model gates. "
                        "Set queued_frame_dedupe_window_ms to 500 by default so queued model "
                        "requests skip frames close to already-sent request frames when backlog exists. "
                        "Use response.state_updates when the user asks to track internal state, "
                        "such as counts. For example, when asked to count how many people "
                        "appear, increment key='person_count' by 1 and set "
                        "response.on_match_text='People seen: {state.person_count}'. "
                        "Keep state keys lowercase snake_case. "
                        "Use increment_unique with value='{evidence.matches}' for OCR/text counts "
                        "where the same on-screen word can fire across multiple frames. "
                        "Use gate params state_updates for realtime state actions that should happen "
                        "when that gate fires, even if no user-facing response should be sent. "
                        "For natural-language memory, use set_text or append_text on keys such as "
                        "'scene_notes' or 'running_summary'; model gates can read current internal "
                        "state during later stream requests. A state update value may reference "
                        "gate evidence with templates like {evidence.state_summary}. "
                        "Pre-generate response.on_match_text as a short user-facing sentence. "
                        "When a gate uses verification_mode='extract', put the extracted value in "
                        "it with {evidence.text}, otherwise every report reads the same and the "
                        "user is only told once. "
                        "Set response.cooldown_seconds low (0-2) when the user asks for a value or "
                        "a running state, so a changed answer is reported promptly; use a longer "
                        "cooldown only for one-off event alerts. "
                        "Set response.on_no_match='ignore'. The realtime verifier should only "
                        "return YES or NO when binary verification is enough."
                ),
                content=[TextPart(user_content)],
            )
        )
        gates = [self._parse_gate_spec(item) for item in data["gates"]]
        response_spec = self._parse_response_spec(data["response"])
        return FunnelSpec(goal=data["goal"], gates=gates, response=response_spec)

    def _parse_gate_spec(self, item: Dict[str, Any]) -> GateSpec:
        gate_type = item["type"]
        params = {key: value for key, value in item.get("params", {}).items() if value is not None}
        if gate_type in {"model_vision_object", "model_vision_query"}:
            model = str(params.get("model", ""))
            if model not in self.config.tiers:
                params["model"] = self.config.mid_model
            if (
                gate_type == "model_vision_query"
                and params.get("verification_mode") == "binary"
                and self._is_easy_binary_visual_check(str(params.get("query", "")))
            ):
                params["model"] = self.config.cheap_model
        return GateSpec(id=item["id"], type=gate_type, params=params)

    def _parse_response_spec(self, item: Dict[str, Any]) -> ResponseSpec:
        model = str(item.get("model", "local"))
        if item.get("on_match_text") and item.get("on_no_match") == "ignore":
            model = "local"
        elif model != "local" and model not in self.config.tiers:
            model = "local"
        return ResponseSpec(
            model=model,
            cooldown_seconds=float(item["cooldown_seconds"]),
            style=str(item["style"]),
            on_match_text=item.get("on_match_text"),
            on_no_match=str(item.get("on_no_match", "ignore")),
            state_updates=[
                StateUpdateSpec(
                    key=str(update["key"]),
                    operation=str(update["operation"]),
                    value=update.get("value"),
                )
                for update in item.get("state_updates", [])
            ],
        )

    def _is_easy_binary_visual_check(self, query: str) -> bool:
        text = query.lower()
        hard_terms = {
            "before",
            "after",
            "same",
            "different",
            "count",
            "number",
            "compare",
            "until",
            "sequence",
            "occluded",
            "hidden",
            "partially",
            "intent",
        }
        return not any(term in text for term in hard_terms)



class ModelResponder(ModelBacked):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.logger: Optional[RuntimeLogger] = None

    def set_logger(self, logger: RuntimeLogger) -> None:
        self.logger = logger

    async def respond(self, spec: FunnelSpec, fire: GateFire) -> ModelResponse:
        payload = {
            "goal": spec.goal,
            "trigger_gate_id": fire.gate_id,
            "reason": fire.reason,
            "evidence": fire.evidence,
            "style": spec.response.style,
        }
        stream_time_ms = int(fire.evidence.get("stream_time_ms", 0))
        if self.logger:
            self.logger.log(
                stream_time_ms,
                f"Sending response request model={self.config.responder_model} trigger={fire.gate_id}",
            )
        request_start = perf_counter()
        text = await asyncio.to_thread(
            self.provider.complete_text,
            ModelRequest(
                model=self.config.responder_model,
                effort="low",
                system="Write a concise user-facing realtime monitoring notification.",
                content=[TextPart(json.dumps(payload, sort_keys=True))],
            ),
        )
        request_elapsed = perf_counter() - request_start
        if self.logger:
            self.logger.log(
                stream_time_ms,
                f"Response request returned model={self.config.responder_model} request={request_elapsed:.2f}s",
            )
        return ModelResponse(text=text, trigger_gate_id=fire.gate_id, evidence=fire.evidence)

