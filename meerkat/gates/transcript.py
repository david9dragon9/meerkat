from __future__ import annotations

from typing import Iterable, Optional, Set

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.gates.base import Gate


class TranscriptKeywordGate(Gate):
    def __init__(self, gate_id: str, keywords: Iterable[str], upstream_gate_id: Optional[str] = None) -> None:
        input_types = [EventType.GATE_FIRE] if upstream_gate_id else [EventType.TRANSCRIPT_DELTA, EventType.USER_PROMPT]
        super().__init__(gate_id, input_types)
        self.keywords: Set[str] = {keyword.lower() for keyword in keywords}
        self.upstream_gate_id = upstream_gate_id

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        raw_text = event.payload.get("text")
        upstream_evidence = None
        if event.type == EventType.GATE_FIRE:
            fire = event.payload.get("gate_fire")
            if not isinstance(fire, GateFire) or fire.gate_id != self.upstream_gate_id:
                return None
            upstream_evidence = fire.evidence
            raw_text = upstream_evidence.get("text_window") or upstream_evidence.get("text")
        text = str(raw_text or "")
        lowered = text.lower()
        matches = sorted(keyword for keyword in self.keywords if keyword in lowered)
        if not matches:
            return None
        evidence = {
            "matches": matches,
            "text": text,
            "text_window": text,
            "stream_time_ms": event.stream_time_ms,
        }
        if upstream_evidence:
            evidence["upstream_evidence"] = upstream_evidence
            if "text" in upstream_evidence:
                evidence["current_text"] = upstream_evidence["text"]
            if upstream_evidence.get("name_context"):
                evidence["name_context"] = upstream_evidence["name_context"]
                evidence["text"] = f"{upstream_evidence['name_context']} {text}".strip()
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason=f"Matched transcript keywords: {', '.join(matches)}",
            evidence=evidence,
        )
