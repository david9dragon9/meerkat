from __future__ import annotations

from typing import Iterable, Optional, Set

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.gates.base import Gate


class ObjectLabelGate(Gate):
    """Cheap object gate using labels already attached to a frame.

    This is the placeholder interface for a real detector. The first production
    detector can preserve this output contract and replace only the label source.
    """

    def __init__(self, gate_id: str, classes: Iterable[str], min_confidence: float = 0.5) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.classes: Set[str] = {item.lower() for item in classes}
        self.min_confidence = min_confidence

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        objects = [str(item).lower() for item in event.payload.get("objects", [])]
        matches = sorted(set(objects).intersection(self.classes))
        if not matches:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason=f"Detected requested object labels: {', '.join(matches)}",
            evidence={"matches": matches, "stream_time_ms": event.stream_time_ms},
        )
