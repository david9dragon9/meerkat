from __future__ import annotations

from typing import Dict, Set


#: Colloquial phrases a user is likely to type, mapped to the detector labels
#: that should satisfy them. Local detectors are trained on fixed label sets
#: (COCO, for the bundled YOLO weights), so a request for a "beach ball" has to
#: match the "sports ball" class it was actually trained on.
DETECTOR_LABEL_ALIASES: Dict[str, Set[str]] = {
    "beach ball": {"sports ball", "ball"},
    "ball": {"sports ball"},
}


def label_matches(detected_label: str, requested_label: str) -> bool:
    """Whether a detector's label satisfies a requested label, aliases included."""
    detected = detected_label.lower()
    requested = requested_label.lower()
    return detected == requested or detected in DETECTOR_LABEL_ALIASES.get(requested, set())
