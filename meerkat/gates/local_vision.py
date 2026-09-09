from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Set

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.gates.base import Gate
from meerkat.gates.labels import DETECTOR_LABEL_ALIASES


class LocalYoloObjectGate(Gate):
    """Cheap local object detector gate.

    This gate is intended to run continuously on the realtime stream and emit
    candidate frames for more expensive gates. It uses a latest-frame queue in
    the runtime, so slow detector work drops stale frames instead of delaying
    playback.
    """

    def __init__(
        self,
        gate_id: str,
        classes: Iterable[str],
        sample_interval_ms: int = 200,
        min_confidence: float = 0.35,
        model_path: str = "yolov8n.pt",
        trigger_on_any_detection: bool = False,
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.classes = {item.lower() for item in classes}
        self.accepted_labels = self._expand_labels(self.classes)
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.model_path = model_path
        self.trigger_on_any_detection = trigger_on_any_detection
        self._last_sample_time_ms = -sample_interval_ms
        self._model = None

    async def warmup(self) -> None:
        if self.logger:
            self.logger.log(None, f"Warming up local YOLO gate gate={self.gate_id} model={self.model_path}")
        start = perf_counter()
        await asyncio.to_thread(self._warmup_model)
        if self.logger:
            self.logger.log(None, f"Local YOLO gate warmed gate={self.gate_id} elapsed={perf_counter() - start:.2f}s")

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms

        start = perf_counter()
        detections = await asyncio.to_thread(self._detect, frame)
        elapsed = perf_counter() - start
        matches = [item for item in detections if item["label"] in self.accepted_labels]
        candidate_matches = matches if matches else (detections if self.trigger_on_any_detection else [])
        if self.logger and candidate_matches:
            labels = ",".join(sorted({str(item["label"]) for item in candidate_matches}))
            status = "matched" if matches else "candidate"
            self.logger.log(
                event.stream_time_ms,
                f"Cheap YOLO object gate returned gate={self.gate_id} status={status} labels={labels} request={elapsed:.2f}s",
            )
        if not candidate_matches:
            return None

        return GateFire(
            gate_id=self.gate_id,
            confidence=max(float(item["confidence"]) for item in candidate_matches),
            reason=(
                "Local YOLO detected target objects: "
                if matches
                else "Local YOLO detected visual candidates: "
            )
            + ", ".join(sorted({str(item["label"]) for item in candidate_matches})),
            evidence={
                "detections": candidate_matches,
                "target_matches": matches,
                "frame": frame,
                "stream_time_ms": event.stream_time_ms,
                "sample_interval_ms": self.sample_interval_ms,
            },
        )

    def _detect(self, frame: object) -> List[Dict[str, object]]:
        model = self._get_model()
        results = model.predict(frame, verbose=False, conf=self.min_confidence)
        detections: List[Dict[str, object]] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                confidence = float(box.conf[0])
                label = str(names[int(box.cls[0])]).lower()
                if confidence < self.min_confidence:
                    continue
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                detections.append({"label": label, "confidence": confidence, "bbox": xyxy})
        return detections

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "LocalYoloObjectGate requires ultralytics. Run `uv sync` or `uv run ...` "
                "after the dependency update."
            ) from exc
        self._model = YOLO(self.model_path)
        return self._model

    def _warmup_model(self) -> None:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("LocalYoloObjectGate warmup requires numpy.") from exc

        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self._detect(dummy_frame)

    def _expand_labels(self, labels: Set[str]) -> Set[str]:
        expanded = set(labels)
        for label in labels:
            expanded.update(DETECTOR_LABEL_ALIASES.get(label, set()))
        return expanded
