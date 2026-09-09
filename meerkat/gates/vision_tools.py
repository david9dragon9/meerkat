from __future__ import annotations

import asyncio
import re
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.gates.base import Gate
from meerkat.gates.labels import label_matches


FOOD_WORDS = {
    "apple",
    "banana",
    "bread",
    "burger",
    "cake",
    "candy",
    "carrot",
    "cheese",
    "chicken",
    "chocolate",
    "coffee",
    "cookie",
    "corn",
    "donut",
    "egg",
    "fish",
    "fries",
    "grape",
    "ice cream",
    "juice",
    "lemon",
    "meat",
    "milk",
    "noodles",
    "orange",
    "pancake",
    "pasta",
    "pizza",
    "potato",
    "rice",
    "salad",
    "sandwich",
    "soup",
    "sushi",
    "taco",
    "tea",
    "tomato",
    "waffle",
}


class OCRTextGate(Gate):
    """OCR/text gate over attached OCR payloads, with optional pytesseract fallback."""

    def __init__(
        self,
        gate_id: str,
        keywords: Iterable[str],
        pattern: Optional[str] = None,
        sample_interval_ms: int = 1000,
        min_confidence: float = 0.5,
        use_tesseract: bool = False,
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.keywords = self._normalize_keywords(keywords)
        self.pattern = re.compile(pattern, re.I) if pattern else None
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.use_tesseract = use_tesseract
        self._last_sample_time_ms = -sample_interval_ms

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        text = self._payload_text(event)
        if not text and self.use_tesseract and event.payload.get("frame") is not None:
            text = await asyncio.to_thread(self._run_tesseract, event.payload["frame"])
        if not text:
            return None
        lowered = text.lower()
        keyword_matches = sorted(keyword for keyword in self.keywords if self._keyword_matches(keyword, lowered))
        pattern_match = bool(self.pattern.search(text)) if self.pattern else False
        any_text_match = not self.keywords and self.pattern is None
        if not keyword_matches and not pattern_match and not any_text_match:
            return None
        matches = keyword_matches + (["pattern"] if pattern_match else []) + (["text"] if any_text_match else [])
        return GateFire(
            gate_id=self.gate_id,
            confidence=1.0,
            reason=f"OCR text matched: {', '.join(matches)}",
            evidence={
                "text": text,
                "matches": matches,
                "stream_time_ms": event.stream_time_ms,
                "frame": event.payload.get("frame"),
            },
        )

    def _payload_text(self, event: StreamEvent) -> str:
        values = [
            event.payload.get("ocr_text"),
            event.payload.get("screen_text"),
            event.payload.get("text"),
        ]
        regions = event.payload.get("text_regions")
        if isinstance(regions, list):
            values.extend(region.get("text") for region in regions if isinstance(region, dict))
        return "\n".join(str(value) for value in values if value)

    def _normalize_keywords(self, keywords: Iterable[str]) -> Set[str]:
        normalized = {item.strip().lower() for item in keywords if item.strip()}
        if "food" in normalized:
            normalized.update(FOOD_WORDS)
            normalized.remove("food")
        return normalized

    def _keyword_matches(self, keyword: str, lowered_text: str) -> bool:
        pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
        return bool(re.search(pattern, lowered_text))

    def _run_tesseract(self, frame: object) -> str:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OCRTextGate use_tesseract=True requires pytesseract. Run `uv sync`.") from exc
        try:
            return str(pytesseract.image_to_string(self._ocr_input_image(frame)))
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "OCRTextGate use_tesseract=True requires the system `tesseract` binary. "
                "Install it separately, for example `brew install tesseract` on macOS."
            ) from exc

    def _ocr_input_image(self, frame: object) -> object:
        """Build a general OCR canvas that emphasizes common UI/text regions."""
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return frame

        if not hasattr(frame, "shape"):
            return frame
        image = frame
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return frame

        crops = self._ocr_region_crops(image, width, height)
        rows = []
        target_width = min(max(width * 2, 960), 1800)
        separator_height = 24
        for crop in crops:
            if crop.size == 0:
                continue
            scale = target_width / max(crop.shape[1], 1)
            resized_height = max(1, int(crop.shape[0] * scale))
            resized = cv2.resize(crop, (target_width, resized_height), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            enhanced = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
            rows.append(enhanced)
            rows.append(np.full((separator_height, target_width), 255, dtype=enhanced.dtype))
        if not rows:
            return frame
        return np.vstack(rows)

    def _ocr_region_crops(self, image: object, width: int, height: int) -> list[object]:
        third_h = max(1, height // 3)
        half_h = max(1, height // 2)
        half_w = max(1, width // 2)
        return [
            image,
            image[:third_h, :],
            image[-third_h:, :],
            image[:, :half_w],
            image[:, -half_w:],
            image[-half_h:, :half_w],
            image[-half_h:, -half_w:],
            image[:half_h, :half_w],
            image[:half_h, -half_w:],
        ]


class LocalYoloSegmentationGate(Gate):
    """Cheap local segmentation gate using an Ultralytics segmentation model."""

    def __init__(
        self,
        gate_id: str,
        classes: Iterable[str],
        sample_interval_ms: int = 500,
        min_confidence: float = 0.35,
        min_area_ratio: float = 0.0,
        model_path: str = "yolo11n-seg.pt",
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.classes = {item.lower() for item in classes}
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.min_area_ratio = min_area_ratio
        self.model_path = model_path
        self._last_sample_time_ms = -sample_interval_ms
        self._model = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._warmup_model)

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        start = perf_counter()
        segments = await asyncio.to_thread(self._segment, frame)
        elapsed = perf_counter() - start
        matches = [
            item
            for item in segments
            if (not self.classes or str(item["label"]).lower() in self.classes)
            and float(item.get("area_ratio", 0.0)) >= self.min_area_ratio
        ]
        if self.logger and matches:
            labels = ",".join(sorted({str(item["label"]) for item in matches}))
            self.logger.log(
                event.stream_time_ms,
                f"Segmentation gate returned gate={self.gate_id} labels={labels} request={elapsed:.2f}s",
            )
        if not matches:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=max(float(item["confidence"]) for item in matches),
            reason="Local segmentation matched: " + ", ".join(sorted({str(item["label"]) for item in matches})),
            evidence={"segments": matches, "frame": frame, "stream_time_ms": event.stream_time_ms},
        )

    def _segment(self, frame: object) -> List[Dict[str, object]]:
        model = self._get_model()
        results = model.predict(frame, verbose=False, conf=self.min_confidence)
        frame_area = float(getattr(frame, "shape", [1, 1])[0] * getattr(frame, "shape", [1, 1])[1])
        output: List[Dict[str, object]] = []
        for result in results:
            names = result.names
            masks = getattr(result, "masks", None)
            for index, box in enumerate(result.boxes):
                confidence = float(box.conf[0])
                if confidence < self.min_confidence:
                    continue
                bbox = [float(value) for value in box.xyxy[0].tolist()]
                area_ratio = self._bbox_area_ratio(bbox, frame_area)
                if masks is not None and getattr(masks, "xy", None) is not None and index < len(masks.xy):
                    area_ratio = max(area_ratio, self._polygon_area_ratio(masks.xy[index], frame_area))
                output.append(
                    {
                        "label": str(names[int(box.cls[0])]).lower(),
                        "confidence": confidence,
                        "bbox": bbox,
                        "area_ratio": area_ratio,
                    }
                )
        return output

    def _get_model(self) -> object:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def _warmup_model(self) -> None:
        import numpy as np  # type: ignore

        self._segment(np.zeros((640, 640, 3), dtype=np.uint8))

    def _bbox_area_ratio(self, bbox: Sequence[float], frame_area: float) -> float:
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        return (width * height) / max(frame_area, 1.0)

    def _polygon_area_ratio(self, points: object, frame_area: float) -> float:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return 0.0
        contour = np.asarray(points, dtype=np.float32)
        return float(cv2.contourArea(contour)) / max(frame_area, 1.0)


class LocalYoloPoseGate(Gate):
    """Pose gate for person/keypoint presence and simple posture heuristics."""

    def __init__(
        self,
        gate_id: str,
        pose: str = "person",
        sample_interval_ms: int = 500,
        min_confidence: float = 0.35,
        model_path: str = "yolo11n-pose.pt",
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.pose = pose
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.model_path = model_path
        self._last_sample_time_ms = -sample_interval_ms
        self._model = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._warmup_model)

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        poses = await asyncio.to_thread(self._estimate, frame)
        matches = [pose for pose in poses if self._pose_matches(pose)]
        if not matches:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=max(float(item["confidence"]) for item in matches),
            reason=f"Pose gate matched: {self.pose}",
            evidence={"poses": matches, "frame": frame, "stream_time_ms": event.stream_time_ms},
        )

    def _estimate(self, frame: object) -> List[Dict[str, object]]:
        model = self._get_model()
        results = model.predict(frame, verbose=False, conf=self.min_confidence)
        output: List[Dict[str, object]] = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            for index, box in enumerate(result.boxes):
                confidence = float(box.conf[0])
                if confidence < self.min_confidence:
                    continue
                item: Dict[str, object] = {
                    "label": "person",
                    "confidence": confidence,
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                }
                if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                    item["keypoints"] = keypoints.xy[index].tolist()
                output.append(item)
        return output

    def _pose_matches(self, pose: Dict[str, object]) -> bool:
        if self.pose in {"person", "any_person", "pose"}:
            return True
        if self.pose == "arms_up":
            keypoints = pose.get("keypoints")
            if not isinstance(keypoints, list) or len(keypoints) < 11:
                return False
            left_wrist_y, right_wrist_y = keypoints[9][1], keypoints[10][1]
            left_shoulder_y, right_shoulder_y = keypoints[5][1], keypoints[6][1]
            return left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y
        return True

    def _get_model(self) -> object:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def _warmup_model(self) -> None:
        import numpy as np  # type: ignore

        self._estimate(np.zeros((640, 640, 3), dtype=np.uint8))


class LocalImageClassificationGate(Gate):
    """Image classification gate using an Ultralytics classification model."""

    def __init__(
        self,
        gate_id: str,
        classes: Iterable[str],
        sample_interval_ms: int = 1000,
        min_confidence: float = 0.35,
        model_path: str = "yolo11n-cls.pt",
    ) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.classes = {item.lower() for item in classes}
        self.sample_interval_ms = sample_interval_ms
        self.min_confidence = min_confidence
        self.model_path = model_path
        self._last_sample_time_ms = -sample_interval_ms
        self._model = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._warmup_model)

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        labels = await asyncio.to_thread(self._classify, frame)
        matches = [item for item in labels if str(item["label"]).lower() in self.classes]
        if not matches:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=max(float(item["confidence"]) for item in matches),
            reason="Image classification matched: " + ", ".join(str(item["label"]) for item in matches),
            evidence={"classes": matches, "frame": frame, "stream_time_ms": event.stream_time_ms},
        )

    def _classify(self, frame: object) -> List[Dict[str, object]]:
        model = self._get_model()
        result = model.predict(frame, verbose=False)[0]
        probs = result.probs
        names = result.names
        output: List[Dict[str, object]] = []
        for class_id in probs.top5:
            confidence = float(probs.data[class_id])
            if confidence >= self.min_confidence:
                output.append({"label": str(names[int(class_id)]).lower(), "confidence": confidence})
        return output

    def _get_model(self) -> object:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def _warmup_model(self) -> None:
        import numpy as np  # type: ignore

        self._classify(np.zeros((224, 224, 3), dtype=np.uint8))


class MotionGate(Gate):
    """Cheap frame-difference motion gate."""

    def __init__(self, gate_id: str, min_motion_ratio: float = 0.02, sample_interval_ms: int = 200) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.min_motion_ratio = min_motion_ratio
        self.sample_interval_ms = sample_interval_ms
        self._last_sample_time_ms = -sample_interval_ms
        self._previous_gray = None

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        motion_ratio = await asyncio.to_thread(self._motion_ratio, frame)
        if motion_ratio < self.min_motion_ratio:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=min(1.0, motion_ratio / max(self.min_motion_ratio, 0.001)),
            reason=f"Motion ratio {motion_ratio:.3f} exceeded threshold.",
            evidence={"motion_ratio": motion_ratio, "stream_time_ms": event.stream_time_ms, "frame": frame},
        )

    def _motion_ratio(self, frame: object) -> float:
        import cv2  # type: ignore

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            self._previous_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self._previous_gray)
        self._previous_gray = gray
        return float((diff > 25).mean())


class ColorPresenceGate(Gate):
    """Cheap color gate for screens, objects, and UI states."""

    COLOR_RANGES = {
        "red": ((0, 80, 80), (10, 255, 255), (170, 80, 80), (180, 255, 255)),
        "green": ((35, 60, 60), (85, 255, 255)),
        "blue": ((90, 60, 60), (130, 255, 255)),
        "yellow": ((20, 60, 60), (35, 255, 255)),
        "orange": ((10, 80, 80), (25, 255, 255)),
        "purple": ((130, 50, 50), (165, 255, 255)),
        "white": ((0, 0, 200), (180, 40, 255)),
        "black": ((0, 0, 0), (180, 255, 45)),
    }

    def __init__(self, gate_id: str, color: str, min_area_ratio: float = 0.02, sample_interval_ms: int = 200) -> None:
        super().__init__(gate_id, [EventType.VIDEO_FRAME])
        self.queue_size = 1
        self.color = color.lower()
        self.min_area_ratio = min_area_ratio
        self.sample_interval_ms = sample_interval_ms
        self._last_sample_time_ms = -sample_interval_ms

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        if event.stream_time_ms - self._last_sample_time_ms < self.sample_interval_ms:
            return None
        frame = event.payload.get("frame")
        if frame is None:
            return None
        self._last_sample_time_ms = event.stream_time_ms
        ratio = await asyncio.to_thread(self._color_ratio, frame)
        if ratio < self.min_area_ratio:
            return None
        return GateFire(
            gate_id=self.gate_id,
            confidence=min(1.0, ratio / max(self.min_area_ratio, 0.001)),
            reason=f"{self.color} area ratio {ratio:.3f} exceeded threshold.",
            evidence={"color": self.color, "area_ratio": ratio, "stream_time_ms": event.stream_time_ms, "frame": frame},
        )

    def _color_ratio(self, frame: object) -> float:
        import cv2  # type: ignore

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = self.COLOR_RANGES.get(self.color)
        if ranges is None:
            return 0.0
        if len(ranges) == 4:
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv, lower, upper)
        return float((mask > 0).mean())


class ObjectSpatialRelationGate(Gate):
    """Cheap relation gate over upstream detector bboxes."""

    def __init__(
        self,
        gate_id: str,
        upstream_gate_id: str,
        subject: str,
        object_: str,
        relation: str = "near",
        max_distance_ratio: float = 0.25,
    ) -> None:
        super().__init__(gate_id, [EventType.GATE_FIRE])
        self.upstream_gate_id = upstream_gate_id
        self.subject = subject.lower()
        self.object = object_.lower()
        self.relation = relation
        self.max_distance_ratio = max_distance_ratio

    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        fire = event.payload.get("gate_fire")
        if not isinstance(fire, GateFire) or fire.gate_id != self.upstream_gate_id:
            return None
        detections = fire.evidence.get("detections", [])
        subjects = [item for item in detections if self._label_matches(item, self.subject) and item.get("bbox")]
        objects = [item for item in detections if self._label_matches(item, self.object) and item.get("bbox")]
        for subject in subjects:
            for obj in objects:
                confidence = self._relation_confidence(subject["bbox"], obj["bbox"], self.relation)
                if confidence > 0:
                    return GateFire(
                        gate_id=self.gate_id,
                        confidence=confidence,
                        reason=f"{self.subject} is {self.relation} {self.object}.",
                        evidence={
                            "subject": subject,
                            "object": obj,
                            "detections": [subject, obj],
                            "relation": self.relation,
                            "stream_time_ms": fire.evidence.get("stream_time_ms", event.stream_time_ms),
                            "frame": fire.evidence.get("frame"),
                        },
                    )
        return None

    def _label_matches(self, item: object, label: str) -> bool:
        if not isinstance(item, dict):
            return False
        return label_matches(str(item.get("label", "")), label)

    def _relation_confidence(self, first: Sequence[float], second: Sequence[float], relation: str) -> float:
        if relation in {"overlap", "touching"}:
            return self._iou(first, second)
        first_center = self._center(first)
        second_center = self._center(second)
        if relation == "left_of" and first_center[0] < second_center[0]:
            return 0.8
        if relation == "right_of" and first_center[0] > second_center[0]:
            return 0.8
        if relation == "above" and first_center[1] < second_center[1]:
            return 0.8
        if relation == "below" and first_center[1] > second_center[1]:
            return 0.8
        if relation in {"near", "close_to"}:
            distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
            scale = max(first[2] - first[0], first[3] - first[1], second[2] - second[0], second[3] - second[1], 1.0)
            return 0.9 if distance / scale <= self.max_distance_ratio * 10 else 0.0
        return 0.0

    def _center(self, bbox: Sequence[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _iou(self, first: Sequence[float], second: Sequence[float]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0
