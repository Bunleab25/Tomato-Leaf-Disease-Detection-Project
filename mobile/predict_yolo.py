from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WEIGHTS = ROOT_DIR / "runs" / "detect" / "runs" / "leaf_disease_detect2" / "weights" / "best.pt"


@dataclass
class Detection:
    bbox_xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str


def load_yolo_model(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))


def detect_leaves(
    image: Image.Image,
    model: YOLO,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> list[Detection]:
    results = model.predict(source=image.convert("RGB"), conf=conf_threshold, imgsz=imgsz, verbose=False)
    detections: list[Detection] = []

    for result in results:
        names = result.names or {}
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, str(cls_id)))
            bbox_xyxy = [float(v) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            detections.append(
                Detection(
                    bbox_xyxy=bbox_xyxy,
                    confidence=confidence,
                    class_id=cls_id,
                    class_name=class_name,
                )
            )

    detections.sort(key=lambda det: det.confidence, reverse=True)
    return detections


def crop_from_bbox(image: Image.Image, bbox_xyxy: list[float]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox_xyxy

    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))

    return image.crop((left, top, right, bottom))


def detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "bbox": [round(v, 2) for v in det.bbox_xyxy],
        "confidence": det.confidence,
        "class_id": det.class_id,
        "class_name": det.class_name,
    }


def bbox_area_xyxy(bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_largest_detection(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return max(detections, key=lambda det: bbox_area_xyxy(det.bbox_xyxy))
