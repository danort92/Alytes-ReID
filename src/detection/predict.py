"""Run toad detection on new images.

Usage:
    python -m src.detection.predict --model data/models/detection/best.pt --input image.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_detector(model_path: Path) -> YOLO:
    """Load a trained YOLOv8 model.

    Args:
        model_path: Path to model weights (.pt file).

    Returns:
        Loaded YOLO model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def detect_toads(
    model: YOLO,
    image_path: Path,
    confidence: float = 0.5,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Detect toads in an image.

    Args:
        model: Loaded YOLO model.
        image_path: Path to input image.
        confidence: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.

    Returns:
        List of detections, each with keys: "bbox", "confidence", "class".
        bbox format is [x1, y1, x2, y2] in pixel coordinates.
    """
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        iou=iou_threshold,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "class": cls,
            })

    logger.info("Found %d toad(s) in %s", len(detections), image_path.name)
    return detections


def detect_toads_batch(
    model: YOLO,
    image_paths: list[Path],
    confidence: float = 0.5,
    iou_threshold: float = 0.45,
) -> dict[str, list[dict[str, Any]]]:
    """Detect toads in multiple images.

    Args:
        model: Loaded YOLO model.
        image_paths: List of image paths.
        confidence: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Dictionary mapping image filenames to detection lists.
    """
    all_detections: dict[str, list[dict[str, Any]]] = {}
    for path in image_paths:
        all_detections[path.name] = detect_toads(
            model, path, confidence, iou_threshold
        )
    return all_detections


def get_best_detection(
    detections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Get the highest-confidence detection.

    Args:
        detections: List of detections.

    Returns:
        Detection with highest confidence, or None if no detections.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d["confidence"])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run toad detection")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image or directory",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = load_detector(args.model)

    if args.input.is_dir():
        image_paths = list(args.input.glob("*.jpg")) + list(args.input.glob("*.png"))
        results = detect_toads_batch(model, image_paths, args.confidence)
        for name, dets in results.items():
            print(f"{name}: {len(dets)} detection(s)")
    else:
        detections = detect_toads(model, args.input, args.confidence)
        for det in detections:
            print(f"  bbox={det['bbox']}, conf={det['confidence']:.3f}")


if __name__ == "__main__":
    main()
