"""Evaluate toad detection model performance.

Usage:
    python -m src.detection.evaluate --model data/models/detection/best.pt --config config/detection.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def evaluate_model(
    model_path: Path,
    config: dict[str, Any],
) -> dict[str, float]:
    """Evaluate detection model on test set using YOLO's built-in val.

    Args:
        model_path: Path to trained model weights.
        config: Detection configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    model = YOLO(str(model_path))
    dataset_yaml = Path(config["data"]["dataset_dir"]) / "dataset.yaml"

    results = model.val(
        data=str(dataset_yaml),
        split="test",
        verbose=True,
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate toad detector")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/detection.yaml"),
        help="Detection config file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    metrics = evaluate_model(args.model, config)

    print("\n=== Detection Evaluation Results ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
