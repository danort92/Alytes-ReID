"""YOLOv8 fine-tuning for toad detection.

Usage:
    python -m src.detection.train --config config/detection.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_detector(config: dict[str, Any]) -> Path:
    """Train YOLOv8 model for toad detection.

    Args:
        config: Training configuration dictionary.

    Returns:
        Path to the best model weights.
    """
    model_name = config["model"]["architecture"]
    model = YOLO(f"{model_name}.pt")

    train_cfg = config["training"]
    dataset_yaml = Path(config["data"]["dataset_dir"]) / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}. "
            "Run src.data.prepare_dataset first."
        )

    results = model.train(
        data=str(dataset_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["image_size"],
        lr0=train_cfg["learning_rate"],
        optimizer=train_cfg["optimizer"],
        patience=train_cfg["patience"],
        augment=train_cfg["augmentation"],
        device=train_cfg["device"] if train_cfg["device"] != "auto" else None,
        project=config["export"]["model_dir"],
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_weights)
    return best_weights


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 toad detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/detection.yaml"),
        help="Detection config file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    train_detector(config)


if __name__ == "__main__":
    main()
