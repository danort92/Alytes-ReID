"""Prepare and merge datasets into YOLO format for training.

Usage:
    python -m src.data.prepare_dataset --config config/detection.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_yolo_dataset_yaml(
    dataset_dir: Path,
    classes: list[str],
) -> Path:
    """Create YOLO dataset.yaml file.

    Args:
        dataset_dir: Root directory of the dataset.
        classes: List of class names.

    Returns:
        Path to the created YAML file.
    """
    yaml_content = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info("Created dataset YAML at %s", yaml_path)
    return yaml_path


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """Split dataset into train/val/test sets.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO label files.
        output_dir: Output directory for split dataset.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with split counts.
    """
    random.seed(seed)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy2(img_path, img_out / img_path.name)
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_path.name)

    counts = {k: len(v) for k, v in splits.items()}
    logger.info("Split dataset: %s", counts)
    return counts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for YOLO training"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/detection.yaml"),
        help="Detection config file",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing YOLO label files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)

    output_dir = Path(config["data"]["dataset_dir"])
    split_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=output_dir,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
    )
    create_yolo_dataset_yaml(
        dataset_dir=output_dir,
        classes=config["data"]["classes"],
    )


if __name__ == "__main__":
    main()
