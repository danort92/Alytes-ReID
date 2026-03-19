"""Training script for toad re-ID with metric learning.

Usage:
    python -m src.reid.train --config config/reid.yaml --data-dir data/processed/reid
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.reid.model import build_model

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_loss_function(config: dict[str, Any]) -> nn.Module:
    """Create loss function based on configuration.

    Args:
        config: Loss configuration section.

    Returns:
        Loss module.
    """
    loss_type = config["loss"]["type"]

    if loss_type == "arcface":
        from pytorch_metric_learning.losses import ArcFaceLoss

        arc_cfg = config["loss"]["arcface"]
        # Note: num_classes must be set based on training data
        # This is a placeholder — actual instantiation happens in train loop
        logger.info("Using ArcFace loss (margin=%.1f, scale=%d)",
                     arc_cfg["margin"], arc_cfg["scale"])
        return None  # Created in train loop with num_classes

    elif loss_type == "triplet":
        from pytorch_metric_learning.losses import TripletMarginLoss

        return TripletMarginLoss(margin=config["loss"]["triplet"]["margin"])

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_reid(
    config: dict[str, Any],
    data_dir: Path,
) -> Path:
    """Train re-ID model.

    Args:
        config: Full configuration dictionary.
        data_dir: Directory containing training data organized by individual ID.

    Returns:
        Path to saved model weights.
    """
    device = config["training"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(config).to(device)
    train_cfg = config["training"]

    optimizer_cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }[train_cfg["optimizer"]]

    optimizer = optimizer_cls(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # TODO: implement full training loop with:
    # - Dataset loading from data_dir (folder per individual)
    # - ArcFace loss with correct num_classes
    # - Validation and early stopping
    # - MLflow experiment tracking
    # - Model checkpointing

    logger.info("Training setup complete. Full training loop to be implemented.")

    output_dir = Path(config["export"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "reid_model.pt"

    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)
    return model_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train toad re-ID model")
    parser.add_argument(
        "--config", type=Path, default=Path("config/reid.yaml"),
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Training data directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    train_reid(config, args.data_dir)


if __name__ == "__main__":
    main()
