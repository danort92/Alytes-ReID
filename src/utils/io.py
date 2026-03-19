"""File I/O helper functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save data to a YAML file.

    Args:
        data: Dictionary to save.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_image(path: Path, rgb: bool = True) -> np.ndarray:
    """Load an image from file.

    Args:
        path: Path to image file.
        rgb: If True, convert to RGB. Otherwise keep BGR.

    Returns:
        Image as numpy array.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image: np.ndarray, path: Path, rgb: bool = True) -> None:
    """Save an image to file.

    Args:
        image: Image array.
        path: Output path.
        rgb: If True, convert from RGB to BGR before saving.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def export_results_csv(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Export matching results to CSV for mark-recapture analysis.

    Args:
        results: List of result dictionaries.
        output_path: Path for the CSV file.
    """
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info("Results exported to %s", output_path)
