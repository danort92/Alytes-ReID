"""Color and histogram normalization for toad images."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_colors(
    image: np.ndarray,
    method: str = "clahe",
    clahe_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Normalize image colors to handle lighting variation.

    Args:
        image: Input image (H, W, 3) in RGB.
        method: Normalization method ("clahe", "histogram_eq", "none").
        clahe_config: CLAHE parameters (clip_limit, tile_grid_size).

    Returns:
        Normalized image.
    """
    if method == "none":
        return image
    elif method == "histogram_eq":
        return _histogram_equalization(image)
    elif method == "clahe":
        config = clahe_config or {"clip_limit": 2.0, "tile_grid_size": [8, 8]}
        return _clahe_normalization(image, config)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization on the L channel (LAB color space).

    Args:
        image: Input image (H, W, 3) in RGB.

    Returns:
        Equalized image in RGB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _clahe_normalization(
    image: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image (H, W, 3) in RGB.
        config: CLAHE parameters.

    Returns:
        Normalized image in RGB.
    """
    clip_limit = config.get("clip_limit", 2.0)
    tile_size = tuple(config.get("tile_grid_size", [8, 8]))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
