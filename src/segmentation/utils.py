"""Mask utility functions for segmentation post-processing."""

from __future__ import annotations

import cv2
import numpy as np


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply binary mask to image, replacing background.

    Args:
        image: Input image (H, W, 3).
        mask: Binary mask (H, W) with values 0 or 255.
        background_color: RGB color for background pixels.

    Returns:
        Masked image with background replaced.
    """
    result = np.full_like(image, background_color)
    mask_bool = mask > 127
    result[mask_bool] = image[mask_bool]
    return result


def get_mask_bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int]:
    """Get bounding box of non-zero mask region.

    Args:
        mask: Binary mask (H, W).
        padding: Padding in pixels around the bounding box.

    Returns:
        Bounding box (x1, y1, x2, y2).
    """
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    h, w = mask.shape[:2]
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w, x_max + padding)
    y2 = min(h, y_max + padding)

    return (x1, y1, x2, y2)


def mask_area(mask: np.ndarray) -> int:
    """Calculate area of non-zero mask region in pixels.

    Args:
        mask: Binary mask (H, W).

    Returns:
        Number of non-zero pixels.
    """
    return int(np.count_nonzero(mask > 127))


def clean_mask(
    mask: np.ndarray,
    min_area: int = 100,
    kernel_size: int = 5,
) -> np.ndarray:
    """Clean binary mask by removing small components and smoothing edges.

    Args:
        mask: Binary mask (H, W) with values 0 or 255.
        min_area: Minimum contour area to keep.
        kernel_size: Morphological operation kernel size.

    Returns:
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    # Close small holes
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open to remove noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Remove small connected components
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(result, [contour], -1, 255, -1)

    return result
