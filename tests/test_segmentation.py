"""Tests for segmentation utilities."""

import numpy as np

from src.segmentation.utils import apply_mask, clean_mask, get_mask_bbox, mask_area


def test_apply_mask() -> None:
    """Background should be replaced with fill color."""
    image = np.ones((10, 10, 3), dtype=np.uint8) * 200
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255

    result = apply_mask(image, mask, background_color=(0, 0, 0))
    assert result[0, 0, 0] == 0  # background pixel
    assert result[5, 5, 0] == 200  # foreground pixel


def test_get_mask_bbox() -> None:
    """Bounding box should tightly enclose the mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:80] = 255
    x1, y1, x2, y2 = get_mask_bbox(mask)
    assert (x1, y1, x2, y2) == (30, 20, 79, 59)


def test_mask_area() -> None:
    """Area should equal the number of non-zero pixels."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    assert mask_area(mask) == 100
