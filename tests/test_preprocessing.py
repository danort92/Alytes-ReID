"""Tests for preprocessing module."""

import numpy as np

from src.preprocessing.normalize import normalize_colors
from src.preprocessing.pipeline import resize_with_padding


def test_resize_with_padding_maintains_shape() -> None:
    """Output should match target size."""
    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    result = resize_with_padding(image, (256, 256))
    assert result.shape == (256, 256, 3)


def test_resize_with_padding_square_input() -> None:
    """Square input should fill the target without padding."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = resize_with_padding(image, (256, 256))
    assert result.shape == (256, 256, 3)


def test_normalize_colors_none() -> None:
    """Method 'none' should return image unchanged."""
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result = normalize_colors(image, method="none")
    np.testing.assert_array_equal(result, image)


def test_normalize_colors_clahe() -> None:
    """CLAHE should return image with same shape."""
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result = normalize_colors(image, method="clahe")
    assert result.shape == image.shape
