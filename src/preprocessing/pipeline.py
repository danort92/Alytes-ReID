"""Full preprocessing pipeline for standardizing segmented toad images.

Usage:
    python -m src.preprocessing.pipeline --config config/preprocessing.yaml --input cropped.png --mask mask.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.preprocessing.align import align_toad
from src.preprocessing.normalize import normalize_colors
from src.segmentation.utils import apply_mask, get_mask_bbox

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask tightly around the mask region.

    Args:
        image: Input image (H, W, 3).
        mask: Binary mask (H, W).
        padding_ratio: Padding as fraction of bbox dimensions.

    Returns:
        Tuple of (cropped_image, cropped_mask).
    """
    x1, y1, x2, y2 = get_mask_bbox(mask)
    h_pad = int((y2 - y1) * padding_ratio)
    w_pad = int((x2 - x1) * padding_ratio)

    h, w = image.shape[:2]
    y1 = max(0, y1 - h_pad)
    y2 = min(h, y2 + h_pad)
    x1 = max(0, x1 - w_pad)
    x2 = min(w, x2 + w_pad)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def resize_with_padding(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_color: tuple[int, int, int] = (0, 0, 0),
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Resize image maintaining aspect ratio, padding to target size.

    Args:
        image: Input image (H, W, 3).
        target_size: Target (height, width).
        pad_color: Padding color.
        interpolation: Interpolation method.

    Returns:
        Resized and padded image.
    """
    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }
    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    result = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return result


def preprocess_toad(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Run full preprocessing pipeline on a segmented toad image.

    Args:
        image: Cropped toad image (H, W, 3) in RGB.
        mask: Binary mask (H, W).
        config: Preprocessing configuration.

    Returns:
        Standardized toad image ready for re-ID.
    """
    # 1. Tight crop
    cropped, cropped_mask = crop_to_mask(
        image, mask, config["cropping"]["padding_ratio"]
    )

    # 2. Background removal
    bg_color = tuple(config["background"]["fill_color"])
    masked = apply_mask(cropped, cropped_mask, background_color=bg_color)

    # 3. Pose alignment
    aligned = align_toad(
        masked,
        cropped_mask,
        method=config["alignment"]["method"],
        orient_head_up=config["alignment"]["orient_head_up"],
    )

    # 4. Color normalization
    normalized = normalize_colors(
        aligned,
        method=config["color_normalization"]["method"],
        clahe_config=config["color_normalization"].get("clahe"),
    )

    # 5. Resize
    target_size = tuple(config["resize"]["target_size"])
    if config["resize"]["maintain_aspect_ratio"]:
        result = resize_with_padding(
            normalized,
            target_size,
            pad_color=tuple(config["resize"]["pad_color"]),
            interpolation=config["resize"]["interpolation"],
        )
    else:
        result = cv2.resize(normalized, (target_size[1], target_size[0]))

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Preprocess toad image")
    parser.add_argument("--config", type=Path, default=Path("config/preprocessing.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Input image")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask")
    parser.add_argument("--output", type=Path, default=Path("output_preprocessed.png"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)

    image = cv2.cvtColor(cv2.imread(str(args.input)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)

    result = preprocess_toad(image, mask, config)
    cv2.imwrite(str(args.output), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    logger.info("Preprocessed image saved to %s", args.output)


if __name__ == "__main__":
    main()
