"""SAM2-based toad segmentation using YOLO detection bounding boxes.

Usage:
    python -m src.segmentation.segment --image image.jpg --detector data/models/detection/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ToadSegmenter:
    """Segment toads using SAM2 prompted with bounding boxes from YOLO."""

    def __init__(
        self,
        sam2_checkpoint: Path | str = "facebook/sam2-hiera-large",
        device: str = "auto",
    ) -> None:
        """Initialize SAM2 segmenter.

        Args:
            sam2_checkpoint: Path or HuggingFace model ID for SAM2.
            device: Device to run on ("auto", "cpu", "cuda").
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.checkpoint = sam2_checkpoint
        logger.info("ToadSegmenter initialized (device=%s)", self.device)

    def load_model(self) -> None:
        """Load SAM2 model. Called lazily on first use."""
        if self.model is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.model = SAM2ImagePredictor(
                build_sam2(
                    config_file="sam2_hiera_l.yaml",
                    ckpt_path=str(self.checkpoint),
                    device=self.device,
                )
            )
        except ImportError:
            logger.warning(
                "sam2 not installed. Install with: pip install segment-anything-2"
            )
            raise

        logger.info("SAM2 model loaded")

    def segment_from_bbox(
        self,
        image: np.ndarray,
        bbox: list[float],
    ) -> np.ndarray:
        """Generate segmentation mask from a bounding box prompt.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB.
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            Binary mask as numpy array (H, W) with values 0 or 255.
        """
        self.load_model()

        self.model.set_image(image)
        input_box = np.array(bbox)

        masks, scores, _ = self.model.predict(
            box=input_box[None, :],
            multimask_output=True,
        )

        # Select the mask with highest confidence
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # Convert to uint8 binary mask
        binary_mask = (mask * 255).astype(np.uint8)
        return binary_mask

    def segment_and_crop(
        self,
        image: np.ndarray,
        bbox: list[float],
        padding: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment toad and return cropped image + mask.

        Args:
            image: Input image (H, W, 3) in RGB.
            bbox: Bounding box [x1, y1, x2, y2].
            padding: Padding ratio around the mask bounding box.

        Returns:
            Tuple of (cropped_image, cropped_mask).
        """
        mask = self.segment_from_bbox(image, bbox)

        # Find mask bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            logger.warning("Empty mask, falling back to detection bbox")
            x1, y1, x2, y2 = [int(c) for c in bbox]
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Add padding
            h, w = image.shape[:2]
            pad_y = int((y_max - y_min) * padding)
            pad_x = int((x_max - x_min) * padding)
            y1 = max(0, y_min - pad_y)
            y2 = min(h, y_max + pad_y)
            x1 = max(0, x_min - pad_x)
            x2 = min(w, x_max + pad_x)

        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        return cropped_image, cropped_mask


def segment_toad_image(
    image_path: Path,
    bbox: list[float],
    sam2_checkpoint: str = "facebook/sam2-hiera-large",
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function: segment a toad from an image file.

    Args:
        image_path: Path to the input image.
        bbox: Detection bounding box [x1, y1, x2, y2].
        sam2_checkpoint: SAM2 model checkpoint.
        device: Device to use.

    Returns:
        Tuple of (cropped_image, binary_mask).
    """
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segmenter = ToadSegmenter(sam2_checkpoint=sam2_checkpoint, device=device)
    return segmenter.segment_and_crop(image, bbox)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Segment toad from image")
    parser.add_argument("--image", type=Path, required=True, help="Input image")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        help="Bounding box: x1 y1 x2 y2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_mask.png"),
        help="Output mask path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cropped, mask = segment_toad_image(args.image, args.bbox)
    cv2.imwrite(str(args.output), mask)
    logger.info("Mask saved to %s", args.output)


if __name__ == "__main__":
    main()
