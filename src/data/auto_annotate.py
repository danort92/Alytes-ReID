"""Auto-annotate images using a COCO-pretrained YOLO11 model.

YOLO11 pretrained on COCO includes class 8 = "frog". We run inference on
downloaded images, keep detections for class 8, and save YOLO-format label
files (remapped to class 0 = "toad" for our single-class detector).

Usage:
    python -m src.data.auto_annotate --images data/raw/inaturalist --output data/raw/auto_labels
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# COCO class 8 = "frog" (covers toads too — COCO has no separate toad class)
COCO_FROG_CLASS = 8

# Remapped class for our single-class detector
TARGET_CLASS = 0


def auto_annotate(
    images_dir: Path,
    output_dir: Path,
    model_name: str = "yolo11s",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
) -> dict[str, int]:
    """Run COCO-pretrained YOLO11 on images and save frog detections as labels.

    Args:
        images_dir: Directory containing images to annotate.
        output_dir: Directory to save YOLO-format .txt label files.
        model_name: YOLO model to use (weights auto-download on first run).
        confidence: Minimum detection confidence.
        iou: NMS IoU threshold.
        device: Device for inference (None = auto).

    Returns:
        Dictionary with annotation statistics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package required. Install with: pip install ultralytics"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(f"{model_name}.pt")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        logger.warning("No images found in %s", images_dir)
        return {"total_images": 0, "annotated": 0, "skipped": 0, "total_boxes": 0}

    logger.info("Running auto-annotation on %d images...", len(image_paths))

    annotated = 0
    skipped = 0
    total_boxes = 0

    # Process in batches for GPU efficiency
    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=confidence,
            iou=iou,
            device=device,
            classes=[COCO_FROG_CLASS],
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                skipped += 1
                continue

            # Write YOLO-format label file
            label_path = output_dir / f"{img_path.stem}.txt"
            lines = []
            for box in boxes:
                # box.xywhn = normalized [x_center, y_center, width, height]
                xywhn = box.xywhn[0].tolist()
                lines.append(
                    f"{TARGET_CLASS} {xywhn[0]:.6f} {xywhn[1]:.6f} "
                    f"{xywhn[2]:.6f} {xywhn[3]:.6f}"
                )

            label_path.write_text("\n".join(lines) + "\n")
            annotated += 1
            total_boxes += len(lines)

    stats = {
        "total_images": len(image_paths),
        "annotated": annotated,
        "skipped": skipped,
        "total_boxes": total_boxes,
    }
    logger.info(
        "Auto-annotation complete: %d/%d images annotated (%d boxes total, %d skipped)",
        annotated,
        len(image_paths),
        total_boxes,
        skipped,
    )
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using COCO-pretrained YOLO11"
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/auto_labels"),
        help="Output directory for label files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s",
        help="YOLO model name (default: yolo11s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = auto_annotate(
        images_dir=args.images,
        output_dir=args.output,
        model_name=args.model,
        confidence=args.confidence,
    )
    print(f"\nResults: {stats}")


if __name__ == "__main__":
    main()
