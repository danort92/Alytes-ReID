"""Auto-annotate images using YOLO-World (open-vocabulary detector).

YOLO-World detects objects by text description, not fixed COCO classes.
We use prompts like "toad", "frog" to find amphibians in any photo —
much more robust than COCO class 8 which misses camouflaged toads.

Usage:
    python -m src.data.auto_annotate --images data/raw/inaturalist --output data/raw/auto_labels
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Text prompts for open-vocabulary detection
DEFAULT_CLASSES = ["toad", "frog"]

# Remapped class for our single-class detector
TARGET_CLASS = 0


def auto_annotate(
    images_dir: Path,
    output_dir: Path,
    model_name: str = "yolov8s-worldv2",
    confidence: float = 0.10,
    iou: float = 0.45,
    device: str | None = None,
    text_classes: list[str] | None = None,
) -> dict[str, int]:
    """Run YOLO-World on images and save detections as YOLO-format labels.

    YOLO-World is an open-vocabulary detector: it finds objects matching
    text descriptions rather than a fixed set of COCO classes. This works
    much better for wildlife photos where COCO classes fail.

    Args:
        images_dir: Directory containing images to annotate.
        output_dir: Directory to save YOLO-format .txt label files.
        model_name: YOLO-World model to use (auto-downloads on first run).
        confidence: Minimum detection confidence.
        iou: NMS IoU threshold.
        device: Device for inference (None = auto).
        text_classes: Text prompts for detection (default: ["toad", "frog"]).

    Returns:
        Dictionary with annotation statistics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package required. Install with: pip install ultralytics"
        )

    if text_classes is None:
        text_classes = DEFAULT_CLASSES

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO-World and set text prompts
    model = YOLO(f"{model_name}.pt")
    model.set_classes(text_classes)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        logger.warning("No images found in %s", images_dir)
        return {"total_images": 0, "annotated": 0, "skipped": 0, "total_boxes": 0}

    logger.info(
        "Running YOLO-World auto-annotation on %d images (classes: %s)...",
        len(image_paths),
        text_classes,
    )

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
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                skipped += 1
                continue

            # Write YOLO-format label file
            # All detections (toad/frog) map to class 0 = "toad"
            label_path = output_dir / f"{img_path.stem}.txt"
            lines = []
            for box in boxes:
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
        description="Auto-annotate images using YOLO-World (open-vocabulary)"
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
        default="yolov8s-worldv2",
        help="YOLO-World model name (default: yolov8s-worldv2)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help="Minimum detection confidence (default: 0.10)",
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
