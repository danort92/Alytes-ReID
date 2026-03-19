"""Download frog detection dataset from Roboflow.

Usage:
    python -m src.data.download_roboflow --output data/raw/roboflow --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROBOFLOW_DATASET_URL = "https://universe.roboflow.com/brad-dwyer/frogs-agkpn"


def download_roboflow_dataset(
    output_dir: Path,
    api_key: str,
    dataset_format: str = "yolov8",
) -> Path:
    """Download the Roboflow frogs dataset.

    Args:
        output_dir: Directory to save the dataset.
        api_key: Roboflow API key.
        dataset_format: Format to download ("yolov8", "coco", "voc").

    Returns:
        Path to the downloaded dataset directory.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "roboflow package required. Install with: pip install roboflow"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("brad-dwyer").project("frogs-agkpn")
    version = project.version(1)
    dataset = version.download(dataset_format, location=str(output_dir))

    logger.info("Dataset downloaded to %s", dataset.location)
    return Path(dataset.location)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Roboflow frogs dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/roboflow"),
        help="Output directory",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Roboflow API key",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov8",
        choices=["yolov8", "coco", "voc"],
        help="Dataset format",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_roboflow_dataset(
        output_dir=args.output,
        api_key=args.api_key,
        dataset_format=args.format,
    )


if __name__ == "__main__":
    main()
