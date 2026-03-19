"""Download Alytes obstetricans images from iNaturalist API.

Usage:
    python -m src.data.download_inat --output data/raw/inaturalist --max-images 1000
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

INAT_API_BASE = "https://api.inaturalist.org/v1"
ALYTES_TAXON_ID = 22080  # Alytes obstetricans


def search_observations(
    taxon_id: int = ALYTES_TAXON_ID,
    quality_grade: str = "research",
    per_page: int = 200,
    page: int = 1,
) -> dict[str, Any]:
    """Search iNaturalist for observations of a given taxon.

    Args:
        taxon_id: iNaturalist taxon ID.
        quality_grade: Filter by quality grade ("research", "needs_id", "casual").
        per_page: Number of results per page (max 200).
        page: Page number.

    Returns:
        JSON response as dictionary.
    """
    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality_grade,
        "photos": "true",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
    }
    response = requests.get(
        f"{INAT_API_BASE}/observations",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def download_image(url: str, output_path: Path) -> bool:
    """Download a single image from URL.

    Args:
        url: Image URL.
        output_path: Local path to save the image.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return True
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


def download_alytes_images(
    output_dir: Path,
    max_images: int = 1000,
    taxon_id: int = ALYTES_TAXON_ID,
) -> list[Path]:
    """Download Alytes images from iNaturalist.

    Args:
        output_dir: Directory to save downloaded images.
        max_images: Maximum number of images to download.
        taxon_id: iNaturalist taxon ID.

    Returns:
        List of paths to downloaded images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    page = 1

    with tqdm(total=max_images, desc="Downloading images") as pbar:
        while len(downloaded) < max_images:
            data = search_observations(
                taxon_id=taxon_id,
                per_page=200,
                page=page,
            )
            results = data.get("results", [])
            if not results:
                logger.info("No more results at page %d", page)
                break

            for obs in results:
                if len(downloaded) >= max_images:
                    break
                for photo in obs.get("photos", []):
                    if len(downloaded) >= max_images:
                        break
                    # Get medium-sized image URL
                    url = photo.get("url", "")
                    if not url:
                        continue
                    # Replace square thumbnail with medium size
                    url = url.replace("square", "medium")

                    obs_id = obs.get("id", "unknown")
                    photo_id = photo.get("id", "unknown")
                    filename = f"inat_{obs_id}_{photo_id}.jpg"
                    filepath = output_dir / filename

                    if filepath.exists():
                        downloaded.append(filepath)
                        pbar.update(1)
                        continue

                    if download_image(url, filepath):
                        downloaded.append(filepath)
                        pbar.update(1)

            page += 1
            # Respect API rate limits
            time.sleep(1.0)

    logger.info("Downloaded %d images to %s", len(downloaded), output_dir)
    return downloaded


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Alytes images from iNaturalist"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/inaturalist"),
        help="Output directory for downloaded images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of images to download",
    )
    parser.add_argument(
        "--taxon-id",
        type=int,
        default=ALYTES_TAXON_ID,
        help="iNaturalist taxon ID",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_alytes_images(
        output_dir=args.output,
        max_images=args.max_images,
        taxon_id=args.taxon_id,
    )


if __name__ == "__main__":
    main()
