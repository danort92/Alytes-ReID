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
ALYTES_GENUS_ID = 22076  # Genus Alytes (includes all midwife toad species)


def search_observations(
    taxon_id: int = ALYTES_TAXON_ID,
    quality_grade: str = "research",
    per_page: int = 200,
    page: int = 1,
    license: str | None = None,
) -> dict[str, Any]:
    """Search iNaturalist for observations of a given taxon.

    Args:
        taxon_id: iNaturalist taxon ID.
        quality_grade: Filter by quality grade (comma-separated, e.g. "research,needs_id").
        per_page: Number of results per page (max 200).
        page: Page number.
        license: Photo license filter (e.g. "cc-by,cc-by-nc,cc0").

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
    if license:
        params["photo_license"] = license
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


def _collect_from_query(
    taxon_id: int,
    quality_grade: str,
    max_images: int,
    output_dir: Path,
    downloaded: list[Path],
    seen_photo_ids: set[str],
    pbar: tqdm,
) -> None:
    """Run a paginated query and download new images."""
    page = 1
    while len(downloaded) < max_images:
        data = search_observations(
            taxon_id=taxon_id,
            quality_grade=quality_grade,
            per_page=200,
            page=page,
        )
        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            if len(downloaded) >= max_images:
                break
            for photo in obs.get("photos", []):
                if len(downloaded) >= max_images:
                    break

                photo_id = str(photo.get("id", "unknown"))
                if photo_id in seen_photo_ids:
                    continue
                seen_photo_ids.add(photo_id)

                url = photo.get("url", "")
                if not url:
                    continue
                # Replace square thumbnail with medium size
                url = url.replace("square", "medium")

                obs_id = obs.get("id", "unknown")
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
        time.sleep(1.0)


def download_alytes_images(
    output_dir: Path,
    max_images: int = 1000,
    taxon_id: int = ALYTES_TAXON_ID,
    include_genus: bool = True,
) -> list[Path]:
    """Download Alytes images from iNaturalist.

    Searches in order:
    1. Research-grade observations of the exact species
    2. Needs-ID observations of the exact species
    3. Research-grade observations of the genus Alytes (if include_genus=True)

    Args:
        output_dir: Directory to save downloaded images.
        max_images: Maximum number of images to download.
        taxon_id: iNaturalist taxon ID for the species.
        include_genus: Also search genus-level observations.

    Returns:
        List of paths to downloaded images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    seen_photo_ids: set[str] = set()

    # Build search queries in priority order
    queries = [
        (taxon_id, "research"),
        (taxon_id, "needs_id"),
    ]
    if include_genus:
        queries.append((ALYTES_GENUS_ID, "research"))

    with tqdm(total=max_images, desc="Downloading images") as pbar:
        for q_taxon, q_grade in queries:
            if len(downloaded) >= max_images:
                break
            logger.info(
                "Searching taxon=%d quality=%s (%d/%d so far)",
                q_taxon, q_grade, len(downloaded), max_images,
            )
            _collect_from_query(
                taxon_id=q_taxon,
                quality_grade=q_grade,
                max_images=max_images,
                output_dir=output_dir,
                downloaded=downloaded,
                seen_photo_ids=seen_photo_ids,
                pbar=pbar,
            )

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
