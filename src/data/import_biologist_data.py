"""Import and deduplicate biologist's photo archives.

Handles ZIP files containing mixed JPG and CR3 (Canon RAW) images.
When both formats exist for the same filename stem, JPG is preferred.

Usage:
    python -m src.data.import_biologist_data archive.zip data/raw/biologist
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

JPG_EXTENSIONS = {".jpg", ".jpeg"}
RAW_EXTENSIONS = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".raw", ".raf"}
ALL_IMAGE_EXTENSIONS = JPG_EXTENSIONS | RAW_EXTENSIONS | {".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images_in_zip(zip_path: Path) -> list[str]:
    """List image file entries inside a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [
            name for name in zf.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in ALL_IMAGE_EXTENSIONS
        ]


def deduplicate_images(file_names: list[str]) -> tuple[list[str], list[str]]:
    """Select images to keep, preferring JPG over RAW when both exist.

    Args:
        file_names: List of file paths/names from the archive.

    Returns:
        Tuple of (kept, skipped) file name lists.
    """
    by_stem: dict[str, list[str]] = {}
    for name in file_names:
        stem = Path(name).stem.upper()
        by_stem.setdefault(stem, []).append(name)

    kept: list[str] = []
    skipped: list[str] = []

    for stem, variants in sorted(by_stem.items()):
        if len(variants) == 1:
            kept.append(variants[0])
            continue

        jpg_files = [f for f in variants if Path(f).suffix.lower() in JPG_EXTENSIONS]
        if jpg_files:
            kept.append(jpg_files[0])
            skipped.extend(f for f in variants if f != jpg_files[0])
        else:
            kept.append(variants[0])
            skipped.extend(variants[1:])

    return kept, skipped


def import_from_zip(
    zip_path: Path,
    output_dir: Path,
    prefer_jpg: bool = True,
) -> dict[str, int | list[str]]:
    """Extract and deduplicate images from a ZIP archive.

    Args:
        zip_path: Path to the ZIP file.
        output_dir: Directory to write deduplicated images.
        prefer_jpg: When True, keep JPG over RAW for duplicate stems.

    Returns:
        Statistics dictionary with counts and file lists.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    all_entries = list_images_in_zip(zip_path)
    logger.info("Found %d image files in archive", len(all_entries))

    if prefer_jpg:
        kept, skipped = deduplicate_images(all_entries)
    else:
        kept = all_entries
        skipped = []

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in kept:
            data = zf.read(entry)
            out_name = Path(entry).name
            out_path = output_dir / out_name
            out_path.write_bytes(data)
            extracted.append(out_name)

    jpg_count = sum(1 for f in extracted if Path(f).suffix.lower() in JPG_EXTENSIONS)
    raw_count = sum(1 for f in extracted if Path(f).suffix.lower() in RAW_EXTENSIONS)
    other_count = len(extracted) - jpg_count - raw_count

    stats: dict[str, int | list[str]] = {
        "total_in_archive": len(all_entries),
        "duplicates_removed": len(skipped),
        "images_extracted": len(extracted),
        "jpg_count": jpg_count,
        "raw_count": raw_count,
        "other_count": other_count,
        "skipped_files": skipped,
    }

    logger.info(
        "Extracted %d images (%d JPG, %d RAW, %d other). Removed %d duplicates.",
        len(extracted), jpg_count, raw_count, other_count, len(skipped),
    )
    return stats


def import_from_directory(
    source_dir: Path,
    output_dir: Path,
    prefer_jpg: bool = True,
) -> dict[str, int | list[str]]:
    """Copy and deduplicate images from a local directory.

    Args:
        source_dir: Directory containing images.
        output_dir: Directory to write deduplicated images.
        prefer_jpg: When True, keep JPG over RAW for duplicate stems.

    Returns:
        Statistics dictionary.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    all_files = [
        f.name for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_IMAGE_EXTENSIONS
    ]

    if prefer_jpg:
        kept, skipped = deduplicate_images(all_files)
    else:
        kept = all_files
        skipped = []

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    for name in kept:
        src = source_dir / name
        dst = output_dir / name
        shutil.copy2(src, dst)
        extracted.append(name)

    jpg_count = sum(1 for f in extracted if Path(f).suffix.lower() in JPG_EXTENSIONS)
    raw_count = sum(1 for f in extracted if Path(f).suffix.lower() in RAW_EXTENSIONS)
    other_count = len(extracted) - jpg_count - raw_count

    return {
        "total_in_source": len(all_files),
        "duplicates_removed": len(skipped),
        "images_extracted": len(extracted),
        "jpg_count": jpg_count,
        "raw_count": raw_count,
        "other_count": other_count,
        "skipped_files": skipped,
    }


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Import and deduplicate biologist's toad photos"
    )
    parser.add_argument("input", type=Path, help="ZIP file or directory with images")
    parser.add_argument("output", type=Path, help="Output directory for clean images")
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep RAW files even when JPG exists for the same image",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.input.suffix.lower() == ".zip":
        stats = import_from_zip(args.input, args.output, prefer_jpg=not args.keep_raw)
    elif args.input.is_dir():
        stats = import_from_directory(args.input, args.output, prefer_jpg=not args.keep_raw)
    else:
        parser.error("Input must be a .zip file or a directory")
        return

    print(f"\n=== Import Results ===")
    print(f"  Images in source:    {stats.get('total_in_archive', stats.get('total_in_source'))}")
    print(f"  Duplicates removed:  {stats['duplicates_removed']}")
    print(f"  Images extracted:    {stats['images_extracted']}")
    print(f"    JPG:  {stats['jpg_count']}")
    print(f"    RAW:  {stats['raw_count']}")
    print(f"    Other: {stats['other_count']}")


if __name__ == "__main__":
    main()
