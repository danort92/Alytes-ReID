"""Tests for biologist data import and deduplication."""

import zipfile
from pathlib import Path

from src.data.import_biologist_data import (
    deduplicate_images,
    import_from_directory,
    import_from_zip,
    list_images_in_zip,
)


def test_deduplicate_prefers_jpg_over_cr3() -> None:
    """When both JPG and CR3 exist for same stem, keep JPG."""
    files = ["IMG_001.JPG", "IMG_001.CR3", "IMG_002.CR3"]
    kept, skipped = deduplicate_images(files)
    assert "IMG_001.JPG" in kept
    assert "IMG_001.CR3" in skipped
    assert "IMG_002.CR3" in kept
    assert len(kept) == 2
    assert len(skipped) == 1


def test_deduplicate_keeps_single_format() -> None:
    """Files without duplicates are always kept."""
    files = ["IMG_010.JPG", "IMG_020.CR3", "IMG_030.png"]
    kept, skipped = deduplicate_images(files)
    assert len(kept) == 3
    assert len(skipped) == 0


def test_deduplicate_case_insensitive_stems() -> None:
    """Deduplication matches stems case-insensitively."""
    files = ["img_001.jpg", "IMG_001.cr3"]
    kept, skipped = deduplicate_images(files)
    assert len(kept) == 1
    assert len(skipped) == 1
    assert kept[0] == "img_001.jpg"


def test_import_from_zip(tmp_path: Path) -> None:
    """Full ZIP import extracts and deduplicates correctly."""
    zip_path = tmp_path / "photos.zip"
    output_dir = tmp_path / "output"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("IMG_001.JPG", b"\xff\xd8fake-jpg-1")
        zf.writestr("IMG_001.CR3", b"\x00raw-data-1")
        zf.writestr("IMG_002.CR3", b"\x00raw-data-2")
        zf.writestr("IMG_003.JPG", b"\xff\xd8fake-jpg-3")

    stats = import_from_zip(zip_path, output_dir)

    assert stats["total_in_archive"] == 4
    assert stats["duplicates_removed"] == 1
    assert stats["images_extracted"] == 3
    assert stats["jpg_count"] == 2
    assert stats["raw_count"] == 1

    extracted_names = {f.name for f in output_dir.iterdir()}
    assert "IMG_001.JPG" in extracted_names
    assert "IMG_001.CR3" not in extracted_names
    assert "IMG_002.CR3" in extracted_names
    assert "IMG_003.JPG" in extracted_names


def test_import_from_zip_ignores_non_images(tmp_path: Path) -> None:
    """Non-image files in ZIP are ignored."""
    zip_path = tmp_path / "mixed.zip"
    output_dir = tmp_path / "output"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("IMG_001.JPG", b"\xff\xd8fake-jpg")
        zf.writestr("readme.txt", b"not an image")
        zf.writestr("data.csv", b"col1,col2")

    stats = import_from_zip(zip_path, output_dir)
    assert stats["total_in_archive"] == 1
    assert stats["images_extracted"] == 1


def test_import_from_directory(tmp_path: Path) -> None:
    """Directory import deduplicates JPG/CR3 pairs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    (source / "IMG_010.JPG").write_bytes(b"\xff\xd8jpg")
    (source / "IMG_010.CR3").write_bytes(b"\x00raw")
    (source / "IMG_011.JPG").write_bytes(b"\xff\xd8jpg2")

    stats = import_from_directory(source, output)

    assert stats["total_in_source"] == 3
    assert stats["duplicates_removed"] == 1
    assert stats["images_extracted"] == 2
    assert (output / "IMG_010.JPG").exists()
    assert not (output / "IMG_010.CR3").exists()


def test_list_images_in_zip_skips_directories(tmp_path: Path) -> None:
    """Directory entries in ZIP should be excluded."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("photos/", "")
        zf.writestr("photos/IMG_001.JPG", b"\xff\xd8jpg")

    images = list_images_in_zip(zip_path)
    assert len(images) == 1
    assert "photos/IMG_001.JPG" in images
