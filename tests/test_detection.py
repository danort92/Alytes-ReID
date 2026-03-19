"""Tests for toad detection module."""

from src.detection.evaluate import compute_iou


def test_compute_iou_identical_boxes() -> None:
    """IoU of identical boxes should be 1.0."""
    box = [10, 10, 50, 50]
    assert compute_iou(box, box) == 1.0


def test_compute_iou_no_overlap() -> None:
    """IoU of non-overlapping boxes should be 0.0."""
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap() -> None:
    """IoU of partially overlapping boxes should be between 0 and 1."""
    box1 = [0, 0, 20, 20]
    box2 = [10, 10, 30, 30]
    iou = compute_iou(box1, box2)
    assert 0.0 < iou < 1.0
