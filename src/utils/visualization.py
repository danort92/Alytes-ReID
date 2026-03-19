"""Visualization utilities for the Alytes-ReID pipeline."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image: Input image (H, W, 3).
        detections: List of detections with "bbox" and "confidence" keys.
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        Image with drawn bounding boxes.
    """
    result = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        conf = det.get("confidence", 0.0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        label = f"toad {conf:.2f}"
        cv2.putText(
            result, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return result


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a semi-transparent mask on an image.

    Args:
        image: Input image (H, W, 3).
        mask: Binary mask (H, W).
        color: Overlay color in BGR.
        alpha: Transparency (0=invisible, 1=opaque).

    Returns:
        Image with mask overlay.
    """
    result = image.copy()
    overlay = result.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = color
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    return result


def draw_yolo_labels(
    image: np.ndarray,
    label_path: Any,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw YOLO-format labels on an image.

    Args:
        image: Input image (H, W, 3).
        label_path: Path to YOLO .txt label file.
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        Image with drawn bounding boxes.
    """
    from pathlib import Path

    result = image.copy()
    h, w = result.shape[:2]
    label_path = Path(label_path)

    if not label_path.exists():
        return result

    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(result, "toad", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


def create_match_grid(
    query_image: np.ndarray,
    match_images: list[np.ndarray],
    match_scores: list[float],
    match_ids: list[str],
    cell_size: tuple[int, int] = (200, 200),
) -> np.ndarray:
    """Create a visual grid showing query and top matches.

    Args:
        query_image: Query toad image.
        match_images: List of matched toad images.
        match_scores: Similarity scores for each match.
        match_ids: Individual IDs for each match.
        cell_size: Size of each cell in the grid (w, h).

    Returns:
        Grid image.
    """
    w, h = cell_size
    n_matches = len(match_images)
    total_w = w * (1 + n_matches)
    grid = np.zeros((h + 30, total_w, 3), dtype=np.uint8)

    # Query
    query_resized = cv2.resize(query_image, (w, h))
    grid[:h, :w] = query_resized
    cv2.putText(grid, "Query", (10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Matches
    for i, (img, score, mid) in enumerate(zip(match_images, match_scores, match_ids)):
        x_offset = w * (i + 1)
        resized = cv2.resize(img, (w, h))
        grid[:h, x_offset:x_offset + w] = resized
        label = f"{mid} ({score:.2f})"
        cv2.putText(grid, label, (x_offset + 5, h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return grid
