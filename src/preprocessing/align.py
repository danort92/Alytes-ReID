"""Pose alignment for toad images using PCA or keypoint methods."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def align_toad(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "pca",
    orient_head_up: bool = True,
) -> np.ndarray:
    """Align toad so that head-tail axis is vertical.

    Args:
        image: Input image (H, W, 3).
        mask: Binary mask (H, W).
        method: Alignment method ("pca" or "keypoint").
        orient_head_up: Ensure head is at the top.

    Returns:
        Aligned image.
    """
    if method == "pca":
        return _align_pca(image, mask, orient_head_up)
    elif method == "keypoint":
        logger.warning("Keypoint alignment not yet implemented, falling back to PCA")
        return _align_pca(image, mask, orient_head_up)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_pca(
    image: np.ndarray,
    mask: np.ndarray,
    orient_head_up: bool = True,
) -> np.ndarray:
    """Align toad using PCA on mask pixels.

    The principal axis of the mask is computed and the image is rotated
    so that this axis becomes vertical.

    Args:
        image: Input image (H, W, 3).
        mask: Binary mask (H, W).
        orient_head_up: If True, ensure head points up (narrower end up).

    Returns:
        Rotated and aligned image.
    """
    coords = np.column_stack(np.where(mask > 127))
    if len(coords) < 10:
        logger.warning("Too few mask pixels for PCA alignment")
        return image

    # PCA to find principal axis
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal axis is the eigenvector with largest eigenvalue
    principal = eigenvectors[:, -1]  # (y, x) direction

    # Angle to rotate principal axis to vertical (y-axis)
    angle_rad = np.arctan2(principal[1], principal[0])
    angle_deg = np.degrees(angle_rad)
    # We want principal axis vertical, so rotate by -(angle - 90)
    rotation_angle = -(angle_deg - 90)

    # Rotate image around mask center
    center = (float(mean[1]), float(mean[0]))  # (x, y) for cv2
    h, w = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Compute new bounding box to avoid clipping
    cos_a = abs(rot_matrix[0, 0])
    sin_a = abs(rot_matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    rot_matrix[0, 2] += (new_w - w) / 2
    rot_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(image, rot_matrix, (new_w, new_h))

    if orient_head_up:
        rotated = _orient_head_up(rotated, mask, rot_matrix, new_w, new_h)

    return rotated


def _orient_head_up(
    image: np.ndarray,
    original_mask: np.ndarray,
    rot_matrix: np.ndarray,
    new_w: int,
    new_h: int,
) -> np.ndarray:
    """Ensure head (narrower part) is at the top of the image.

    Uses the width distribution along the vertical axis: the head end
    is typically narrower than the body.

    Args:
        image: Rotated image.
        original_mask: Original (pre-rotation) binary mask.
        rot_matrix: Rotation matrix used.
        new_w: Width of rotated image.
        new_h: Height of rotated image.

    Returns:
        Image with head oriented upward, flipped if necessary.
    """
    # Rotate the mask too
    rotated_mask = cv2.warpAffine(original_mask, rot_matrix, (new_w, new_h))

    # Compare width in top third vs bottom third
    h = rotated_mask.shape[0]
    third = h // 3
    top_width = np.sum(rotated_mask[:third, :] > 127)
    bottom_width = np.sum(rotated_mask[2 * third:, :] > 127)

    # Head is narrower → should have fewer pixels
    if top_width > bottom_width:
        # Head is at bottom, flip 180
        image = cv2.rotate(image, cv2.ROTATE_180)

    return image
