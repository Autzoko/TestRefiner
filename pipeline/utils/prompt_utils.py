"""Utilities for generating prompts from segmentation masks."""

from typing import List, Optional, Tuple

import numpy as np


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """Extract tight bounding box [x1, y1, x2, y2] from binary mask.

    Returns None if mask is empty.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def mask_to_centroid(mask: np.ndarray) -> List[int]:
    """Extract centroid [x, y] from binary mask.

    Falls back to image center if mask is empty.
    """
    mask = mask.astype(bool)
    if not mask.any():
        h, w = mask.shape[:2]
        return [w // 2, h // 2]
    ys, xs = np.where(mask)
    return [int(np.mean(xs)), int(np.mean(ys))]


def transform_coords(
    coords: List[int],
    orig_size: Tuple[int, int],
    target_size: int = 1024,
    is_box: bool = False,
) -> List[float]:
    """Rescale pixel coordinates from original image space to target square space.

    Uses aspect-ratio-preserving scaling (longest side maps to target_size).

    Args:
        coords: [x, y] for point or [x1, y1, x2, y2] for box.
        orig_size: (H, W) of original image.
        target_size: Target square dimension (default 1024 for UltraSAM).
        is_box: Whether coords represent a bounding box.

    Returns:
        Rescaled coordinates as list of floats.
    """
    h, w = orig_size
    scale = target_size / max(h, w)
    if is_box:
        x1, y1, x2, y2 = coords
        return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
    else:
        x, y = coords
        return [x * scale, y * scale]
