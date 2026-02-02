"""Utilities for generating prompts from segmentation masks."""

from typing import List, Optional, Tuple

import numpy as np


def mask_to_bbox(
    mask: np.ndarray, expand_ratio: float = 0.0
) -> Optional[List[int]]:
    """Extract bounding box [x1, y1, x2, y2] from binary mask.

    Args:
        mask: Binary mask (H, W).
        expand_ratio: Expand the tight box by this fraction of its width/height
            on each side. E.g. 0.1 adds 10% of box width to left and right,
            10% of box height to top and bottom. Clamped to image bounds.

    Returns:
        [x1, y1, x2, y2] or None if mask is empty.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return None
    h, w = mask.shape[:2]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    if expand_ratio > 0:
        bw = x2 - x1
        bh = y2 - y1
        dx = bw * expand_ratio
        dy = bh * expand_ratio
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w - 1, x2 + dx)
        y2 = min(h - 1, y2 + dy)

    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


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
