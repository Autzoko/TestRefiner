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


def compute_crop_box(
    box: List[int],
    image_h: int,
    image_w: int,
    expand_ratio: float = 0.5,
    min_crop_size: int = 64,
) -> Tuple[int, int, int, int]:
    """Compute a crop region around a bounding box with expansion.

    Args:
        box: [x1, y1, x2, y2] in original pixel space.
        image_h: Full image height.
        image_w: Full image width.
        expand_ratio: Fraction of box width/height to expand on each side.
        min_crop_size: Minimum crop dimension (prevents degenerate tiny crops).

    Returns:
        (cx1, cy1, cx2, cy2) — crop box clamped to image bounds.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return (0, 0, image_w, image_h)

    dx = bw * expand_ratio
    dy = bh * expand_ratio

    cx1 = int(max(0, x1 - dx))
    cy1 = int(max(0, y1 - dy))
    cx2 = int(min(image_w, x2 + dx))
    cy2 = int(min(image_h, y2 + dy))

    # Enforce minimum crop size
    if (cx2 - cx1) < min_crop_size:
        center_x = (cx1 + cx2) // 2
        cx1 = max(0, center_x - min_crop_size // 2)
        cx2 = min(image_w, cx1 + min_crop_size)
    if (cy2 - cy1) < min_crop_size:
        center_y = (cy1 + cy2) // 2
        cy1 = max(0, center_y - min_crop_size // 2)
        cy2 = min(image_h, cy1 + min_crop_size)

    return (cx1, cy1, cx2, cy2)


def transform_prompts_to_crop_space(
    prompts: dict,
    crop_box: Tuple[int, int, int, int],
) -> dict:
    """Shift prompt coordinates from original image space into crop space.

    Args:
        prompts: Dict with "box" [x1,y1,x2,y2] and "point" [x,y] in orig space.
        crop_box: (cx1, cy1, cx2, cy2) defining the crop region in orig space.

    Returns:
        New prompts dict with coordinates in crop-local space.
    """
    cx1, cy1, cx2, cy2 = crop_box
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    bx1, by1, bx2, by2 = prompts["box"]
    px, py = prompts["point"]

    new_prompts = dict(prompts)
    new_prompts["box"] = [bx1 - cx1, by1 - cy1, bx2 - cx1, by2 - cy1]
    new_prompts["point"] = [px - cx1, py - cy1]
    new_prompts["image_size"] = [crop_h, crop_w]
    return new_prompts


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
