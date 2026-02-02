"""Segmentation metrics: Dice, IoU, HD95."""

import numpy as np


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between binary masks.

    Returns 1.0 if both masks are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum())


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU (Jaccard) between binary masks.

    Returns 1.0 if both masks are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute 95th-percentile Hausdorff distance.

    Returns float('inf') if either mask is empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("inf")
    try:
        from medpy.metric.binary import hd95
        return hd95(pred, gt)
    except ImportError:
        from scipy.ndimage import distance_transform_edt
        pred_border = pred ^ _erode(pred)
        gt_border = gt ^ _erode(gt)
        if not pred_border.any() or not gt_border.any():
            return float("inf")
        dt_gt = distance_transform_edt(~gt_border)
        dt_pred = distance_transform_edt(~pred_border)
        d_pred_to_gt = dt_gt[pred_border]
        d_gt_to_pred = dt_pred[gt_border]
        all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion (1-pixel) via shifting."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)
