"""Evaluation metrics for segmentation."""
import numpy as np
import torch


def compute_dice(pred, gt):
    """Compute Dice coefficient.
    Args:
        pred: binary numpy array or torch tensor
        gt: binary numpy array or torch tensor
    Returns:
        float: Dice coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    intersection = (pred & gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum())


def compute_iou(pred, gt):
    """Compute IoU (Jaccard index)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_hd95(pred, gt):
    """Compute 95th percentile Hausdorff distance using medpy."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf')
    try:
        from medpy.metric.binary import hd95
        return hd95(pred, gt)
    except Exception:
        return float('inf')


def compute_mask_iou_batch(pred_masks, gt_masks):
    """Compute IoU between predicted and GT masks in batch (for UltraSAM training).
    Args:
        pred_masks: (B, N, H, W) logits
        gt_masks: (B, H, W) binary
    Returns:
        (B, N) IoU values
    """
    B, N, H, W = pred_masks.shape
    gt_expanded = gt_masks.unsqueeze(1).expand(-1, N, -1, -1)
    pred_binary = (pred_masks.sigmoid() > 0.5).float()
    intersection = (pred_binary * gt_expanded).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + gt_expanded.sum(dim=(2, 3)) - intersection
    union = torch.clamp(union, min=1e-6)
    return intersection / union


def evaluate_segmentation(pred, gt):
    """Compute all metrics for a single prediction."""
    return {
        'dice': compute_dice(pred, gt),
        'iou': compute_iou(pred, gt),
        'hd95': compute_hd95(pred, gt),
    }
