#!/usr/bin/env python
"""UltraSAM-only inference reproducing the original evaluation pipeline.

Exactly reproduces the original UltraSAM inference process:
  1. FixScaleResize to 1024x1024 (keep aspect ratio, pad shorter side)
  2. Center-point prompt from GT mask (mean of all foreground pixel coords)
  3. First pass: point prompt only, single mask output (multimask_output=False)
  4. Mask refinement: feed predicted mask logits as mask prompt + derived box
     prompt from predicted binary mask, multimask output (3 masks)
  5. Select best mask by IoU prediction score
  6. Threshold at logits > 0, rescale 256->pad_shape->crop->ori_shape

Usage:
    python inference_ultrasam.py \
        --dataset busi \
        --data_dir /path/to/BUSI \
        --ultrasam_ckpt weights/ultrasam_standalone.pth \
        --fold 0 \
        --save_dir output/ultrasam_inference
"""

import os
import sys
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.ultrasam import build_ultrasam_vit_b
from utils.metrics import compute_dice, compute_iou, compute_hd95


# ---------------------------------------------------------------------------
# Preprocessing utilities (match original UltraSAM pipeline)
# ---------------------------------------------------------------------------

def fix_scale_resize(image, target_size=1024):
    """Resize image keeping aspect ratio, pad to target_size x target_size.

    Reproduces mmdet FixScaleResize(scale=(1024, 1024), keep_ratio=True).

    Args:
        image: (H, W, 3) uint8 RGB image.
        target_size: target side length.

    Returns:
        padded:       (target_size, target_size, 3) padded image.
        scale_factor: (w_scale, h_scale) — new / original.
        img_shape:    (h_resized, w_resized) before padding.
        pad_shape:    (target_size, target_size).
        ori_shape:    (h_orig, w_orig).
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    scale_factor = (new_w / w, new_h / h)
    return padded, scale_factor, (new_h, new_w), (target_size, target_size), (h, w)


def preprocess_image(image):
    """ImageNet normalisation (BGR order matching original DetDataPreprocessor).

    Original config:
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    We assume *image* is already RGB, so apply directly.
    """
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = image.astype(np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


# ---------------------------------------------------------------------------
# Prompt generation (match original GetPointFromMask / GetPointBox)
# ---------------------------------------------------------------------------

def get_center_point_from_gt(mask, scale_factor):
    """Compute center point of GT mask, scale to resized-image coords.

    Reproduces GetPointFromMask(test=True, normalize=False, get_center_point=True):
        indices = np.argwhere(mask)           # original-space (row, col)
        y, x = indices.mean(axis=0)
        x_out = x * x_scale / 1 + 0.5        # normalize=False → img_width=1
        y_out = y * y_scale / 1 + 0.5
    """
    indices = np.argwhere(mask > 0)  # (N, 2) as (row=y, col=x)
    if len(indices) == 0:
        return None
    center = indices.mean(axis=0)  # (y_center, x_center)
    y_c, x_c = center
    w_scale, h_scale = scale_factor
    x_out = x_c * w_scale + 0.5
    y_out = y_c * h_scale + 0.5
    return np.array([[x_out, y_out]], dtype=np.float32)  # (1, 2)


def mask_to_bbox(binary_mask):
    """Extract tight bounding box from a 2-D boolean tensor.

    Returns (x1, y1, x2, y2) as float tensor, or None if empty.
    """
    if not binary_mask.any():
        return None
    rows = binary_mask.any(dim=1)
    cols = binary_mask.any(dim=0)
    y1 = rows.float().argmax().item()
    y2 = (rows.shape[0] - rows.flip(0).float().argmax()).item()
    x1 = cols.float().argmax().item()
    x2 = (cols.shape[0] - cols.flip(0).float().argmax()).item()
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Post-processing (match original SAMHead.predict rescale logic)
# ---------------------------------------------------------------------------

def rescale_mask(mask_logits, pad_shape, img_shape, ori_shape):
    """Rescale 256x256 mask to original image size.

    Exactly reproduces the original post-processing:
        1. Resize to pad_shape  (bilinear)
        2. Crop to img_shape    (remove padding)
        3. Resize to ori_shape  (bilinear)
        4. Threshold > 0        (bool)

    Args:
        mask_logits: (1, 1, 256, 256) raw logits (NOT thresholded).
        pad_shape:   (H_pad, W_pad), e.g. (1024, 1024).
        img_shape:   (H_img, W_img), e.g. (1024, 683).
        ori_shape:   (H_ori, W_ori), e.g. (600, 400).

    Returns:
        binary_mask: (H_ori, W_ori) numpy uint8 array.
    """
    # Threshold first (original: masks = mask_logits > 0, then resize the bool as float)
    binary = (mask_logits > 0).float()
    binary = F.interpolate(binary, size=pad_shape, mode='bilinear', align_corners=False)
    binary = binary[..., :img_shape[0], :img_shape[1]]
    binary = F.interpolate(binary, size=ori_shape, mode='bilinear', align_corners=False)
    return binary.squeeze(0).squeeze(0).bool().cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_busi_samples(root_dir, split, fold, n_folds, seed=42):
    """Load BUSI sample paths using the same splits as BUSIDataset."""
    from datasets.busi_dataset import BUSIDataset
    ds = BUSIDataset(root_dir, split=split, fold=fold, n_folds=n_folds,
                     img_size=None, transform=None, seed=seed)
    return ds.samples


def load_samples_for_dataset(dataset_name, data_dir, split, fold, n_folds, seed=42):
    """Return list of {'image': path, 'masks': [paths], 'name': str}."""
    if dataset_name == 'busi':
        return load_busi_samples(data_dir, split, fold, n_folds, seed)
    else:
        # For other datasets, use get_dataset and access .samples
        from datasets import get_dataset
        ds = get_dataset(dataset_name, data_dir, split=split, fold=fold,
                         n_folds=n_folds, img_size=None, transform=None)
        return ds.samples


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description='UltraSAM-only Inference')
    p.add_argument('--dataset', type=str, default='busi',
                   choices=['busi', 'busbra', 'bus'])
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--ultrasam_ckpt', type=str, required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--split', type=str, default='val',
                   choices=['train', 'val'])
    # Refinement control
    p.add_argument('--no_refinement', action='store_true',
                   help='Skip mask refinement (single-pass point-only inference)')
    # Output
    p.add_argument('--save_dir', type=str, default='output/ultrasam_inference')
    p.add_argument('--save_overlay', action='store_true')
    return p.parse_args()


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print("Loading UltraSAM...")
    model = build_ultrasam_vit_b(use_mask_refinement=True)
    state_dict = torch.load(args.ultrasam_ckpt, map_location='cpu', weights_only=False)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
        for k in result.missing_keys[:10]:
            print(f"    {k}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        for k in result.unexpected_keys[:10]:
            print(f"    {k}")
    model = model.to(device).eval()
    print(f"  Device: {device}")

    # Load dataset samples (paths only, no tensor conversion)
    samples = load_samples_for_dataset(
        args.dataset, args.data_dir, args.split, args.fold, args.n_folds,
    )
    print(f"  {len(samples)} {args.split} samples loaded")

    use_refinement = not args.no_refinement
    if use_refinement:
        print("  Mask refinement: ON (2-pass, point → box+mask)")
    else:
        print("  Mask refinement: OFF (single-pass, point only)")

    # Run inference
    dice_scores, iou_scores, hd95_scores = [], [], []

    for sample in tqdm(samples, desc='UltraSAM Inference'):
        # --- Load raw image and GT mask at original resolution -------------
        image_bgr = cv2.imread(sample['image'])
        if image_bgr is None:
            print(f"  Warning: cannot read {sample['image']}, skipping")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        gt_mask = None
        for mp in sample['masks']:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = (m > 127).astype(np.uint8)
                gt_mask = m if gt_mask is None else np.maximum(gt_mask, m)
        if gt_mask is None or gt_mask.max() == 0:
            continue

        # --- 1. FixScaleResize ---------------------------------------------
        padded_img, scale_factor, img_shape, pad_shape, ori_shape = \
            fix_scale_resize(image_rgb, target_size=1024)

        # --- 2. Center-point prompt from GT mask ---------------------------
        center = get_center_point_from_gt(gt_mask, scale_factor)
        if center is None:
            continue

        # --- 3. Preprocess image -------------------------------------------
        sam_input = preprocess_image(padded_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # --- 4. Encode image (reuse for both passes) -------------------
            image_embeddings = model.encode_image(sam_input)

            # --- 5. FIRST PASS: point prompt, multimask_output=False -------
            point_coords = torch.from_numpy(center).unsqueeze(0).to(device)   # (1, 1, 2)
            point_labels = torch.tensor([[2]], dtype=torch.long, device=device)  # POS=2

            mask_logits_1, iou_pred_1 = model.forward_with_prompts(
                images=None,
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            # mask_logits_1: (1, 1, 256, 256)

            if use_refinement:
                # --- 6. REFINEMENT: mask prompt + box prompt ---------------
                # Get binary mask and rescale to original size for bbox
                binary_1 = (mask_logits_1 > 0).float()
                mask_full = F.interpolate(binary_1, size=pad_shape,
                                          mode='bilinear', align_corners=False)
                mask_full = mask_full[..., :img_shape[0], :img_shape[1]]
                mask_full = F.interpolate(mask_full, size=ori_shape,
                                          mode='bilinear', align_corners=False)
                mask_full_bool = mask_full.squeeze(0).squeeze(0).bool()

                bbox_ori = mask_to_bbox(mask_full_bool)

                if bbox_ori is not None:
                    # Scale bbox from ori_shape to resized-image coords
                    w_scale, h_scale = scale_factor
                    bbox_resized = bbox_ori.clone()
                    bbox_resized[0] *= w_scale   # x1
                    bbox_resized[1] *= h_scale   # y1
                    bbox_resized[2] *= w_scale   # x2
                    bbox_resized[3] *= h_scale   # y2

                    box_coords = bbox_resized.unsqueeze(0).to(device)  # (1, 4)

                    # Second pass: box prompt + mask logits as mask prompt
                    mask_logits_2, iou_pred_2 = model.forward_with_prompts(
                        images=None,
                        image_embeddings=image_embeddings,
                        box_coords=box_coords,
                        mask_props=mask_logits_1,   # raw logits from pass 1
                        multimask_output=True,
                    )
                    # mask_logits_2: (1, 3, 256, 256), iou_pred_2: (1, 3)

                    # Select best mask by IoU prediction
                    best_idx = iou_pred_2[0].argmax().item()
                    final_logits = mask_logits_2[:, best_idx:best_idx+1]
                else:
                    # Empty prediction from pass 1, fall back
                    final_logits = mask_logits_1
            else:
                # No refinement: use first-pass output directly
                final_logits = mask_logits_1

        # --- 7. Rescale to original size -----------------------------------
        pred_mask = rescale_mask(final_logits, pad_shape, img_shape, ori_shape)

        # --- 8. Compute metrics --------------------------------------------
        dice = compute_dice(pred_mask, gt_mask)
        iou = compute_iou(pred_mask, gt_mask)
        hd = compute_hd95(pred_mask, gt_mask)

        dice_scores.append(dice)
        iou_scores.append(iou)
        if hd != float('inf'):
            hd95_scores.append(hd)

        # --- 9. Save prediction --------------------------------------------
        name = sample['name'].replace('/', '_')
        mask_path = os.path.join(args.save_dir, f'{name}_pred.png')
        cv2.imwrite(mask_path, pred_mask * 255)

        if args.save_overlay and image_rgb is not None:
            overlay = image_rgb.copy()
            overlay[pred_mask > 0] = (
                overlay[pred_mask > 0] * 0.5 +
                np.array([0, 255, 0]) * 0.5
            ).astype(np.uint8)
            overlay_path = os.path.join(args.save_dir, f'{name}_overlay.png')
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # --- 10. Print results -------------------------------------------------
    if dice_scores:
        print(f"\nResults ({len(dice_scores)} samples):")
        print(f"  Dice: {np.mean(dice_scores):.4f} +/- {np.std(dice_scores):.4f}")
        print(f"  IoU:  {np.mean(iou_scores):.4f} +/- {np.std(iou_scores):.4f}")
        if hd95_scores:
            print(f"  HD95: {np.mean(hd95_scores):.2f} +/- {np.std(hd95_scores):.2f}")
    else:
        print("\nNo valid samples found for evaluation.")


if __name__ == '__main__':
    args = get_args()
    run_inference(args)
