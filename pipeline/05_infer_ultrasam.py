"""Run UltraSAM inference using prompts generated from TransUNet predictions.

Uses mmengine/mmdet to build the UltraSAM model, inject prompts, and run inference.

Usage:
    python pipeline/05_infer_ultrasam.py \
        --prompt_dir outputs/prompts/busi \
        --image_dir outputs/preprocessed/busi/images_fullres \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/ultrasam_preds/busi \
        --prompt_type box \
        --device cuda:0
"""

import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

from utils.prompt_utils import transform_coords


def load_ultrasam_model(config_path, ckpt_path, device):
    """Build and load UltraSAM model using mmengine."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model, cfg


def preprocess_image(image_bgr, target_size=1024):
    """Resize image to target_size (aspect-ratio preserving, zero-padded).

    Returns:
        padded: (target_size, target_size, 3) uint8
        scale: scaling factor applied
        pad_h, pad_w: padding offsets
    """
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale, 0, 0


def run_ultrasam_inference(model, image_bgr, prompts, prompt_type, device, target_size=1024):
    """Run UltraSAM inference on a single image with given prompts.

    Returns: binary mask at original resolution.
    """
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    orig_h, orig_w = image_bgr.shape[:2]
    padded, scale, _, _ = preprocess_image(image_bgr, target_size)

    # Convert to tensor (C, H, W), normalized
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32).transpose(2, 0, 1))

    # Normalize with ImageNet stats
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Build data sample with prompts
    data_sample = DetDataSample()
    gt_instances = InstanceData()

    if prompt_type in ("box", "both"):
        box = prompts["box"]
        box_1024 = transform_coords(box, (orig_h, orig_w), target_size, is_box=True)
        gt_instances.bboxes = torch.tensor([box_1024], dtype=torch.float32).to(device)
        gt_instances.bp_type = torch.tensor([0]).to(device)  # 0 = box prompt

    if prompt_type in ("point", "both"):
        point = prompts["point"]
        point_1024 = transform_coords(point, (orig_h, orig_w), target_size, is_box=False)
        point_tensor = torch.tensor([[point_1024]], dtype=torch.float32).to(device)
        label_tensor = torch.tensor([[1]], dtype=torch.int64).to(device)
        if hasattr(gt_instances, "bboxes"):
            # Both mode: attach point info alongside box
            gt_instances.points = point_tensor
            gt_instances.point_labels = label_tensor
        else:
            gt_instances.points = point_tensor
            gt_instances.point_labels = label_tensor
            gt_instances.bp_type = torch.tensor([1]).to(device)  # 1 = point prompt

    data_sample.gt_instances = gt_instances

    # Set image metadata
    data_sample.set_metainfo({
        "img_shape": (target_size, target_size),
        "ori_shape": (orig_h, orig_w),
        "scale_factor": (scale, scale),
        "pad_shape": (target_size, target_size),
        "batch_input_shape": (target_size, target_size),
    })

    # Run model
    with torch.no_grad():
        results = model.predict(img_tensor, [data_sample])

    # Extract mask from results
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    # Try to get mask from pred_instances
    if hasattr(result, "pred_instances") and hasattr(result.pred_instances, "masks"):
        mask_pred = result.pred_instances.masks[0]
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.cpu().numpy()
    elif hasattr(result, "pred_instances") and hasattr(result.pred_instances, "mask_logits"):
        logits = result.pred_instances.mask_logits[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        mask_pred = (logits > 0).astype(np.uint8)
        if mask_pred.ndim == 3:
            mask_pred = mask_pred[0]
    else:
        print("Warning: Could not extract mask from model output")
        mask_pred = np.zeros((target_size, target_size), dtype=np.uint8)

    # Handle mask dimensions
    if mask_pred.ndim == 3:
        mask_pred = mask_pred[0]

    # Crop out padding and resize to original
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    mask_cropped = mask_pred[:new_h, :new_w]
    mask_orig = cv2.resize(
        mask_cropped.astype(np.uint8), (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return (mask_orig > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="UltraSAM inference with TransUNet prompts")
    parser.add_argument("--prompt_dir", type=str, required=True,
                        help="Directory with prompt JSONs (fold_*/)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with full-res PNG images")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="Path to UltraSAM config file")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="box",
                        choices=["box", "point", "both"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Resolve paths relative to project root
    config_path = os.path.join(ROOT_DIR, args.ultrasam_config) \
        if not os.path.isabs(args.ultrasam_config) else args.ultrasam_config
    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    print(f"Loading UltraSAM from {config_path}")
    model, cfg = load_ultrasam_model(config_path, ckpt_path, args.device)

    # Process each fold
    fold_dirs = sorted(glob(os.path.join(args.prompt_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {args.prompt_dir}")
        return

    total = 0
    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        out_fold = os.path.join(args.output_dir, fold_name)
        os.makedirs(out_fold, exist_ok=True)

        prompt_files = sorted(glob(os.path.join(fold_dir, "*.json")))
        print(f"\n=== {fold_name}: {len(prompt_files)} samples ===")

        for i, pf in enumerate(prompt_files):
            with open(pf) as f:
                prompts = json.load(f)

            case_name = prompts["case"]
            img_path = os.path.join(args.image_dir, f"{case_name}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"Warning: Could not read {img_path}")
                continue

            mask = run_ultrasam_inference(
                model, image_bgr, prompts, args.prompt_type, args.device
            )

            cv2.imwrite(
                os.path.join(out_fold, f"{case_name}_pred.png"),
                mask * 255,
            )
            total += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(prompt_files)}")

        print(f"  {fold_name}: {len(prompt_files)} predictions saved")

    print(f"\nTotal UltraSAM predictions: {total}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
