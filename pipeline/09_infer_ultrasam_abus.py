"""Direct UltraSAM inference on ABUS dataset with predefined splits.

This script runs UltraSAM inference on ABUS dataset using GT-derived prompts,
without requiring TransUNet or k-fold cross-validation.

Usage:
    python pipeline/09_infer_ultrasam_abus.py \
        --data_dir outputs/preprocessed/abus \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/ultrasam_preds/abus \
        --split test \
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

# Add UltraSam root to sys.path
ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)

from utils.prompt_utils import (
    compute_crop_box,
    mask_to_bbox,
    mask_to_centroid,
    transform_coords,
    transform_prompts_to_crop_space,
)
from utils.metrics import compute_dice, compute_iou, compute_hd95


def load_ultrasam_model(config_path, ckpt_path, device):
    """Build and load UltraSAM model."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS

    register_all_modules()
    cfg = Config.fromfile(config_path)

    # Process custom imports
    if hasattr(cfg, "custom_imports"):
        import importlib
        for mod_name in cfg.custom_imports.get("imports", []):
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                print(f"Warning: Failed to import {mod_name}: {e}")

    # Apply MonkeyPatch
    try:
        from endosam.models.utils.custom_functional import (
            multi_head_attention_forward as custom_mha_forward,
        )
        torch.nn.functional.multi_head_attention_forward = custom_mha_forward
        print("Applied UltraSAM MonkeyPatch")
    except ImportError as e:
        print(f"Warning: Could not apply MonkeyPatch: {e}")

    model = MODELS.build(cfg.model)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("Loading finetuned checkpoint")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        load_checkpoint(model, ckpt_path, map_location="cpu")

    model = model.to(device)
    model.eval()
    return model, cfg


def generate_prompts_from_mask(mask, box_expand=0.0):
    """Generate box and point prompts from GT mask."""
    bbox = mask_to_bbox(mask, expand_ratio=box_expand)
    if bbox is None:
        h, w = mask.shape[:2]
        bbox = [0, 0, w - 1, h - 1]

    centroid = mask_to_centroid(mask)

    return {
        "box": bbox,
        "point": centroid,
        "image_size": list(mask.shape[:2]),
        "empty_prediction": not mask.any(),
    }


def preprocess_image(image_bgr, target_size=1024):
    """Resize image to target_size with aspect-ratio preserving."""
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale, new_h, new_w


def build_data_sample(prompts, prompt_type, orig_h, orig_w, new_h, new_w,
                      scale, target_size, device):
    """Build DetDataSample with prompts."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    data_sample = DetDataSample()
    gt_instances = InstanceData()

    box = prompts["box"]
    box_1024 = transform_coords(box, (orig_h, orig_w), target_size, is_box=True)
    x1, y1, x2, y2 = box_1024

    point = prompts["point"]
    point_1024 = transform_coords(point, (orig_h, orig_w), target_size, is_box=False)
    px, py = point_1024

    if prompt_type == "box":
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gt_instances.points = torch.tensor(
            [[[cx, cy]]], dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
        gt_instances.prompt_types = torch.tensor([1], dtype=torch.long, device=device)

    elif prompt_type == "point":
        gt_instances.points = torch.tensor(
            [[[px, py]]], dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[0.0, 0.0], [float(target_size), float(target_size)]]],
            dtype=torch.float32, device=device)
        gt_instances.prompt_types = torch.tensor([0], dtype=torch.long, device=device)

    elif prompt_type == "both":
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gt_instances.points = torch.tensor(
            [[[px, py]], [[cx, cy]]], dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[0.0, 0.0], [float(target_size), float(target_size)]],
             [[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
        gt_instances.prompt_types = torch.tensor([0, 1], dtype=torch.long, device=device)

    gt_instances.labels = torch.tensor(
        [0] * len(gt_instances.prompt_types), dtype=torch.long, device=device)

    data_sample.gt_instances = gt_instances
    data_sample.set_metainfo({
        "img_shape": (new_h, new_w),
        "ori_shape": (orig_h, orig_w),
        "scale_factor": (new_w / orig_w, new_h / orig_h),
        "pad_shape": (target_size, target_size),
        "batch_input_shape": (target_size, target_size),
    })

    return data_sample


def run_inference(model, image_bgr, prompts, prompt_type, device, target_size=1024):
    """Run UltraSAM inference on a single image."""
    orig_h, orig_w = image_bgr.shape[:2]
    padded_bgr, scale, new_h, new_w = preprocess_image(image_bgr, target_size)

    img_tensor = torch.tensor(
        padded_bgr.astype(np.float32).transpose(2, 0, 1).copy(),
    )

    data_sample = build_data_sample(
        prompts, prompt_type, orig_h, orig_w, new_h, new_w,
        scale, target_size, device,
    )

    with torch.no_grad():
        data = {"inputs": [img_tensor], "data_samples": [data_sample]}
        data = model.data_preprocessor(data, False)
        results = model(**data, mode="predict")

    result = results[0] if isinstance(results, list) else results

    if not hasattr(result, "pred_instances"):
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    pred = result.pred_instances
    if hasattr(pred, "masks") and len(pred.masks) > 0:
        if prompt_type == "both" and len(pred.masks) > 1:
            masks_np = np.array(pred.masks.cpu().tolist())
            mask_out = np.any(masks_np > 0, axis=0).astype(np.uint8)
        else:
            mask_out = np.array(pred.masks[0].cpu().tolist()).astype(np.uint8)
    else:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    if mask_out.shape != (orig_h, orig_w):
        mask_out = cv2.resize(mask_out, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return (mask_out > 0).astype(np.uint8)


def run_inference_crop(model, image_bgr, prompts, prompt_type, device,
                       crop_expand=0.5, target_size=1024, fixed_aspect_ratio=None):
    """Run UltraSAM inference on a cropped region."""
    full_h, full_w = image_bgr.shape[:2]

    if prompts.get("empty_prediction", False):
        return run_inference(model, image_bgr, prompts, prompt_type, device, target_size)

    crop_box = compute_crop_box(
        prompts["box"], full_h, full_w, expand_ratio=crop_expand,
        fixed_aspect_ratio=fixed_aspect_ratio)
    cx1, cy1, cx2, cy2 = crop_box
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    if crop_h >= full_h * 0.95 and crop_w >= full_w * 0.95:
        return run_inference(model, image_bgr, prompts, prompt_type, device, target_size)

    crop_bgr = image_bgr[cy1:cy2, cx1:cx2].copy()
    crop_prompts = transform_prompts_to_crop_space(prompts, crop_box)

    crop_mask = run_inference(model, crop_bgr, crop_prompts, prompt_type, device, target_size)

    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = crop_mask

    return full_mask


def main():
    parser = argparse.ArgumentParser(description="UltraSAM inference on ABUS dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed ABUS data directory")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="Path to UltraSAM config")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on")
    parser.add_argument("--prompt_type", type=str, default="box",
                        choices=["box", "point", "both"])
    parser.add_argument("--box_expand", type=float, default=0.0,
                        help="Expand prompt box by this ratio")
    parser.add_argument("--crop", action="store_true",
                        help="Use crop mode")
    parser.add_argument("--crop_expand", type=float, default=0.5,
                        help="Crop expansion ratio")
    parser.add_argument("--square", action="store_true",
                        help="Force square crops")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Resolve paths
    config_path = os.path.join(ROOT_DIR, args.ultrasam_config) \
        if not os.path.isabs(args.ultrasam_config) else args.ultrasam_config
    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    # Check data directory
    split_dir = os.path.join(args.data_dir, args.split)
    if not os.path.isdir(split_dir):
        print(f"Error: Split directory not found: {split_dir}")
        return

    # Load model
    print(f"Loading UltraSAM from {config_path}")
    model, cfg = load_ultrasam_model(config_path, ckpt_path, args.device)

    # Create output directories
    pred_dir = os.path.join(args.output_dir, args.split, "predictions")
    gt_dir = os.path.join(args.output_dir, args.split, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    fixed_aspect_ratio = 1.0 if args.square else None

    # Find all npz files
    npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))
    images_dir = os.path.join(split_dir, "images_fullres")

    print(f"\nProcessing {len(npz_files)} samples from {args.split} split")
    print(f"Prompt type: {args.prompt_type}, Crop mode: {args.crop}")

    metrics_list = []
    for i, npz_path in enumerate(npz_files):
        case_name = os.path.splitext(os.path.basename(npz_path))[0]

        # Load data
        data = np.load(npz_path)
        gt_mask = data["label"]

        # Load full-res image
        img_path = os.path.join(images_dir, f"{case_name}.png")
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Generate prompts from GT
        prompts = generate_prompts_from_mask(gt_mask, box_expand=args.box_expand)

        # Run inference
        try:
            if args.crop:
                pred_mask = run_inference_crop(
                    model, image_bgr, prompts, args.prompt_type,
                    args.device, crop_expand=args.crop_expand,
                    fixed_aspect_ratio=fixed_aspect_ratio)
            else:
                pred_mask = run_inference(
                    model, image_bgr, prompts, args.prompt_type, args.device)
        except Exception as e:
            print(f"Error on {case_name}: {e}")
            continue

        # Save predictions
        cv2.imwrite(os.path.join(pred_dir, f"{case_name}_pred.png"), pred_mask * 255)
        cv2.imwrite(os.path.join(gt_dir, f"{case_name}_gt.png"), gt_mask * 255)

        # Compute metrics
        dice = compute_dice(pred_mask, gt_mask)
        iou = compute_iou(pred_mask, gt_mask)
        try:
            hd95 = compute_hd95(pred_mask, gt_mask)
        except:
            hd95 = float('inf')

        metrics_list.append({
            "case": case_name,
            "dice": dice,
            "iou": iou,
            "hd95": hd95,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(npz_files)}")

    # Save metrics
    if metrics_list:
        metrics_path = os.path.join(args.output_dir, args.split, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_list, f, indent=2)

        # Compute summary
        dices = [m["dice"] for m in metrics_list]
        ious = [m["iou"] for m in metrics_list]
        hd95s = [m["hd95"] for m in metrics_list if m["hd95"] != float('inf')]

        print(f"\n=== {args.split.upper()} Results ===")
        print(f"Samples: {len(metrics_list)}")
        print(f"Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        print(f"IoU:  {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
        if hd95s:
            print(f"HD95: {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f}")

        # Save summary
        summary = {
            "split": args.split,
            "n_samples": len(metrics_list),
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
            "hd95_mean": float(np.mean(hd95s)) if hd95s else None,
            "hd95_std": float(np.std(hd95s)) if hd95s else None,
        }
        summary_path = os.path.join(args.output_dir, args.split, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}/{args.split}/")


if __name__ == "__main__":
    main()
