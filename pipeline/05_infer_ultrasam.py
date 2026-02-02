"""Run UltraSAM inference using prompts generated from TransUNet predictions.

Uses mmengine/mmdet to build the UltraSAM model, inject prompts, and run inference.

Prompt modes:
  - box:   Use bounding box prompt only (prompt_type=BOX)
  - point: Use centroid point prompt only (prompt_type=POINT)
  - both:  Send both as two separate instances (one POINT + one BOX)

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

# Add UltraSam root to sys.path so that endosam.* imports resolve
ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)

from utils.prompt_utils import transform_coords


def _tensor_to_numpy(t):
    """Convert a torch.Tensor to numpy without relying on the torch-numpy bridge."""
    if isinstance(t, torch.Tensor):
        return np.array(t.cpu().detach().tolist())
    return np.array(t)


def load_ultrasam_model(config_path, ckpt_path, device):
    """Build and load UltraSAM model using mmengine."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint

    # Register all mmdet built-in modules (DetDataPreprocessor, etc.)
    from mmdet.utils import register_all_modules
    register_all_modules()

    cfg = Config.fromfile(config_path)

    # Process custom_imports from the config to register UltraSam modules
    if hasattr(cfg, "custom_imports"):
        import importlib
        modules = cfg.custom_imports.get("imports", [])
        allow_failed = cfg.custom_imports.get("allow_failed_imports", False)
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                if not allow_failed:
                    raise
                print(f"Warning: Failed to import {mod_name}: {e}")

    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model, cfg


def preprocess_image(image_bgr, target_size=1024):
    """Resize image to target_size (aspect-ratio preserving, zero-padded).

    Returns:
        padded: (target_size, target_size, 3) uint8 RGB
        scale: scaling factor applied
        new_h, new_w: actual content dimensions within the padded image
    """
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    # Convert BGR to RGB
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return padded_rgb, scale, new_h, new_w


def build_data_sample(prompts, prompt_type, orig_h, orig_w, new_h, new_w,
                      scale, target_size, device):
    """Build a DetDataSample with correctly formatted prompts.

    UltraSAM gt_instances expects:
      - points:       (N, num_pts, 2) float tensor — point prompts in 1024-space
      - boxes:        (N, 2, 2) float tensor — [[x1,y1],[x2,y2]] in 1024-space
      - labels:       (N,) long tensor — class labels (0 for all)
      - prompt_types: (N,) long tensor — 0=POINT, 1=BOX per instance

    All attributes must be present regardless of prompt mode.
    """
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    data_sample = DetDataSample()
    gt_instances = InstanceData()

    # Transform prompts to 1024-space
    box = prompts["box"]
    box_1024 = transform_coords(box, (orig_h, orig_w), target_size, is_box=True)
    x1, y1, x2, y2 = box_1024

    point = prompts["point"]
    point_1024 = transform_coords(point, (orig_h, orig_w), target_size, is_box=False)
    px, py = point_1024

    if prompt_type == "box":
        # Single instance with BOX prompt
        # For box prompt, use box center as the point (required but not primary)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gt_instances.points = torch.tensor(
            [[[cx, cy]]], dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
        gt_instances.labels = torch.tensor(
            [0], dtype=torch.long, device=device)
        gt_instances.prompt_types = torch.tensor(
            [1], dtype=torch.long, device=device)

    elif prompt_type == "point":
        # Single instance with POINT prompt
        gt_instances.points = torch.tensor(
            [[[px, py]]], dtype=torch.float32, device=device)
        # Placeholder box (required attribute, not used for POINT prompt type)
        gt_instances.boxes = torch.tensor(
            [[[0.0, 0.0], [float(target_size), float(target_size)]]],
            dtype=torch.float32, device=device)
        gt_instances.labels = torch.tensor(
            [0], dtype=torch.long, device=device)
        gt_instances.prompt_types = torch.tensor(
            [0], dtype=torch.long, device=device)

    elif prompt_type == "both":
        # Two instances: instance 0 = POINT, instance 1 = BOX
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gt_instances.points = torch.tensor(
            [[[px, py]],       # instance 0: actual point
             [[cx, cy]]],      # instance 1: box center (placeholder)
            dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[0.0, 0.0], [float(target_size), float(target_size)]],  # inst 0: placeholder
             [[x1, y1], [x2, y2]]],                                    # inst 1: actual box
            dtype=torch.float32, device=device)
        gt_instances.labels = torch.tensor(
            [0, 0], dtype=torch.long, device=device)
        gt_instances.prompt_types = torch.tensor(
            [0, 1], dtype=torch.long, device=device)  # POINT, BOX

    data_sample.gt_instances = gt_instances

    # Set image metadata for the 3-stage rescaling in SAMHead:
    #   pad_shape -> img_shape -> ori_shape
    data_sample.set_metainfo({
        "img_shape": (new_h, new_w),          # actual content within padded image
        "ori_shape": (orig_h, orig_w),         # true original image size
        "scale_factor": (new_w / orig_w, new_h / orig_h),
        "pad_shape": (target_size, target_size),
        "batch_input_shape": (target_size, target_size),
    })

    return data_sample


def run_ultrasam_inference(model, image_bgr, prompts, prompt_type, device,
                           target_size=1024):
    """Run UltraSAM inference on a single image with given prompts.

    Returns: binary mask (np.uint8) at original resolution.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    padded_rgb, scale, new_h, new_w = preprocess_image(image_bgr, target_size)

    # Build image tensor: (1, 3, H, W) float32 — raw pixel values.
    # DetDataPreprocessor inside model.predict() handles normalization.
    img_tensor = torch.tensor(
        padded_rgb.astype(np.float32).transpose(2, 0, 1).copy(),
        device=device,
    ).unsqueeze(0)

    # Build data sample with prompts
    data_sample = build_data_sample(
        prompts, prompt_type, orig_h, orig_w, new_h, new_w,
        scale, target_size, device,
    )

    # Run model — model.predict() wraps with data_preprocessor + SAM.predict()
    with torch.no_grad():
        results = model.predict(img_tensor, [data_sample])

    # Extract mask from results
    result = results[0] if isinstance(results, list) else results

    if not hasattr(result, "pred_instances"):
        print("Warning: No pred_instances in result")
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    pred = result.pred_instances

    if hasattr(pred, "masks") and len(pred.masks) > 0:
        # SAMHead already rescales masks to ori_shape via 3-stage interpolation
        if prompt_type == "both" and len(pred.masks) > 1:
            # Two instances: merge masks with logical OR
            masks_np = _tensor_to_numpy(pred.masks)
            mask_out = np.any(masks_np > 0, axis=0).astype(np.uint8)
        else:
            # Single instance or pick best by IoU score
            if len(pred.masks) > 1 and hasattr(pred, "scores"):
                scores = _tensor_to_numpy(pred.scores)
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0
            mask_out = _tensor_to_numpy(pred.masks[best_idx]).astype(np.uint8)
    elif hasattr(pred, "mask_logits") and len(pred.mask_logits) > 0:
        logits = _tensor_to_numpy(pred.mask_logits[0])
        if logits.ndim == 3:
            logits = logits[0]
        mask_out = (logits > 0).astype(np.uint8)
    else:
        print("Warning: No masks or mask_logits in pred_instances")
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Ensure correct output shape (should already be ori_shape from SAMHead)
    if mask_out.shape != (orig_h, orig_w):
        mask_out = cv2.resize(
            mask_out, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST,
        )

    return (mask_out > 0).astype(np.uint8)


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
    print(f"Prompt type: {args.prompt_type}")
    model, cfg = load_ultrasam_model(config_path, ckpt_path, args.device)

    # Process each fold
    fold_dirs = sorted(glob(os.path.join(args.prompt_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {args.prompt_dir}")
        return

    total = 0
    errors = 0
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
                errors += 1
                continue

            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"Warning: Could not read {img_path}")
                errors += 1
                continue

            try:
                mask = run_ultrasam_inference(
                    model, image_bgr, prompts, args.prompt_type, args.device
                )
            except Exception as e:
                print(f"Error on {case_name}: {e}")
                errors += 1
                continue

            cv2.imwrite(
                os.path.join(out_fold, f"{case_name}_pred.png"),
                mask * 255,
            )
            total += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(prompt_files)}")

        print(f"  {fold_name}: done")

    print(f"\nTotal UltraSAM predictions: {total}")
    if errors:
        print(f"Errors/skipped: {errors}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
