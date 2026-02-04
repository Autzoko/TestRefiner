"""Run TransUNet inference on ABUS dataset.

Supports two modes:
1. K-fold mode: Use --model_dir pointing to directory with fold_X subdirectories
   Runs inference on each fold's validation samples.
2. Single model mode: Use --model_path with --split to specify which split to evaluate

Usage:
    # K-fold mode (inference on validation samples per fold)
    python pipeline/03_infer_transunet_abus.py \
        --data_dir outputs/preprocessed/abus \
        --model_dir outputs/transunet_models/abus_kfold \
        --output_dir outputs/transunet_preds/abus_kfold \
        --device cuda:0

    # Single model mode (inference on specific split)
    python pipeline/03_infer_transunet_abus.py \
        --data_dir outputs/preprocessed/abus \
        --model_path outputs/transunet_models/abus/best.pth \
        --output_dir outputs/transunet_preds/abus \
        --split test \
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

# Add TransUNet to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "TransUNet"))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils.metrics import compute_dice, compute_iou, compute_hd95


def build_model(config, img_size=224, n_classes=2):
    """Build TransUNet model."""
    config_vit = CONFIGS_ViT_seg[config]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3

    if config.find("R50") != -1:
        config_vit.patches.grid = (
            int(img_size / config_vit.patches.size[0]),
            int(img_size / config_vit.patches.size[1]),
        )

    model = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    return model


def run_inference(model, image, img_size, device):
    """Run inference on a single image.

    Args:
        model: TransUNet model
        image: RGB image (H, W, 3) uint8
        img_size: Model input size
        device: torch device

    Returns:
        Binary prediction mask at original resolution
    """
    orig_h, orig_w = image.shape[:2]

    # Resize to model input size
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalize and convert to tensor
    img_tensor = resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Resize prediction back to original size
    pred_resized = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)

    return pred_resized


def infer_single_split(model, data_dir, split, output_dir, img_size, device):
    """Run inference on a single split.

    Args:
        model: Loaded TransUNet model
        data_dir: Preprocessed data directory
        split: Split name (train/val/test) or fold output directory
        output_dir: Output directory for predictions
        img_size: Model input size
        device: torch device

    Returns:
        List of metric dicts
    """
    # Create output directories
    pred_dir = os.path.join(output_dir, "predictions")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Find all npz files
    npz_files = sorted(glob(os.path.join(data_dir, "*.npz")))

    metrics_list = []
    for i, npz_path in enumerate(npz_files):
        case_name = os.path.splitext(os.path.basename(npz_path))[0]

        # Load data
        data = np.load(npz_path)
        image = data["image"]  # RGB
        gt_mask = data["label"]

        # Run inference
        pred_mask = run_inference(model, image, img_size, device)

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

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(npz_files)}")

    return metrics_list


def print_and_save_summary(metrics_list, output_dir, label="Results"):
    """Print and save metrics summary."""
    dices = [m["dice"] for m in metrics_list]
    ious = [m["iou"] for m in metrics_list]
    hd95s = [m["hd95"] for m in metrics_list if m["hd95"] != float('inf')]

    print(f"\n=== {label} ===")
    print(f"Samples: {len(metrics_list)}")
    print(f"Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"IoU:  {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
    if hd95s:
        print(f"HD95: {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f}")

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)

    # Save summary
    summary = {
        "n_samples": len(metrics_list),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "hd95_mean": float(np.mean(hd95s)) if hd95s else None,
        "hd95_std": float(np.std(hd95s)) if hd95s else None,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="TransUNet inference on ABUS")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed ABUS data directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model (best.pth) for single model mode")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with fold_X subdirectories for k-fold mode")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on (single model mode only)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Model input size")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Validate arguments
    if args.model_path is None and args.model_dir is None:
        print("Error: Must specify either --model_path or --model_dir")
        return
    if args.model_path is not None and args.model_dir is not None:
        print("Error: Cannot specify both --model_path and --model_dir")
        return

    # ========== K-fold mode ==========
    if args.model_dir is not None:
        # Find fold directories
        fold_dirs = sorted(glob(os.path.join(args.model_dir, "fold_*")))
        if not fold_dirs:
            print(f"Error: No fold_* directories found in {args.model_dir}")
            return

        print(f"K-fold mode: Found {len(fold_dirs)} folds")

        # Collect all npz files from train + val directories
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")

        all_npz_files = {}
        for d in [train_dir, val_dir]:
            if os.path.isdir(d):
                for f in glob(os.path.join(d, "*.npz")):
                    case_name = os.path.splitext(os.path.basename(f))[0]
                    all_npz_files[case_name] = f

        print(f"Total samples available: {len(all_npz_files)}")

        all_metrics = []
        fold_summaries = {}

        for fold_dir in fold_dirs:
            fold_name = os.path.basename(fold_dir)
            fold_idx = int(fold_name.split("_")[1])

            model_path = os.path.join(fold_dir, "best.pth")
            split_path = os.path.join(fold_dir, "split.json")

            if not os.path.exists(model_path):
                print(f"  Warning: Model not found for {fold_name}, skipping")
                continue

            if not os.path.exists(split_path):
                print(f"  Warning: split.json not found for {fold_name}, skipping")
                continue

            # Load split info
            with open(split_path) as f:
                split_info = json.load(f)
            val_names = split_info["val"]

            print(f"\n{fold_name}: {len(val_names)} validation samples")

            # Build and load model
            model = build_model(args.vit_name, img_size=args.img_size, n_classes=args.n_classes)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model = model.to(args.device)
            model.eval()

            # Create fold output directory
            fold_output_dir = os.path.join(args.output_dir, fold_name)
            os.makedirs(fold_output_dir, exist_ok=True)

            # Run inference on validation samples
            pred_dir = os.path.join(fold_output_dir, "predictions")
            gt_dir = os.path.join(fold_output_dir, "gt")
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)

            fold_metrics = []
            for case_name in val_names:
                if case_name not in all_npz_files:
                    print(f"    Warning: {case_name} not found in data, skipping")
                    continue

                npz_path = all_npz_files[case_name]
                data = np.load(npz_path)
                image = data["image"]
                gt_mask = data["label"]

                pred_mask = run_inference(model, image, args.img_size, args.device)

                cv2.imwrite(os.path.join(pred_dir, f"{case_name}_pred.png"), pred_mask * 255)
                cv2.imwrite(os.path.join(gt_dir, f"{case_name}_gt.png"), gt_mask * 255)

                dice = compute_dice(pred_mask, gt_mask)
                iou = compute_iou(pred_mask, gt_mask)
                try:
                    hd95 = compute_hd95(pred_mask, gt_mask)
                except:
                    hd95 = float('inf')

                metric = {
                    "case": case_name,
                    "fold": fold_idx,
                    "dice": dice,
                    "iou": iou,
                    "hd95": hd95,
                }
                fold_metrics.append(metric)
                all_metrics.append(metric)

            # Save fold summary
            summary = print_and_save_summary(fold_metrics, fold_output_dir, f"Fold {fold_idx}")
            fold_summaries[fold_idx] = summary

        # Overall summary
        print(f"\n{'='*60}")
        print("Overall K-Fold Results")
        print(f"{'='*60}")

        print_and_save_summary(all_metrics, args.output_dir, "All Folds Combined")

        # Per-fold summary
        print("\nPer-fold Dice scores:")
        fold_dices = []
        for fold_idx in sorted(fold_summaries.keys()):
            d = fold_summaries[fold_idx]["dice_mean"]
            fold_dices.append(d)
            print(f"  Fold {fold_idx}: {d:.4f}")
        print(f"  Mean: {np.mean(fold_dices):.4f} +/- {np.std(fold_dices):.4f}")

        print(f"\nResults saved to: {args.output_dir}/")

    # ========== Single model mode ==========
    else:
        # Check paths
        split_dir = os.path.join(args.data_dir, args.split)
        if not os.path.isdir(split_dir):
            print(f"Error: Split directory not found: {split_dir}")
            return

        if not os.path.exists(args.model_path):
            print(f"Error: Model not found: {args.model_path}")
            return

        # Build and load model
        print(f"Loading model from {args.model_path}")
        model = build_model(args.vit_name, img_size=args.img_size, n_classes=args.n_classes)
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        model = model.to(args.device)
        model.eval()

        print(f"\nProcessing {args.split} split")

        # Run inference
        output_dir = os.path.join(args.output_dir, args.split)
        metrics_list = infer_single_split(
            model, split_dir, args.split, output_dir, args.img_size, args.device
        )

        # Save results
        print_and_save_summary(metrics_list, output_dir, f"{args.split.upper()} Results")

        print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
