"""Cross-dataset inference: evaluate TransUNet trained on one dataset against another.

Loads all fold models from the source dataset (e.g., BUSI) and runs inference
on the entire target dataset (e.g., BUSBRA). Supports per-fold evaluation and
ensemble prediction via majority voting across folds.

Usage:
    # Step 1: Preprocess BUSBRA
    python pipeline/01_preprocess.py \
        --dataset busbra \
        --data_dir /path/to/BUSBRA \
        --output_dir outputs/preprocessed/busbra

    # Step 2: Cross-dataset inference (BUSI -> BUSBRA)
    python pipeline/12_cross_dataset_infer.py \
        --data_dir outputs/preprocessed/busbra \
        --model_dir /path/to/transunet_best_models/busi \
        --output_dir outputs/cross_dataset/busi_to_busbra \
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
sys.path.insert(0, os.path.join(ROOT_DIR, "TransUNet"))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils.metrics import compute_dice, compute_iou, compute_hd95


def build_model(vit_name, img_size, n_classes):
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(img_size / config_vit.patches.size[0]),
            int(img_size / config_vit.patches.size[1]),
        )
    return ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)


def run_inference(model, image, img_size, device):
    """Run single-image inference, return prediction at original resolution."""
    orig_h, orig_w = image.shape[:2]
    img_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(
        img_resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize back to original resolution
    pred_orig = cv2.resize(pred_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_orig


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset TransUNet inference (e.g., BUSI-trained -> BUSBRA)"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed target dataset directory (with npz files)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with fold_*/best.pth from source dataset training")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions and metrics")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Gather all npz files in target dataset
    npz_files = sorted(glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"Error: No npz files found in {args.data_dir}")
        return
    print(f"Found {len(npz_files)} target samples in {args.data_dir}")

    # Find all fold model checkpoints
    fold_dirs = sorted(glob(os.path.join(args.model_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {args.model_dir}")
        return

    # Filter to folds that have best.pth
    fold_ckpts = []
    for fd in fold_dirs:
        ckpt = os.path.join(fd, "best.pth")
        if os.path.exists(ckpt):
            fold_ckpts.append((os.path.basename(fd), ckpt))
    print(f"Found {len(fold_ckpts)} fold models: {[f[0] for f in fold_ckpts]}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    ensemble_pred_dir = os.path.join(args.output_dir, "ensemble")
    ensemble_gt_dir = os.path.join(args.output_dir, "ensemble", "gt")
    os.makedirs(ensemble_pred_dir, exist_ok=True)
    os.makedirs(ensemble_gt_dir, exist_ok=True)

    # --- Per-fold inference ---
    # Store per-fold predictions: fold_name -> {case_name: pred_mask}
    all_fold_preds = {}
    per_fold_metrics = {}

    for fold_name, ckpt_path in fold_ckpts:
        print(f"\n=== {fold_name} ===")
        model = build_model(args.vit_name, args.img_size, args.n_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model = model.to(args.device)
        model.eval()

        fold_pred_dir = os.path.join(args.output_dir, fold_name)
        fold_gt_dir = os.path.join(args.output_dir, fold_name, "gt")
        os.makedirs(fold_pred_dir, exist_ok=True)
        os.makedirs(fold_gt_dir, exist_ok=True)

        fold_preds = {}
        fold_sample_metrics = []

        for npz_path in npz_files:
            case_name = os.path.splitext(os.path.basename(npz_path))[0]
            data = np.load(npz_path)
            image = data["image"]  # H, W, 3 uint8
            label = data["label"]  # H, W uint8

            pred = run_inference(model, image, args.img_size, args.device)
            fold_preds[case_name] = pred

            # Save prediction and GT
            cv2.imwrite(
                os.path.join(fold_pred_dir, f"{case_name}_pred.png"),
                pred * 255,
            )
            cv2.imwrite(
                os.path.join(fold_gt_dir, f"{case_name}_gt.png"),
                label * 255,
            )

            # Compute metrics
            dice = compute_dice(pred, label)
            iou = compute_iou(pred, label)
            hd95 = compute_hd95(pred, label)

            fold_sample_metrics.append({
                "case": case_name,
                "dice": dice,
                "iou": iou,
                "hd95": hd95 if hd95 != float("inf") else None,
            })

        all_fold_preds[fold_name] = fold_preds

        # Aggregate fold metrics
        dices = [m["dice"] for m in fold_sample_metrics]
        ious = [m["iou"] for m in fold_sample_metrics]
        hd95s = [m["hd95"] for m in fold_sample_metrics if m["hd95"] is not None]

        fold_summary = {
            "fold": fold_name,
            "n_samples": len(fold_sample_metrics),
            "dice_mean": float(np.mean(dices)) if dices else 0,
            "dice_std": float(np.std(dices)) if dices else 0,
            "iou_mean": float(np.mean(ious)) if ious else 0,
            "iou_std": float(np.std(ious)) if ious else 0,
            "hd95_mean": float(np.mean(hd95s)) if hd95s else None,
            "hd95_std": float(np.std(hd95s)) if hd95s else None,
            "per_sample": fold_sample_metrics,
        }
        per_fold_metrics[fold_name] = fold_summary

        print(f"  {fold_name}: Dice={fold_summary['dice_mean']:.4f}±{fold_summary['dice_std']:.4f}, "
              f"IoU={fold_summary['iou_mean']:.4f}±{fold_summary['iou_std']:.4f}")
        if fold_summary["hd95_mean"] is not None:
            print(f"           HD95={fold_summary['hd95_mean']:.2f}±{fold_summary['hd95_std']:.2f}")

        # Save fold metrics
        with open(os.path.join(fold_pred_dir, "metrics.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)

        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Ensemble (majority voting across folds) ---
    print(f"\n=== Ensemble (majority vote, {len(fold_ckpts)} folds) ===")
    ensemble_sample_metrics = []

    for npz_path in npz_files:
        case_name = os.path.splitext(os.path.basename(npz_path))[0]
        data = np.load(npz_path)
        label = data["label"]

        # Collect predictions from all folds
        preds = []
        for fold_name in all_fold_preds:
            if case_name in all_fold_preds[fold_name]:
                preds.append(all_fold_preds[fold_name][case_name])

        if not preds:
            continue

        # Majority voting: pixel is foreground if > half the folds predict foreground
        vote_sum = np.zeros_like(preds[0], dtype=np.float32)
        for p in preds:
            vote_sum += p.astype(np.float32)
        ensemble_pred = (vote_sum > len(preds) / 2.0).astype(np.uint8)

        # Save ensemble prediction and GT
        cv2.imwrite(
            os.path.join(ensemble_pred_dir, f"{case_name}_pred.png"),
            ensemble_pred * 255,
        )
        cv2.imwrite(
            os.path.join(ensemble_gt_dir, f"{case_name}_gt.png"),
            label * 255,
        )

        # Compute metrics
        dice = compute_dice(ensemble_pred, label)
        iou = compute_iou(ensemble_pred, label)
        hd95 = compute_hd95(ensemble_pred, label)

        ensemble_sample_metrics.append({
            "case": case_name,
            "dice": dice,
            "iou": iou,
            "hd95": hd95 if hd95 != float("inf") else None,
        })

    # Aggregate ensemble metrics
    dices = [m["dice"] for m in ensemble_sample_metrics]
    ious = [m["iou"] for m in ensemble_sample_metrics]
    hd95s = [m["hd95"] for m in ensemble_sample_metrics if m["hd95"] is not None]

    ensemble_summary = {
        "method": "majority_vote",
        "n_folds": len(fold_ckpts),
        "n_samples": len(ensemble_sample_metrics),
        "dice_mean": float(np.mean(dices)) if dices else 0,
        "dice_std": float(np.std(dices)) if dices else 0,
        "iou_mean": float(np.mean(ious)) if ious else 0,
        "iou_std": float(np.std(ious)) if ious else 0,
        "hd95_mean": float(np.mean(hd95s)) if hd95s else None,
        "hd95_std": float(np.std(hd95s)) if hd95s else None,
        "per_sample": ensemble_sample_metrics,
    }

    print(f"  Ensemble: Dice={ensemble_summary['dice_mean']:.4f}±{ensemble_summary['dice_std']:.4f}, "
          f"IoU={ensemble_summary['iou_mean']:.4f}±{ensemble_summary['iou_std']:.4f}")
    if ensemble_summary["hd95_mean"] is not None:
        print(f"            HD95={ensemble_summary['hd95_mean']:.2f}±{ensemble_summary['hd95_std']:.2f}")

    # Save ensemble metrics
    with open(os.path.join(ensemble_pred_dir, "metrics.json"), "w") as f:
        json.dump(ensemble_summary, f, indent=2)

    # --- Overall summary ---
    overall = {
        "per_fold": per_fold_metrics,
        "ensemble": ensemble_summary,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(overall, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Cross-Dataset Evaluation Summary")
    print(f"  Source models: {args.model_dir}")
    print(f"  Target data:   {args.data_dir} ({len(npz_files)} samples)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Dice':>12} {'IoU':>12} {'HD95':>12}")
    print(f"{'-'*51}")
    for fold_name, summary in per_fold_metrics.items():
        hd_str = f"{summary['hd95_mean']:.2f}" if summary["hd95_mean"] is not None else "N/A"
        print(f"  {fold_name:<13} {summary['dice_mean']:.4f}±{summary['dice_std']:.4f} "
              f"{summary['iou_mean']:.4f}±{summary['iou_std']:.4f} {hd_str:>8}")
    hd_str = f"{ensemble_summary['hd95_mean']:.2f}" if ensemble_summary["hd95_mean"] is not None else "N/A"
    print(f"  {'ensemble':<13} {ensemble_summary['dice_mean']:.4f}±{ensemble_summary['dice_std']:.4f} "
          f"{ensemble_summary['iou_mean']:.4f}±{ensemble_summary['iou_std']:.4f} {hd_str:>8}")
    print(f"{'='*70}")
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
