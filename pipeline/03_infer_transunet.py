"""Run TransUNet inference on all validation folds, save prediction masks and metrics.

Usage:
    python pipeline/03_infer_transunet.py \
        --data_dir outputs/preprocessed/busi \
        --model_dir outputs/transunet_models/busi \
        --output_dir outputs/transunet_preds/busi \
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


def main():
    parser = argparse.ArgumentParser(description="Inference TransUNet on val folds")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed data directory (with npz files)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with fold_*/best.pth and split.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Find all folds
    fold_dirs = sorted(glob(os.path.join(args.model_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {args.model_dir}")
        return

    all_metrics = {}

    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        print(f"\n=== Processing {fold_name} ===")

        # Load split
        split_path = os.path.join(fold_dir, "split.json")
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping.")
            continue
        with open(split_path) as f:
            split = json.load(f)
        val_names = split["val"]

        # Load model
        ckpt_path = os.path.join(fold_dir, "best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping.")
            continue

        model = build_model(args.vit_name, args.img_size, args.n_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model = model.to(args.device)
        model.eval()

        # Output dirs
        pred_dir = os.path.join(args.output_dir, fold_name)
        gt_dir = os.path.join(args.output_dir, fold_name, "gt")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        fold_metrics = []
        with torch.no_grad():
            for case_name in val_names:
                npz_path = os.path.join(args.data_dir, f"{case_name}.npz")
                if not os.path.exists(npz_path):
                    print(f"Warning: {npz_path} not found, skipping.")
                    continue

                data = np.load(npz_path)
                image = data["image"]  # H, W, 3 uint8
                label = data["label"]  # H, W uint8
                orig_h, orig_w = image.shape[:2]

                # Resize for model input
                img_resized = cv2.resize(
                    image, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR
                )
                img_tensor = torch.from_numpy(
                    img_resized.astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(args.device)

                # Inference
                logits = model(img_tensor)
                pred_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Resize prediction back to original resolution
                pred_orig = cv2.resize(
                    pred_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                # Save prediction (binary 0/255)
                cv2.imwrite(
                    os.path.join(pred_dir, f"{case_name}_pred.png"),
                    pred_orig * 255,
                )

                # Save GT at original resolution
                cv2.imwrite(
                    os.path.join(gt_dir, f"{case_name}_gt.png"),
                    label * 255,
                )

                # Compute metrics
                dice = compute_dice(pred_orig, label)
                iou = compute_iou(pred_orig, label)
                hd95 = compute_hd95(pred_orig, label)

                fold_metrics.append({
                    "case": case_name,
                    "dice": dice,
                    "iou": iou,
                    "hd95": hd95 if hd95 != float("inf") else None,
                })

        # Aggregate fold metrics
        dices = [m["dice"] for m in fold_metrics]
        ious = [m["iou"] for m in fold_metrics]
        hd95s = [m["hd95"] for m in fold_metrics if m["hd95"] is not None]

        fold_summary = {
            "fold": fold_name,
            "n_samples": len(fold_metrics),
            "dice_mean": float(np.mean(dices)) if dices else 0,
            "dice_std": float(np.std(dices)) if dices else 0,
            "iou_mean": float(np.mean(ious)) if ious else 0,
            "iou_std": float(np.std(ious)) if ious else 0,
            "hd95_mean": float(np.mean(hd95s)) if hd95s else None,
            "hd95_std": float(np.std(hd95s)) if hd95s else None,
            "per_sample": fold_metrics,
        }
        all_metrics[fold_name] = fold_summary

        print(f"  {fold_name}: Dice={fold_summary['dice_mean']:.4f}±{fold_summary['dice_std']:.4f}, "
              f"IoU={fold_summary['iou_mean']:.4f}±{fold_summary['iou_std']:.4f}")

        # Save fold metrics
        with open(os.path.join(pred_dir, "metrics.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)

    # Save overall metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Overall summary
    all_dices = []
    for v in all_metrics.values():
        all_dices.extend([s["dice"] for s in v["per_sample"]])
    if all_dices:
        print(f"\nOverall TransUNet: Dice={np.mean(all_dices):.4f}±{np.std(all_dices):.4f}")


if __name__ == "__main__":
    main()
