"""Compare TransUNet vs UltraSAM segmentation performance.

Computes per-sample and aggregated metrics (Dice, IoU, HD95), outputs CSV tables,
and optionally generates side-by-side visualizations.

Usage:
    python pipeline/06_compare.py \
        --transunet_dir outputs/transunet_preds/busi \
        --ultrasam_dir outputs/ultrasam_preds/busi \
        --data_dir outputs/preprocessed/busi \
        --output_dir outputs/comparison/busi \
        --vis
"""

import argparse
import csv
import json
import os
import sys
from glob import glob

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

from utils.metrics import compute_dice, compute_iou, compute_hd95
from utils.vis_utils import visualize_comparison


def load_mask(path):
    """Load a binary mask PNG (0/255 → 0/1)."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Compare TransUNet vs UltraSAM")
    parser.add_argument("--transunet_dir", type=str, required=True,
                        help="TransUNet prediction directory (with fold_*/)")
    parser.add_argument("--ultrasam_dir", type=str, required=True,
                        help="UltraSAM prediction directory (with fold_*/)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed data directory (with npz files)")
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="Prompt directory for visualization (optional)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vis", action="store_true",
                        help="Generate visualization figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fold_dirs = sorted(glob(os.path.join(args.transunet_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories in {args.transunet_dir}")
        return

    all_rows = []
    fold_summaries = []

    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        ultrasam_fold = os.path.join(args.ultrasam_dir, fold_name)
        gt_fold = os.path.join(fold_dir, "gt")

        if not os.path.isdir(ultrasam_fold):
            print(f"Warning: {ultrasam_fold} not found, skipping {fold_name}")
            continue

        # Vis output
        if args.vis:
            vis_dir = os.path.join(args.output_dir, "vis", fold_name)
            os.makedirs(vis_dir, exist_ok=True)

        # Find matching predictions
        tu_preds = sorted(glob(os.path.join(fold_dir, "*_pred.png")))
        fold_rows = []

        for tu_path in tu_preds:
            case_name = os.path.basename(tu_path).replace("_pred.png", "")
            gt_path = os.path.join(gt_fold, f"{case_name}_gt.png")
            us_path = os.path.join(ultrasam_fold, f"{case_name}_pred.png")

            gt = load_mask(gt_path)
            tu_pred = load_mask(tu_path)
            us_pred = load_mask(us_path)

            if gt is None or tu_pred is None:
                continue

            # TransUNet metrics
            tu_dice = compute_dice(tu_pred, gt)
            tu_iou = compute_iou(tu_pred, gt)
            tu_hd95 = compute_hd95(tu_pred, gt)

            # UltraSAM metrics (may be missing)
            if us_pred is not None:
                us_dice = compute_dice(us_pred, gt)
                us_iou = compute_iou(us_pred, gt)
                us_hd95 = compute_hd95(us_pred, gt)
            else:
                us_dice = us_iou = us_hd95 = None

            row = {
                "fold": fold_name,
                "case": case_name,
                "tu_dice": tu_dice,
                "tu_iou": tu_iou,
                "tu_hd95": tu_hd95 if tu_hd95 != float("inf") else None,
                "us_dice": us_dice,
                "us_iou": us_iou,
                "us_hd95": us_hd95 if us_hd95 is not None and us_hd95 != float("inf") else None,
            }
            fold_rows.append(row)
            all_rows.append(row)

            # Visualization
            if args.vis and us_pred is not None:
                npz_path = os.path.join(args.data_dir, f"{case_name}.npz")
                if os.path.exists(npz_path):
                    data = np.load(npz_path)
                    image = data["image"]

                    prompts = None
                    if args.prompt_dir:
                        prompt_path = os.path.join(
                            args.prompt_dir, fold_name, f"{case_name}.json"
                        )
                        if os.path.exists(prompt_path):
                            with open(prompt_path) as f:
                                prompts = json.load(f)

                    visualize_comparison(
                        image, gt, tu_pred, us_pred, prompts,
                        save_path=os.path.join(vis_dir, f"{case_name}.png"),
                    )

        # Fold summary
        if fold_rows:
            tu_dices = [r["tu_dice"] for r in fold_rows]
            tu_ious = [r["tu_iou"] for r in fold_rows]
            us_dices = [r["us_dice"] for r in fold_rows if r["us_dice"] is not None]
            us_ious = [r["us_iou"] for r in fold_rows if r["us_iou"] is not None]

            summary = {
                "fold": fold_name,
                "n": len(fold_rows),
                "tu_dice": f"{np.mean(tu_dices):.4f}±{np.std(tu_dices):.4f}",
                "tu_iou": f"{np.mean(tu_ious):.4f}±{np.std(tu_ious):.4f}",
                "us_dice": f"{np.mean(us_dices):.4f}±{np.std(us_dices):.4f}" if us_dices else "N/A",
                "us_iou": f"{np.mean(us_ious):.4f}±{np.std(us_ious):.4f}" if us_ious else "N/A",
            }
            fold_summaries.append(summary)
            print(f"{fold_name} (n={summary['n']}): "
                  f"TU Dice={summary['tu_dice']}, US Dice={summary['us_dice']}")

    # Overall summary
    if all_rows:
        tu_dices = [r["tu_dice"] for r in all_rows]
        tu_ious = [r["tu_iou"] for r in all_rows]
        us_dices = [r["us_dice"] for r in all_rows if r["us_dice"] is not None]
        us_ious = [r["us_iou"] for r in all_rows if r["us_iou"] is not None]
        tu_hd95s = [r["tu_hd95"] for r in all_rows if r["tu_hd95"] is not None]
        us_hd95s = [r["us_hd95"] for r in all_rows if r["us_hd95"] is not None]

        print(f"\n{'='*70}")
        print(f"Overall ({len(all_rows)} samples):")
        print(f"  TransUNet  Dice: {np.mean(tu_dices):.4f}±{np.std(tu_dices):.4f}  "
              f"IoU: {np.mean(tu_ious):.4f}±{np.std(tu_ious):.4f}  "
              f"HD95: {np.mean(tu_hd95s):.2f}±{np.std(tu_hd95s):.2f}" if tu_hd95s else "")
        if us_dices:
            print(f"  UltraSAM   Dice: {np.mean(us_dices):.4f}±{np.std(us_dices):.4f}  "
                  f"IoU: {np.mean(us_ious):.4f}±{np.std(us_ious):.4f}  "
                  f"HD95: {np.mean(us_hd95s):.2f}±{np.std(us_hd95s):.2f}" if us_hd95s else "")
        print(f"{'='*70}")

    # Save per-sample CSV
    per_sample_path = os.path.join(args.output_dir, "per_sample.csv")
    if all_rows:
        with open(per_sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-sample metrics: {per_sample_path}")

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, "metrics_summary.csv")
    if fold_summaries:
        # Add overall row
        overall = {
            "fold": "overall",
            "n": len(all_rows),
            "tu_dice": f"{np.mean(tu_dices):.4f}±{np.std(tu_dices):.4f}",
            "tu_iou": f"{np.mean(tu_ious):.4f}±{np.std(tu_ious):.4f}",
            "us_dice": f"{np.mean(us_dices):.4f}±{np.std(us_dices):.4f}" if us_dices else "N/A",
            "us_iou": f"{np.mean(us_ious):.4f}±{np.std(us_ious):.4f}" if us_ious else "N/A",
        }
        fold_summaries.append(overall)

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fold_summaries[0].keys())
            writer.writeheader()
            writer.writerows(fold_summaries)
        print(f"Summary metrics:   {summary_path}")

    if args.vis:
        print(f"Visualizations:    {os.path.join(args.output_dir, 'vis')}")


if __name__ == "__main__":
    main()
