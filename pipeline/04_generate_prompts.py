"""Generate box and point prompts from segmentation masks.

By default, prompts are extracted from TransUNet prediction masks.
Use --use_gt to generate prompts from ground-truth masks instead
(for oracle / upper-bound comparison).

Usage:
    # From TransUNet predictions:
    python pipeline/04_generate_prompts.py \
        --pred_dir outputs/transunet_preds/busi \
        --output_dir outputs/prompts/busi

    # From ground-truth masks:
    python pipeline/04_generate_prompts.py \
        --pred_dir outputs/transunet_preds/busi \
        --output_dir outputs/prompts/busi_gt \
        --use_gt

    # Expand bounding box by 20% on each side:
    python pipeline/04_generate_prompts.py \
        --pred_dir outputs/transunet_preds/busi \
        --output_dir outputs/prompts/busi \
        --box_expand 0.2
"""

import argparse
import json
import os
from glob import glob

import cv2
import numpy as np

# Allow running from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

from utils.prompt_utils import mask_to_bbox, mask_to_centroid


def main():
    parser = argparse.ArgumentParser(description="Generate prompts from TransUNet predictions")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="TransUNet prediction directory (contains fold_*/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for prompt JSON files")
    parser.add_argument("--box_expand", type=float, default=0.0,
                        help="Expand bounding box by this ratio on each side "
                             "(e.g. 0.2 = 20%% of box width/height added per side)")
    parser.add_argument("--use_gt", action="store_true",
                        help="Generate prompts from GT masks instead of predictions")
    args = parser.parse_args()

    source = "ground-truth" if args.use_gt else "predictions"
    print(f"Prompt source: {source}")
    if args.box_expand > 0:
        print(f"Box expand ratio: {args.box_expand}")

    fold_dirs = sorted(glob(os.path.join(args.pred_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {args.pred_dir}")
        return

    total = 0
    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        out_fold = os.path.join(args.output_dir, fold_name)
        os.makedirs(out_fold, exist_ok=True)

        if args.use_gt:
            mask_files = sorted(glob(os.path.join(fold_dir, "gt", "*_gt.png")))
            suffix = "_gt.png"
        else:
            mask_files = sorted(glob(os.path.join(fold_dir, "*_pred.png")))
            suffix = "_pred.png"

        for mask_path in mask_files:
            basename = os.path.basename(mask_path).replace(suffix, "")

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                continue
            binary = (mask > 127).astype(np.uint8)
            h, w = binary.shape

            # Extract prompts
            bbox = mask_to_bbox(binary, expand_ratio=args.box_expand)
            centroid = mask_to_centroid(binary)

            # Fallback for empty predictions
            if bbox is None:
                bbox = [0, 0, w - 1, h - 1]

            prompt_data = {
                "case": basename,
                "image_size": [h, w],
                "box": bbox,
                "point": centroid,
                "box_expand_ratio": args.box_expand,
                "prompt_source": "gt" if args.use_gt else "prediction",
                "empty_prediction": not binary.any(),
            }

            out_path = os.path.join(out_fold, f"{basename}.json")
            with open(out_path, "w") as f:
                json.dump(prompt_data, f, indent=2)
            total += 1

        print(f"{fold_name}: {len(mask_files)} prompts generated")

    print(f"\nTotal prompts generated: {total}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
