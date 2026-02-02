"""Generate box and point prompts from TransUNet prediction masks.

Usage:
    python pipeline/04_generate_prompts.py \
        --pred_dir outputs/transunet_preds/busi \
        --output_dir outputs/prompts/busi
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
    args = parser.parse_args()

    fold_dirs = sorted(glob(os.path.join(args.pred_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {args.pred_dir}")
        return

    total = 0
    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        out_fold = os.path.join(args.output_dir, fold_name)
        os.makedirs(out_fold, exist_ok=True)

        pred_files = sorted(glob(os.path.join(fold_dir, "*_pred.png")))
        for pred_path in pred_files:
            basename = os.path.basename(pred_path).replace("_pred.png", "")

            # Read prediction mask
            mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read {pred_path}")
                continue
            binary = (mask > 127).astype(np.uint8)
            h, w = binary.shape

            # Extract prompts
            bbox = mask_to_bbox(binary)
            centroid = mask_to_centroid(binary)

            # Fallback for empty predictions
            if bbox is None:
                bbox = [0, 0, w - 1, h - 1]

            prompt_data = {
                "case": basename,
                "image_size": [h, w],
                "box": bbox,
                "point": centroid,
                "empty_prediction": not binary.any(),
            }

            out_path = os.path.join(out_fold, f"{basename}.json")
            with open(out_path, "w") as f:
                json.dump(prompt_data, f, indent=2)
            total += 1

        print(f"{fold_name}: {len(pred_files)} prompts generated")

    print(f"\nTotal prompts generated: {total}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
