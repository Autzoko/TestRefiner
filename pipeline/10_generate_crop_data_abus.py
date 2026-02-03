"""Generate cropped training data for UltraSAM finetuning from ABUS dataset.

This script works with ABUS's predefined train/val/test splits instead of k-fold CV.

Usage:
    python pipeline/10_generate_crop_data_abus.py \
        --data_dir outputs/preprocessed/abus \
        --output_dir outputs/crop_data/abus \
        --crop_expand 0.5 \
        --square
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

from utils.prompt_utils import compute_crop_box, mask_to_bbox


def create_coco_annotation(
    ann_id: int,
    image_id: int,
    mask: np.ndarray,
    category_id: int = 1,
) -> Optional[Dict]:
    """Create a COCO annotation from a binary mask."""
    from pycocotools import mask as mask_util

    mask = (mask > 0).astype(np.uint8)
    if not mask.any():
        return None

    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    area = int(mask.sum())

    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, width, height],
        "area": area,
        "iscrowd": 0,
        "segmentation": rle,
    }


def process_sample(
    image_path: str,
    npz_path: str,
    image_id: int,
    ann_id_start: int,
    output_images_dir: str,
    crop_expand: float = 0.5,
    min_crop_size: int = 64,
    fixed_aspect_ratio: Optional[float] = None,
) -> Tuple[Optional[Dict], List[Dict], int]:
    """Process a single sample to create cropped data."""
    # Load image
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None, [], ann_id_start

    # Load mask from npz
    data = np.load(npz_path)
    if "label" not in data:
        return None, [], ann_id_start
    mask = (data["label"] > 0).astype(np.uint8)

    if not mask.any():
        return None, [], ann_id_start

    full_h, full_w = image_bgr.shape[:2]

    # Get bbox from GT mask
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None, [], ann_id_start

    # Compute crop region
    crop_box = compute_crop_box(
        bbox, full_h, full_w, expand_ratio=crop_expand,
        min_crop_size=min_crop_size, fixed_aspect_ratio=fixed_aspect_ratio)
    cx1, cy1, cx2, cy2 = crop_box
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    # Skip if crop covers nearly the entire image
    if crop_h >= full_h * 0.95 and crop_w >= full_w * 0.95:
        return None, [], ann_id_start

    # Crop image and mask
    crop_image = image_bgr[cy1:cy2, cx1:cx2].copy()
    crop_mask = mask[cy1:cy2, cx1:cx2].copy()

    if not crop_mask.any():
        return None, [], ann_id_start

    # Generate filename
    case_name = os.path.splitext(os.path.basename(image_path))[0]
    crop_filename = f"{case_name}_crop.png"
    crop_path = os.path.join(output_images_dir, crop_filename)

    # Save cropped image
    cv2.imwrite(crop_path, crop_image)

    # Create image info
    image_info = {
        "id": image_id,
        "file_name": crop_filename,
        "width": crop_w,
        "height": crop_h,
        "original_file": os.path.basename(image_path),
        "crop_box": [cx1, cy1, cx2, cy2],
    }

    # Create annotation
    annotation = create_coco_annotation(
        ann_id=ann_id_start,
        image_id=image_id,
        mask=crop_mask,
        category_id=1,
    )

    annotations = [annotation] if annotation else []
    next_ann_id = ann_id_start + len(annotations)

    return image_info, annotations, next_ann_id


def generate_crop_dataset(
    data_dir: str,
    output_dir: str,
    crop_expand: float = 0.5,
    min_crop_size: int = 64,
    fixed_aspect_ratio: Optional[float] = None,
):
    """Generate cropped dataset from preprocessed ABUS data."""
    splits = ["train", "val"]  # Only generate for train and val

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        images_dir = os.path.join(split_dir, "images_fullres")

        if not os.path.isdir(images_dir):
            print(f"Warning: {images_dir} not found, skipping {split}")
            continue

        # Create output directories
        output_split_dir = os.path.join(output_dir, split)
        output_images_dir = os.path.join(output_split_dir, "images")
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(os.path.join(output_split_dir, "annotations"), exist_ok=True)

        # Find all npz files
        npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))
        print(f"\n=== Processing {split}: {len(npz_files)} samples ===")

        all_images = []
        all_annotations = []
        image_id = 1
        ann_id = 1

        for npz_path in npz_files:
            case_name = os.path.splitext(os.path.basename(npz_path))[0]
            image_path = os.path.join(images_dir, f"{case_name}.png")

            if not os.path.exists(image_path):
                continue

            image_info, annotations, ann_id = process_sample(
                image_path, npz_path, image_id, ann_id,
                output_images_dir, crop_expand, min_crop_size, fixed_aspect_ratio,
            )

            if image_info:
                all_images.append(image_info)
                all_annotations.extend(annotations)
                image_id += 1

        # Save COCO format annotations
        coco_data = {
            "images": all_images,
            "annotations": all_annotations,
            "categories": [{"id": 1, "name": "lesion"}],
        }

        ann_path = os.path.join(output_split_dir, "annotations", "train.json")
        with open(ann_path, "w") as f:
            json.dump(coco_data, f, indent=2)

        print(f"  {split}: {len(all_images)} images, {len(all_annotations)} annotations")
        print(f"  Saved to: {output_split_dir}")

    # Create combined dataset (train + val for full finetuning)
    print("\n=== Creating combined dataset ===")
    combined_dir = os.path.join(output_dir, "combined")
    combined_images_dir = os.path.join(combined_dir, "images")
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(os.path.join(combined_dir, "annotations"), exist_ok=True)

    all_images = []
    all_annotations = []
    image_id = 1
    ann_id = 1

    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        images_dir = os.path.join(split_dir, "images_fullres")

        if not os.path.isdir(images_dir):
            continue

        npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))

        for npz_path in npz_files:
            case_name = os.path.splitext(os.path.basename(npz_path))[0]
            image_path = os.path.join(images_dir, f"{case_name}.png")

            if not os.path.exists(image_path):
                continue

            image_info, annotations, ann_id = process_sample(
                image_path, npz_path, image_id, ann_id,
                combined_images_dir, crop_expand, min_crop_size, fixed_aspect_ratio,
            )

            if image_info:
                all_images.append(image_info)
                all_annotations.extend(annotations)
                image_id += 1

    coco_data = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{"id": 1, "name": "lesion"}],
    }

    ann_path = os.path.join(combined_dir, "annotations", "train.json")
    with open(ann_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"  Combined: {len(all_images)} images, {len(all_annotations)} annotations")
    print(f"\nCropped data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cropped training data for UltraSAM from ABUS")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed ABUS data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cropped data")
    parser.add_argument("--crop_expand", type=float, default=0.5,
                        help="Expansion ratio for crop regions")
    parser.add_argument("--min_crop_size", type=int, default=64,
                        help="Minimum crop dimension")
    parser.add_argument("--square", action="store_true",
                        help="Force square crops")
    parser.add_argument("--aspect_ratio", type=float, default=None,
                        help="Fixed W/H aspect ratio (overrides --square)")
    args = parser.parse_args()

    fixed_aspect_ratio = args.aspect_ratio
    if fixed_aspect_ratio is None and args.square:
        fixed_aspect_ratio = 1.0

    generate_crop_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        crop_expand=args.crop_expand,
        min_crop_size=args.min_crop_size,
        fixed_aspect_ratio=fixed_aspect_ratio,
    )


if __name__ == "__main__":
    main()
