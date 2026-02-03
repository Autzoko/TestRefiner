"""Generate cropped training data for UltraSAM finetuning.

This script takes preprocessed images and GT masks, generates crop regions
around the GT masks, and creates a COCO-format dataset for UltraSAM training.

The cropped data simulates what UltraSAM sees during crop-mode inference,
allowing the model to learn features specific to cropped regions.

Usage:
    python pipeline/07_generate_crop_data.py \
        --data_dir outputs/preprocessed/busi \
        --output_dir outputs/crop_data/busi \
        --crop_expand 0.5 \
        --n_folds 5
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

from utils.prompt_utils import compute_crop_box, mask_to_bbox, mask_to_centroid


def create_coco_annotation(
    ann_id: int,
    image_id: int,
    mask: np.ndarray,
    category_id: int = 1,
) -> Optional[Dict]:
    """Create a COCO annotation from a binary mask.

    Args:
        ann_id: Unique annotation ID.
        image_id: ID of the image this annotation belongs to.
        mask: Binary mask (H, W) with 0/1 or 0/255 values.
        category_id: Category ID (default 1 for foreground).

    Returns:
        COCO annotation dict or None if mask is empty.
    """
    mask = (mask > 0).astype(np.uint8)
    if not mask.any():
        return None

    # Get bounding box
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    # Compute area
    area = int(mask.sum())

    # RLE encoding for segmentation
    from pycocotools import mask as mask_util
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")

    annotation = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, width, height],
        "area": area,
        "iscrowd": 0,
        "segmentation": rle,
    }

    return annotation


def process_sample(
    image_path: str,
    mask_path: str,
    image_id: int,
    ann_id_start: int,
    output_images_dir: str,
    crop_expand: float = 0.5,
    min_crop_size: int = 64,
) -> Tuple[Optional[Dict], List[Dict], int]:
    """Process a single sample to create cropped data.

    Args:
        image_path: Path to full-res image.
        mask_path: Path to GT mask.
        image_id: Unique ID for this image.
        ann_id_start: Starting annotation ID.
        output_images_dir: Directory to save cropped images.
        crop_expand: Expansion ratio for crop region.
        min_crop_size: Minimum crop dimension.

    Returns:
        Tuple of (image_info dict, list of annotation dicts, next ann_id).
    """
    # Load image and mask
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or mask is None:
        return None, [], ann_id_start

    mask = (mask > 127).astype(np.uint8)

    if not mask.any():
        return None, [], ann_id_start

    full_h, full_w = image_bgr.shape[:2]

    # Get bbox from GT mask
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None, [], ann_id_start

    # Compute crop region
    crop_box = compute_crop_box(bbox, full_h, full_w, expand_ratio=crop_expand,
                                 min_crop_size=min_crop_size)
    cx1, cy1, cx2, cy2 = crop_box
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1

    # Skip if crop covers nearly the entire image (>95%)
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
    n_folds: int = 5,
    min_crop_size: int = 64,
):
    """Generate cropped dataset from preprocessed data.

    Args:
        data_dir: Directory with preprocessed data (images_fullres/, masks_fullres/).
        output_dir: Output directory for cropped data.
        crop_expand: Expansion ratio for crop regions.
        n_folds: Number of cross-validation folds (for split consistency).
        min_crop_size: Minimum crop dimension.
    """
    images_dir = os.path.join(data_dir, "images_fullres")
    masks_dir = os.path.join(data_dir, "masks_fullres")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    if not os.path.exists(masks_dir):
        print(f"Error: Masks directory not found: {masks_dir}")
        return

    # Create output directories
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # Get all image files
    image_files = sorted(glob(os.path.join(images_dir, "*.png")))
    print(f"Found {len(image_files)} images")

    # Create k-fold splits (same random state as TransUNet training)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)

    case_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    case_to_idx = {name: i for i, name in enumerate(case_names)}

    # For each fold, generate train/val split
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(image_files)):
        print(f"\n=== Fold {fold_idx} ===")

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        fold_images_dir = os.path.join(fold_dir, "images")
        os.makedirs(fold_images_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "annotations"), exist_ok=True)

        # Process training set
        train_images = []
        train_annotations = []
        image_id = 1
        ann_id = 1

        for idx in train_indices:
            image_path = image_files[idx]
            case_name = case_names[idx]
            mask_path = os.path.join(masks_dir, f"{case_name}.png")

            if not os.path.exists(mask_path):
                print(f"  Warning: Mask not found for {case_name}")
                continue

            image_info, annotations, ann_id = process_sample(
                image_path, mask_path, image_id, ann_id,
                fold_images_dir, crop_expand, min_crop_size,
            )

            if image_info:
                train_images.append(image_info)
                train_annotations.extend(annotations)
                image_id += 1

        # Create COCO format for training
        train_coco = {
            "images": train_images,
            "annotations": train_annotations,
            "categories": [{"id": 1, "name": "object"}],
        }

        train_json_path = os.path.join(fold_dir, "annotations", "train.json")
        with open(train_json_path, "w") as f:
            json.dump(train_coco, f, indent=2)

        print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")

        # Process validation set
        val_images = []
        val_annotations = []
        val_image_id = 1
        val_ann_id = 1

        for idx in val_indices:
            image_path = image_files[idx]
            case_name = case_names[idx]
            mask_path = os.path.join(masks_dir, f"{case_name}.png")

            if not os.path.exists(mask_path):
                continue

            image_info, annotations, val_ann_id = process_sample(
                image_path, mask_path, val_image_id, val_ann_id,
                fold_images_dir, crop_expand, min_crop_size,
            )

            if image_info:
                val_images.append(image_info)
                val_annotations.extend(annotations)
                val_image_id += 1

        # Create COCO format for validation
        val_coco = {
            "images": val_images,
            "annotations": val_annotations,
            "categories": [{"id": 1, "name": "object"}],
        }

        val_json_path = os.path.join(fold_dir, "annotations", "val.json")
        with open(val_json_path, "w") as f:
            json.dump(val_coco, f, indent=2)

        print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")

        # Save split info
        split_info = {
            "train_cases": [case_names[i] for i in train_indices],
            "val_cases": [case_names[i] for i in val_indices],
            "crop_expand": crop_expand,
        }
        split_path = os.path.join(fold_dir, "split.json")
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

    # Also generate a combined dataset (all folds together for non-CV training)
    print("\n=== Generating combined dataset ===")
    combined_images_dir = os.path.join(output_dir, "combined", "images")
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "combined", "annotations"), exist_ok=True)

    all_images = []
    all_annotations = []
    image_id = 1
    ann_id = 1

    for image_path in image_files:
        case_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(masks_dir, f"{case_name}.png")

        if not os.path.exists(mask_path):
            continue

        image_info, annotations, ann_id = process_sample(
            image_path, mask_path, image_id, ann_id,
            combined_images_dir, crop_expand, min_crop_size,
        )

        if image_info:
            all_images.append(image_info)
            all_annotations.extend(annotations)
            image_id += 1

    combined_coco = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{"id": 1, "name": "object"}],
    }

    combined_json_path = os.path.join(output_dir, "combined", "annotations", "train.json")
    with open(combined_json_path, "w") as f:
        json.dump(combined_coco, f, indent=2)

    print(f"Combined: {len(all_images)} images, {len(all_annotations)} annotations")
    print(f"\nCropped data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate cropped training data for UltraSAM")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with preprocessed data (images_fullres/, masks_fullres/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cropped data")
    parser.add_argument("--crop_expand", type=float, default=0.5,
                        help="Expansion ratio for crop regions (default: 0.5)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--min_crop_size", type=int, default=64,
                        help="Minimum crop dimension in pixels (default: 64)")
    args = parser.parse_args()

    generate_crop_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        crop_expand=args.crop_expand,
        n_folds=args.n_folds,
        min_crop_size=args.min_crop_size,
    )


if __name__ == "__main__":
    main()
