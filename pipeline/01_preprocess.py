"""Convert raw ultrasound PNG datasets to npz format.

Supported datasets:
  - busi: BUSI dataset with benign/, malignant/ subdirs containing *_mask*.png
  - busbra: BUSBRA dataset with Images/ and Masks/ subdirs
  - abus: ABUS dataset with Train/, Validation/, Test/ subdirs (3D->2D slices)

Usage:
    python pipeline/01_preprocess.py \
        --dataset busi \
        --data_dir /path/to/Dataset_BUSI_with_GT \
        --output_dir outputs/preprocessed/busi

    python pipeline/01_preprocess.py \
        --dataset abus \
        --data_dir /path/to/ABUS \
        --output_dir outputs/preprocessed/abus
"""

import argparse
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np


def preprocess_busi(data_dir: str, output_dir: str) -> None:
    """Preprocess BUSI dataset.

    Structure: data_dir/{benign,malignant}/<case> (<id>).png + <case> (<id>)_mask*.png
    Skips 'normal' subdirectory.
    Merges multiple masks via logical OR.
    """
    npz_dir = output_dir
    img_dir = os.path.join(output_dir, "images_fullres")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    count = 0
    for category in ["benign", "malignant"]:
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"Warning: {cat_dir} not found, skipping.")
            continue

        # Find all non-mask images
        all_files = sorted(glob(os.path.join(cat_dir, "*.png")))
        image_files = [f for f in all_files if "_mask" not in os.path.basename(f)]

        for img_path in image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            case_name = f"{category}_{basename}".replace(" ", "_")

            # Read image
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Find and merge masks
            mask_pattern = os.path.join(
                cat_dir, f"{basename}_mask*.png"
            )
            mask_files = sorted(glob(mask_pattern))
            if not mask_files:
                print(f"Warning: No masks found for {img_path}, skipping.")
                continue

            combined_mask = np.zeros(image.shape[:2], dtype=bool)
            for mf in mask_files:
                m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    combined_mask |= (m > 127)

            label = combined_mask.astype(np.uint8)
            h, w = image.shape[:2]

            # Save npz
            np.savez(
                os.path.join(npz_dir, f"{case_name}.npz"),
                image=image,
                label=label,
                original_size=np.array([h, w]),
            )

            # Save full-res PNG
            cv2.imwrite(
                os.path.join(img_dir, f"{case_name}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            count += 1

    print(f"BUSI preprocessing complete: {count} samples saved to {output_dir}")


def preprocess_busbra(data_dir: str, output_dir: str) -> None:
    """Preprocess BUSBRA dataset.

    Structure: data_dir/Images/<id>.png, data_dir/Masks/<id>.png
    """
    npz_dir = output_dir
    img_dir = os.path.join(output_dir, "images_fullres")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")

    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        print(f"Error: Expected Images/ and Masks/ subdirs in {data_dir}")
        return

    image_files = sorted(glob(os.path.join(images_dir, "*.png")))
    count = 0
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{basename}.png")

        if not os.path.exists(mask_path):
            print(f"Warning: No mask for {basename}, skipping.")
            continue

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = (mask > 127).astype(np.uint8)
        h, w = image.shape[:2]

        np.savez(
            os.path.join(npz_dir, f"{basename}.npz"),
            image=image,
            label=label,
            original_size=np.array([h, w]),
        )
        cv2.imwrite(
            os.path.join(img_dir, f"{basename}.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        count += 1

    print(f"BUSBRA preprocessing complete: {count} samples saved to {output_dir}")


def preprocess_abus(data_dir: str, output_dir: str) -> None:
    """Preprocess ABUS dataset (3D->2D slices).

    Structure: data_dir/{Train,Validation,Test}/<case>_<slice>.png + <case>_<slice>_mask.png

    Preserves the predefined train/val/test splits.
    Output structure:
        output_dir/
            train/
                *.npz
                images_fullres/*.png
            val/
                *.npz
                images_fullres/*.png
            test/
                *.npz
                images_fullres/*.png
    """
    split_mapping = {
        "Train": "train",
        "Validation": "val",
        "Test": "test",
    }

    total_count = 0
    for src_split, dst_split in split_mapping.items():
        split_dir = os.path.join(data_dir, src_split)
        if not os.path.isdir(split_dir):
            print(f"Warning: {split_dir} not found, skipping.")
            continue

        # Create output directories for this split
        npz_dir = os.path.join(output_dir, dst_split)
        img_dir = os.path.join(output_dir, dst_split, "images_fullres")
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # Find all non-mask images
        all_files = sorted(glob(os.path.join(split_dir, "*.png")))
        image_files = [f for f in all_files if "_mask" not in os.path.basename(f)]

        count = 0
        for img_path in image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # Construct mask path: <basename>_mask.png
            mask_path = os.path.join(split_dir, f"{basename}_mask.png")

            if not os.path.exists(mask_path):
                print(f"Warning: No mask for {basename}, skipping.")
                continue

            # Read image
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}, skipping.")
                continue
            label = (mask > 127).astype(np.uint8)
            h, w = image.shape[:2]

            # Clean case name (replace spaces and special chars)
            case_name = basename.replace(" ", "_").replace("(", "").replace(")", "")

            # Save npz
            np.savez(
                os.path.join(npz_dir, f"{case_name}.npz"),
                image=image,
                label=label,
                original_size=np.array([h, w]),
            )

            # Save full-res PNG
            cv2.imwrite(
                os.path.join(img_dir, f"{case_name}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            count += 1

        print(f"  {src_split}: {count} samples")
        total_count += count

    print(f"ABUS preprocessing complete: {total_count} samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ultrasound datasets to npz")
    parser.add_argument("--dataset", type=str, required=True, choices=["busi", "busbra", "abus"],
                        help="Dataset type")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to raw dataset root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for npz files")
    args = parser.parse_args()

    if args.dataset == "busi":
        preprocess_busi(args.data_dir, args.output_dir)
    elif args.dataset == "busbra":
        preprocess_busbra(args.data_dir, args.output_dir)
    elif args.dataset == "abus":
        preprocess_abus(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
