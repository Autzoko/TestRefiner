"""Preprocess ABUS 3D dataset to 2D slices.

This script extracts 2D slices from 3D ABUS volumes. For each 3D volume,
only the slice with the largest lesion mask area is retained. Optionally,
neighboring slices above and below the max slice can also be saved.

The output format is compatible with the UltraRefiner pipeline for
TransUNet training and UltraSAM finetuning.

Data format:
    ABUS/
        Train/
            DATA/DATA_XXX.nrrd
            MASK/MASK_XXX.nrrd
            labels.csv
        Validation/
            ...
        Test/
            ...

Output format:
    outputs/preprocessed/abus/
        train/
            *.npz (image, label, original_size)
            images_fullres/*.png
        val/
            *.npz
            images_fullres/*.png
        test/
            *.npz
            images_fullres/*.png

Usage:
    # Only save the slice with maximum mask area (default)
    python pipeline/01_preprocess_abus.py \
        --data_dir ABUS/ABUS \
        --output_dir outputs/preprocessed/abus

    # Save max slice + 2 neighbors above and below (total 5 slices)
    python pipeline/01_preprocess_abus.py \
        --data_dir ABUS/ABUS \
        --output_dir outputs/preprocessed/abus \
        --neighbors 2

    # Save max slice + 5 neighbors (total 11 slices)
    python pipeline/01_preprocess_abus.py \
        --data_dir ABUS/ABUS \
        --output_dir outputs/preprocessed/abus \
        --neighbors 5
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def read_nrrd(filepath: str) -> np.ndarray:
    """Read NRRD file and return numpy array."""
    try:
        import nrrd
        data, _ = nrrd.read(filepath)
        return data
    except ImportError:
        raise ImportError("Please install pynrrd: pip install pynrrd")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range."""
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min()) * 255
    else:
        image = np.zeros_like(image)
    return image.astype(np.uint8)


def find_max_mask_slice(mask_3d: np.ndarray) -> Tuple[int, List[int]]:
    """Find the slice with the largest mask area.

    Args:
        mask_3d: 3D mask volume (H, W, D) or (D, H, W)

    Returns:
        max_slice_idx: Index of slice with maximum mask area
        slice_areas: List of mask areas for each slice
    """
    num_slices = mask_3d.shape[2]
    slice_areas = []

    for i in range(num_slices):
        mask_slice = mask_3d[:, :, i]
        area = np.sum(mask_slice > 0)
        slice_areas.append(area)

    max_slice_idx = int(np.argmax(slice_areas))
    return max_slice_idx, slice_areas


def get_neighbor_slices(
    max_idx: int,
    num_slices: int,
    neighbors: int,
    slice_areas: List[int]
) -> List[int]:
    """Get indices of max slice and its neighbors.

    Args:
        max_idx: Index of the slice with maximum mask area
        num_slices: Total number of slices in the volume
        neighbors: Number of neighbors above and below to include
        slice_areas: List of mask areas for each slice

    Returns:
        List of slice indices to save (only those with non-zero mask)
    """
    if neighbors == 0:
        return [max_idx]

    # Get range of indices
    start_idx = max(0, max_idx - neighbors)
    end_idx = min(num_slices - 1, max_idx + neighbors)

    # Include all slices in range that have non-zero mask
    indices = []
    for i in range(start_idx, end_idx + 1):
        if slice_areas[i] > 0:
            indices.append(i)

    return indices


def process_volume(
    data_path: str,
    mask_path: str,
    output_npz_dir: str,
    output_img_dir: str,
    case_id: int,
    label: str,
    neighbors: int = 0,
) -> Dict:
    """Process a single 3D volume and save 2D slices.

    Args:
        data_path: Path to 3D data NRRD file
        mask_path: Path to 3D mask NRRD file
        output_npz_dir: Directory for NPZ output files
        output_img_dir: Directory for PNG image files
        case_id: Case ID number
        label: Label ('M' for malignant, 'B' for benign)
        neighbors: Number of neighbor slices to save (0 = only max slice)

    Returns:
        Dict with processing statistics
    """
    # Read 3D volumes
    try:
        data_3d = read_nrrd(data_path)
        mask_3d = read_nrrd(mask_path)
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return {"saved": 0, "max_area": 0, "max_idx": -1}

    # Check shape consistency
    if data_3d.shape != mask_3d.shape:
        print(f"Warning: Shape mismatch for case {case_id}: "
              f"data {data_3d.shape} vs mask {mask_3d.shape}")
        return {"saved": 0, "max_area": 0, "max_idx": -1}

    num_slices = data_3d.shape[2]

    # Find slice with maximum mask area
    max_idx, slice_areas = find_max_mask_slice(mask_3d)
    max_area = slice_areas[max_idx]

    if max_area == 0:
        print(f"Warning: No mask found for case {case_id}")
        return {"saved": 0, "max_area": 0, "max_idx": max_idx}

    # Get indices to save
    indices_to_save = get_neighbor_slices(max_idx, num_slices, neighbors, slice_areas)

    # Determine category
    category = "malignant" if label == "M" else "benign"

    saved_count = 0
    for slice_idx in indices_to_save:
        # Extract 2D slice
        data_slice = data_3d[:, :, slice_idx]
        mask_slice = mask_3d[:, :, slice_idx]

        # Skip if mask is empty (shouldn't happen due to filtering, but just in case)
        if mask_slice.max() == 0:
            continue

        # Normalize image
        data_norm = normalize_image(data_slice)

        # Convert grayscale to RGB (for consistency with BUSI format)
        data_rgb = cv2.cvtColor(data_norm, cv2.COLOR_GRAY2RGB)

        # Binary mask
        mask_binary = (mask_slice > 0).astype(np.uint8)

        # Build case name
        # Format: {category}_{case_id:03d}_slice{slice_idx:03d}
        # Mark the max slice specially
        if slice_idx == max_idx:
            case_name = f"{category}_{case_id:03d}_max"
        else:
            offset = slice_idx - max_idx
            sign = "p" if offset > 0 else "n"
            case_name = f"{category}_{case_id:03d}_{sign}{abs(offset):02d}"

        h, w = data_rgb.shape[:2]

        # Save NPZ
        npz_path = os.path.join(output_npz_dir, f"{case_name}.npz")
        np.savez(
            npz_path,
            image=data_rgb,
            label=mask_binary,
            original_size=np.array([h, w]),
        )

        # Save full-res PNG
        png_path = os.path.join(output_img_dir, f"{case_name}.png")
        cv2.imwrite(png_path, cv2.cvtColor(data_rgb, cv2.COLOR_RGB2BGR))

        saved_count += 1

    return {
        "saved": saved_count,
        "max_area": max_area,
        "max_idx": max_idx,
        "total_slices": num_slices,
    }


def process_split(
    abus_root: str,
    output_root: str,
    split_name: str,
    neighbors: int = 0,
) -> Dict:
    """Process a single data split (Train/Validation/Test).

    Args:
        abus_root: Root directory of ABUS dataset
        output_root: Root directory for output
        split_name: Name of the split (Train, Validation, Test)
        neighbors: Number of neighbor slices to save

    Returns:
        Dict with processing statistics
    """
    # Map split names
    output_split_map = {
        "Train": "train",
        "Validation": "val",
        "Test": "test",
    }
    output_split = output_split_map.get(split_name, split_name.lower())

    # Read labels
    labels_path = os.path.join(abus_root, split_name, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"Warning: labels.csv not found for {split_name}")
        return {"cases": 0, "slices": 0}

    labels_df = pd.read_csv(labels_path)

    # Create output directories
    npz_dir = os.path.join(output_root, output_split)
    img_dir = os.path.join(output_root, output_split, "images_fullres")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    stats = {
        "cases": 0,
        "slices": 0,
        "malignant": 0,
        "benign": 0,
    }

    print(f"\nProcessing {split_name} ({len(labels_df)} cases)...")

    for idx, row in labels_df.iterrows():
        case_id = row["case_id"]
        label = row["label"]

        # Handle Windows-style paths in CSV
        data_rel_path = row["data_path"].replace("\\", "/")
        mask_rel_path = row["mask_path"].replace("\\", "/")

        data_path = os.path.join(abus_root, split_name, data_rel_path)
        mask_path = os.path.join(abus_root, split_name, mask_rel_path)

        if not os.path.exists(data_path):
            print(f"  Warning: Data file not found: {data_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"  Warning: Mask file not found: {mask_path}")
            continue

        result = process_volume(
            data_path, mask_path, npz_dir, img_dir,
            case_id, label, neighbors
        )

        if result["saved"] > 0:
            stats["cases"] += 1
            stats["slices"] += result["saved"]
            if label == "M":
                stats["malignant"] += result["saved"]
            else:
                stats["benign"] += result["saved"]

            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(labels_df)} cases...")

    print(f"  {split_name}: {stats['cases']} cases, {stats['slices']} slices "
          f"(M:{stats['malignant']}, B:{stats['benign']})")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ABUS 3D dataset to 2D slices (max area slice only)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to ABUS root directory (containing Train/Validation/Test)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--neighbors", type=int, default=0,
        help="Number of neighbor slices to save above/below max slice "
             "(default: 0, only max slice)"
    )
    args = parser.parse_args()

    print("="*60)
    print("ABUS 3D to 2D Preprocessing")
    print("="*60)
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Neighbors: {args.neighbors} (total slices per case: "
          f"{1 if args.neighbors == 0 else f'up to {2*args.neighbors + 1}'})")
    print("="*60)

    # Check input directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Input directory not found: {args.data_dir}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    splits = ["Train", "Validation", "Test"]
    total_stats = {"cases": 0, "slices": 0}

    for split in splits:
        split_path = os.path.join(args.data_dir, split)
        if os.path.isdir(split_path):
            stats = process_split(
                args.data_dir, args.output_dir, split, args.neighbors
            )
            total_stats["cases"] += stats["cases"]
            total_stats["slices"] += stats["slices"]
        else:
            print(f"Warning: Split directory not found: {split_path}")

    print("\n" + "="*60)
    print("Preprocessing Summary")
    print("="*60)
    print(f"Total cases processed: {total_stats['cases']}")
    print(f"Total 2D slices saved: {total_stats['slices']}")
    print(f"\nOutput saved to: {args.output_dir}")
    print("  - NPZ files: {split}/*.npz")
    print("  - PNG images: {split}/images_fullres/*.png")
    print("\nNext steps:")
    print("  1. Train TransUNet: python pipeline/02_train_transunet_abus.py ...")
    print("  2. Generate crops: python pipeline/10_generate_crop_data_abus.py ...")


if __name__ == "__main__":
    main()
