"""Analyze how point prompt quality affects UltraSAM inference results.

This script performs deep analysis on point prompts:
I. Compare TransUNet-generated vs GT-generated point prompts:
   - Calculate portion of TransUNet points inside GT lesion area
   - Visualize both points on same image (different colors)
   - Show Dice scores comparison
   - Calculate distance between point prompts
   - Overlay predictions with GT mask

II. Perturbation study:
   - Add controlled disturbance to GT point prompts
   - Measure how point quality affects UltraSAM performance

Usage:
    python pipeline/11_analyze_point_prompts.py \
        --data_dir outputs/preprocessed/busi \
        --transunet_pred_dir outputs/transunet_preds/busi \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/analysis/point_prompts/busi \
        --device cuda:0
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)

from utils.prompt_utils import mask_to_bbox, mask_to_centroid, transform_coords
from utils.metrics import compute_dice, compute_iou


def load_ultrasam_model(config_path, ckpt_path, device):
    """Build and load UltraSAM model."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS

    register_all_modules()
    cfg = Config.fromfile(config_path)

    if hasattr(cfg, "custom_imports"):
        import importlib
        for mod_name in cfg.custom_imports.get("imports", []):
            try:
                importlib.import_module(mod_name)
            except ImportError:
                pass

    try:
        from endosam.models.utils.custom_functional import (
            multi_head_attention_forward as custom_mha_forward,
        )
        torch.nn.functional.multi_head_attention_forward = custom_mha_forward
    except ImportError:
        pass

    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


def run_ultrasam_point_inference(model, image_bgr, point, device, target_size=1024):
    """Run UltraSAM inference with a single point prompt."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    orig_h, orig_w = image_bgr.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    img_tensor = torch.tensor(padded.astype(np.float32).transpose(2, 0, 1).copy())

    # Transform point to 1024 space
    point_1024 = transform_coords(point, (orig_h, orig_w), target_size, is_box=False)
    px, py = point_1024

    data_sample = DetDataSample()
    gt_instances = InstanceData()

    gt_instances.points = torch.tensor([[[px, py]]], dtype=torch.float32, device=device)
    gt_instances.boxes = torch.tensor(
        [[[0.0, 0.0], [float(target_size), float(target_size)]]],
        dtype=torch.float32, device=device)
    gt_instances.labels = torch.tensor([0], dtype=torch.long, device=device)
    gt_instances.prompt_types = torch.tensor([0], dtype=torch.long, device=device)

    data_sample.gt_instances = gt_instances
    data_sample.set_metainfo({
        "img_shape": (new_h, new_w),
        "ori_shape": (orig_h, orig_w),
        "scale_factor": (new_w / orig_w, new_h / orig_h),
        "pad_shape": (target_size, target_size),
        "batch_input_shape": (target_size, target_size),
    })

    with torch.no_grad():
        data = {"inputs": [img_tensor], "data_samples": [data_sample]}
        data = model.data_preprocessor(data, False)
        results = model(**data, mode="predict")

    result = results[0]
    if hasattr(result, "pred_instances") and hasattr(result.pred_instances, "masks"):
        if len(result.pred_instances.masks) > 0:
            mask = np.array(result.pred_instances.masks[0].cpu().tolist())
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
            return (mask > 0).astype(np.uint8)

    return np.zeros((orig_h, orig_w), dtype=np.uint8)


def compute_point_distance(p1: List[int], p2: List[int]) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_point_in_mask(point: List[int], mask: np.ndarray) -> bool:
    """Check if a point is inside the mask region."""
    x, y = int(point[0]), int(point[1])
    h, w = mask.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return mask[y, x] > 0
    return False


def add_point_perturbation(
    point: List[int],
    mask: np.ndarray,
    perturbation_type: str = "gaussian",
    sigma: float = 10.0,
    direction: str = "random",
) -> List[int]:
    """Add perturbation to a point prompt.

    Args:
        point: Original point [x, y]
        mask: GT mask for bounds checking
        perturbation_type: "gaussian", "uniform", "directional"
        sigma: Standard deviation for gaussian, or max displacement for uniform
        direction: For directional: "outward", "inward", "random"

    Returns:
        Perturbed point [x, y]
    """
    h, w = mask.shape[:2]
    x, y = point

    if perturbation_type == "gaussian":
        dx = np.random.normal(0, sigma)
        dy = np.random.normal(0, sigma)
    elif perturbation_type == "uniform":
        dx = np.random.uniform(-sigma, sigma)
        dy = np.random.uniform(-sigma, sigma)
    elif perturbation_type == "directional":
        # Move toward or away from mask centroid
        centroid = mask_to_centroid(mask)
        cx, cy = centroid

        # Direction vector from point to centroid
        vec_x = cx - x
        vec_y = cy - y
        length = np.sqrt(vec_x**2 + vec_y**2)

        if length > 0:
            vec_x /= length
            vec_y /= length

            if direction == "outward":
                dx = -vec_x * sigma
                dy = -vec_y * sigma
            elif direction == "inward":
                dx = vec_x * sigma
                dy = vec_y * sigma
            else:  # random
                angle = np.random.uniform(0, 2 * np.pi)
                dx = np.cos(angle) * sigma
                dy = np.sin(angle) * sigma
        else:
            dx, dy = 0, 0
    else:
        dx, dy = 0, 0

    # Apply perturbation and clamp to image bounds
    new_x = int(np.clip(x + dx, 0, w - 1))
    new_y = int(np.clip(y + dy, 0, h - 1))

    return [new_x, new_y]


def create_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    gt_point: List[int],
    transunet_point: List[int],
    pred_gt_point: np.ndarray,
    pred_transunet_point: np.ndarray,
    case_name: str,
    dice_gt: float,
    dice_transunet: float,
    distance: float,
    point_in_lesion: bool,
    output_path: str,
):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: Image with both points
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.scatter([gt_point[0]], [gt_point[1]], c='green', s=200, marker='*',
               label=f'GT Point', edgecolors='white', linewidths=2)
    ax.scatter([transunet_point[0]], [transunet_point[1]], c='red', s=200, marker='o',
               label=f'TransUNet Point', edgecolors='white', linewidths=2)
    # Draw line between points
    ax.plot([gt_point[0], transunet_point[0]], [gt_point[1], transunet_point[1]],
            'y--', linewidth=2, alpha=0.7)
    ax.set_title(f'Points Comparison\nDistance: {distance:.1f}px, TransUNet in lesion: {point_in_lesion}')
    ax.legend(loc='upper right')
    ax.axis('off')

    # Panel 2: GT mask with points
    ax = axes[0, 1]
    ax.imshow(gt_mask, cmap='gray')
    ax.scatter([gt_point[0]], [gt_point[1]], c='green', s=200, marker='*',
               edgecolors='white', linewidths=2)
    ax.scatter([transunet_point[0]], [transunet_point[1]], c='red', s=200, marker='o',
               edgecolors='white', linewidths=2)
    ax.set_title('GT Mask with Points')
    ax.axis('off')

    # Panel 3: Dice comparison bar chart
    ax = axes[0, 2]
    bars = ax.bar(['GT Point', 'TransUNet Point'], [dice_gt, dice_transunet],
                  color=['green', 'red'], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Dice Score')
    ax.set_title(f'Dice Comparison\nDiff: {dice_gt - dice_transunet:.4f}')
    for bar, val in zip(bars, [dice_gt, dice_transunet]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', fontsize=10)

    # Panel 4: Prediction from GT point
    ax = axes[1, 0]
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # GT mask in blue
    overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    # Prediction in green
    overlay[pred_gt_point > 0] = overlay[pred_gt_point > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ax.imshow(overlay.astype(np.uint8))
    ax.scatter([gt_point[0]], [gt_point[1]], c='green', s=200, marker='*',
               edgecolors='white', linewidths=2)
    ax.set_title(f'GT Point Prediction (Dice: {dice_gt:.4f})\nBlue=GT, Green=Pred')
    ax.axis('off')

    # Panel 5: Prediction from TransUNet point
    ax = axes[1, 1]
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    overlay[pred_transunet_point > 0] = overlay[pred_transunet_point > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    ax.imshow(overlay.astype(np.uint8))
    ax.scatter([transunet_point[0]], [transunet_point[1]], c='red', s=200, marker='o',
               edgecolors='white', linewidths=2)
    ax.set_title(f'TransUNet Point Prediction (Dice: {dice_transunet:.4f})\nBlue=GT, Red=Pred')
    ax.axis('off')

    # Panel 6: Overlay all three masks
    ax = axes[1, 2]
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # Create RGB overlay: GT=Blue, GT_pred=Green, TransUNet_pred=Red
    mask_overlay = np.zeros_like(overlay, dtype=np.float32)
    mask_overlay[gt_mask > 0, 2] = 1.0  # Blue for GT
    mask_overlay[pred_gt_point > 0, 1] = 1.0  # Green for GT point pred
    mask_overlay[pred_transunet_point > 0, 0] = 1.0  # Red for TransUNet point pred
    overlay = overlay * 0.4 + (mask_overlay * 255 * 0.6)
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title('Overlay: Blue=GT, Green=GT_pred, Red=TransUNet_pred')
    ax.axis('off')

    plt.suptitle(f'{case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_perturbation_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    original_point: List[int],
    perturbed_points: List[Tuple[List[int], float, float]],  # (point, sigma, dice)
    predictions: List[np.ndarray],
    case_name: str,
    output_path: str,
):
    """Create visualization for perturbation study."""
    n_perturbations = len(perturbed_points)
    n_cols = min(4, n_perturbations + 1)
    n_rows = (n_perturbations + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Original point
    ax = axes[0, 0]
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    overlay[predictions[0] > 0] = overlay[predictions[0] > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ax.imshow(overlay.astype(np.uint8))
    ax.scatter([original_point[0]], [original_point[1]], c='green', s=200, marker='*',
               edgecolors='white', linewidths=2)
    original_dice = compute_dice(predictions[0], gt_mask)
    ax.set_title(f'Original (σ=0)\nDice: {original_dice:.4f}')
    ax.axis('off')

    # Perturbed points
    for i, ((point, sigma, dice), pred) in enumerate(zip(perturbed_points, predictions[1:])):
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        ax = axes[row, col]

        overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        overlay[gt_mask > 0] = overlay[gt_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
        overlay[pred > 0] = overlay[pred > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        ax.imshow(overlay.astype(np.uint8))
        ax.scatter([original_point[0]], [original_point[1]], c='green', s=100, marker='*',
                   edgecolors='white', linewidths=1, alpha=0.5)
        ax.scatter([point[0]], [point[1]], c='red', s=200, marker='o',
                   edgecolors='white', linewidths=2)
        # Draw displacement line
        ax.plot([original_point[0], point[0]], [original_point[1], point[1]],
                'y-', linewidth=2, alpha=0.7)
        dist = compute_point_distance(original_point, point)
        ax.set_title(f'σ={sigma:.0f}px, dist={dist:.1f}px\nDice: {dice:.4f} (Δ={dice-original_dice:+.4f})')
        ax.axis('off')

    # Hide empty subplots
    for i in range(len(perturbed_points) + 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.suptitle(f'Perturbation Study: {case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_point_prompts(
    data_dir: str,
    transunet_pred_dir: str,
    ultrasam_config: str,
    ultrasam_ckpt: str,
    output_dir: str,
    device: str = "cuda:0",
    max_samples: int = None,
    perturbation_sigmas: List[float] = [5, 10, 20, 30, 50],
):
    """Main analysis function."""
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    perturbation_dir = os.path.join(output_dir, "perturbation_study")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(perturbation_dir, exist_ok=True)

    # Load UltraSAM model
    print("Loading UltraSAM model...")
    model = load_ultrasam_model(ultrasam_config, ultrasam_ckpt, device)

    # Find all folds
    fold_dirs = sorted(glob(os.path.join(transunet_pred_dir, "fold_*")))
    if not fold_dirs:
        print(f"Error: No fold directories found in {transunet_pred_dir}")
        print(f"  Looking in: {transunet_pred_dir}")
        print(f"  Contents: {os.listdir(transunet_pred_dir) if os.path.exists(transunet_pred_dir) else 'DIR NOT FOUND'}")
        return

    all_results = []
    perturbation_results = {sigma: [] for sigma in perturbation_sigmas}
    skipped_counts = {"no_gt": 0, "no_image": 0, "load_failed": 0}

    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        print(f"\n=== Processing {fold_name} ===")

        gt_dir = os.path.join(fold_dir, "gt")
        pred_dir = fold_dir  # TransUNet predictions

        # Check directories exist
        if not os.path.exists(gt_dir):
            print(f"  Warning: GT directory not found: {gt_dir}")
            continue

        # Find prediction files
        pred_files = sorted(glob(os.path.join(pred_dir, "*_pred.png")))
        print(f"  Found {len(pred_files)} prediction files")

        if len(pred_files) == 0:
            print(f"  Warning: No *_pred.png files in {pred_dir}")
            continue

        for i, pred_file in enumerate(pred_files):
            if max_samples and i >= max_samples:
                break

            case_name = os.path.basename(pred_file).replace("_pred.png", "")
            gt_file = os.path.join(gt_dir, f"{case_name}_gt.png")

            if not os.path.exists(gt_file):
                skipped_counts["no_gt"] += 1
                if skipped_counts["no_gt"] <= 3:
                    print(f"  Warning: GT not found: {gt_file}")
                continue

            # Load GT mask and TransUNet prediction
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                skipped_counts["load_failed"] += 1
                continue
            gt_mask = (gt_mask > 127).astype(np.uint8)

            transunet_pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
            if transunet_pred is None:
                skipped_counts["load_failed"] += 1
                continue
            transunet_pred = (transunet_pred > 127).astype(np.uint8)

            # Load original image - try PNG first, fall back to npz
            img_path = os.path.join(data_dir, "images_fullres", f"{case_name}.png")
            npz_path = os.path.join(data_dir, f"{case_name}.npz")

            image_bgr = None
            if os.path.exists(img_path):
                image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            elif os.path.exists(npz_path):
                # Load from npz file
                try:
                    npz_data = np.load(npz_path)
                    image_rgb = npz_data["image"]  # RGB format
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    if skipped_counts["load_failed"] <= 3:
                        print(f"  Warning: Failed to load npz {npz_path}: {e}")
                    skipped_counts["load_failed"] += 1
                    continue
            else:
                skipped_counts["no_image"] += 1
                if skipped_counts["no_image"] <= 3:
                    print(f"  Warning: Image not found: {img_path} or {npz_path}")
                continue

            if image_bgr is None:
                skipped_counts["load_failed"] += 1
                continue

            # Generate points
            gt_point = mask_to_centroid(gt_mask)
            transunet_point = mask_to_centroid(transunet_pred)

            # Check if TransUNet point is in lesion
            point_in_lesion = is_point_in_mask(transunet_point, gt_mask)

            # Calculate distance
            distance = compute_point_distance(gt_point, transunet_point)

            # Run UltraSAM inference with both points
            pred_gt_point = run_ultrasam_point_inference(model, image_bgr, gt_point, device)
            pred_transunet_point = run_ultrasam_point_inference(model, image_bgr, transunet_point, device)

            # Compute Dice scores
            dice_gt = compute_dice(pred_gt_point, gt_mask)
            dice_transunet = compute_dice(pred_transunet_point, gt_mask)

            # Store results
            result = {
                "case": case_name,
                "fold": fold_name,
                "gt_point": gt_point,
                "transunet_point": transunet_point,
                "point_in_lesion": point_in_lesion,
                "distance": distance,
                "dice_gt_point": dice_gt,
                "dice_transunet_point": dice_transunet,
                "dice_diff": dice_gt - dice_transunet,
            }
            all_results.append(result)

            # Create visualization
            vis_path = os.path.join(vis_dir, f"{fold_name}_{case_name}.png")
            create_visualization(
                image_bgr, gt_mask, gt_point, transunet_point,
                pred_gt_point, pred_transunet_point,
                case_name, dice_gt, dice_transunet, distance,
                point_in_lesion, vis_path
            )

            # Perturbation study
            perturbed_data = []
            predictions = [pred_gt_point]

            for sigma in perturbation_sigmas:
                perturbed_point = add_point_perturbation(
                    gt_point, gt_mask, perturbation_type="gaussian", sigma=sigma
                )
                pred_perturbed = run_ultrasam_point_inference(
                    model, image_bgr, perturbed_point, device
                )
                dice_perturbed = compute_dice(pred_perturbed, gt_mask)

                perturbed_data.append((perturbed_point, sigma, dice_perturbed))
                predictions.append(pred_perturbed)

                perturbation_results[sigma].append({
                    "case": case_name,
                    "original_dice": dice_gt,
                    "perturbed_dice": dice_perturbed,
                    "dice_drop": dice_gt - dice_perturbed,
                    "actual_distance": compute_point_distance(gt_point, perturbed_point),
                })

            # Create perturbation visualization
            perturbation_vis_path = os.path.join(perturbation_dir, f"{fold_name}_{case_name}.png")
            create_perturbation_visualization(
                image_bgr, gt_mask, gt_point, perturbed_data,
                predictions, case_name, perturbation_vis_path
            )

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(pred_files)}")

    # Print skip summary
    print(f"\nSkipped samples:")
    print(f"  No GT file: {skipped_counts['no_gt']}")
    print(f"  No image file: {skipped_counts['no_image']}")
    print(f"  Load failed: {skipped_counts['load_failed']}")

    # Save detailed results
    results_path = os.path.join(output_dir, "point_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Compute and save summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    # Part I: TransUNet vs GT Point Analysis
    n_samples = len(all_results)

    if n_samples == 0:
        print("\nERROR: No samples were processed!")
        print("Please check:")
        print(f"  1. TransUNet predictions exist in: {transunet_pred_dir}/fold_*/")
        print(f"  2. GT masks exist in: {transunet_pred_dir}/fold_*/gt/")
        print(f"  3. Images exist in: {data_dir}/images_fullres/*.png OR {data_dir}/*.npz")
        print("\nExpected file patterns:")
        print("  - Predictions: fold_*/case_name_pred.png")
        print("  - GT masks: fold_*/gt/case_name_gt.png")
        print("  - Images: images_fullres/case_name.png OR case_name.npz (with 'image' key)")
        return

    n_in_lesion = sum(1 for r in all_results if r["point_in_lesion"])
    avg_distance = np.mean([r["distance"] for r in all_results])
    avg_dice_gt = np.mean([r["dice_gt_point"] for r in all_results])
    avg_dice_transunet = np.mean([r["dice_transunet_point"] for r in all_results])
    avg_dice_diff = np.mean([r["dice_diff"] for r in all_results])

    print(f"\nPart I: TransUNet vs GT Point Comparison")
    print(f"-" * 40)
    print(f"Total samples: {n_samples}")
    print(f"TransUNet points in lesion: {n_in_lesion}/{n_samples} ({100*n_in_lesion/n_samples:.1f}%)")
    print(f"Average point distance: {avg_distance:.2f} pixels")
    print(f"\nDice Scores:")
    print(f"  GT Point:       {avg_dice_gt:.4f} ± {np.std([r['dice_gt_point'] for r in all_results]):.4f}")
    print(f"  TransUNet Point: {avg_dice_transunet:.4f} ± {np.std([r['dice_transunet_point'] for r in all_results]):.4f}")
    print(f"  Difference:      {avg_dice_diff:.4f} (GT better)")

    # Stratified analysis: in-lesion vs out-of-lesion
    in_lesion = [r for r in all_results if r["point_in_lesion"]]
    out_lesion = [r for r in all_results if not r["point_in_lesion"]]

    if in_lesion:
        print(f"\nWhen TransUNet point IS in lesion ({len(in_lesion)} samples):")
        print(f"  Dice diff: {np.mean([r['dice_diff'] for r in in_lesion]):.4f}")
    if out_lesion:
        print(f"When TransUNet point is OUTSIDE lesion ({len(out_lesion)} samples):")
        print(f"  Dice diff: {np.mean([r['dice_diff'] for r in out_lesion]):.4f}")

    # Part II: Perturbation Analysis
    print(f"\nPart II: Perturbation Study")
    print(f"-" * 40)
    print(f"{'Sigma (px)':<12} {'Avg Dice':<12} {'Avg Drop':<12} {'Avg Dist':<12}")
    print(f"{'-'*48}")

    perturbation_summary = []
    for sigma in perturbation_sigmas:
        results = perturbation_results[sigma]
        avg_dice = np.mean([r["perturbed_dice"] for r in results])
        avg_drop = np.mean([r["dice_drop"] for r in results])
        avg_dist = np.mean([r["actual_distance"] for r in results])
        print(f"{sigma:<12.0f} {avg_dice:<12.4f} {avg_drop:<12.4f} {avg_dist:<12.1f}")
        perturbation_summary.append({
            "sigma": sigma,
            "avg_dice": avg_dice,
            "avg_dice_drop": avg_drop,
            "avg_actual_distance": avg_dist,
        })

    # Create summary plots
    create_summary_plots(all_results, perturbation_summary, output_dir)

    # Save summary
    summary = {
        "n_samples": n_samples,
        "transunet_points_in_lesion_ratio": n_in_lesion / n_samples,
        "avg_point_distance": avg_distance,
        "avg_dice_gt_point": avg_dice_gt,
        "avg_dice_transunet_point": avg_dice_transunet,
        "avg_dice_difference": avg_dice_diff,
        "perturbation_summary": perturbation_summary,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


def create_summary_plots(results: List[Dict], perturbation_summary: List[Dict], output_dir: str):
    """Create summary visualization plots."""

    # Plot 1: Dice score distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Scatter plot: Distance vs Dice difference
    ax = axes[0, 0]
    distances = [r["distance"] for r in results]
    dice_diffs = [r["dice_diff"] for r in results]
    colors = ['green' if r["point_in_lesion"] else 'red' for r in results]
    ax.scatter(distances, dice_diffs, c=colors, alpha=0.6, s=50)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Point Distance (pixels)')
    ax.set_ylabel('Dice Difference (GT - TransUNet)')
    ax.set_title('Point Distance vs Performance Gap\nGreen=in lesion, Red=outside lesion')

    # Histogram of Dice scores
    ax = axes[0, 1]
    dice_gt = [r["dice_gt_point"] for r in results]
    dice_tu = [r["dice_transunet_point"] for r in results]
    ax.hist(dice_gt, bins=20, alpha=0.5, label='GT Point', color='green')
    ax.hist(dice_tu, bins=20, alpha=0.5, label='TransUNet Point', color='red')
    ax.set_xlabel('Dice Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Dice Score Distribution')
    ax.legend()

    # Box plot comparison
    ax = axes[1, 0]
    in_lesion = [r["dice_transunet_point"] for r in results if r["point_in_lesion"]]
    out_lesion = [r["dice_transunet_point"] for r in results if not r["point_in_lesion"]]
    gt_dice = [r["dice_gt_point"] for r in results]

    bp = ax.boxplot([gt_dice, in_lesion, out_lesion],
                    labels=['GT Point\n(all)', 'TransUNet\n(in lesion)', 'TransUNet\n(outside)'])
    ax.set_ylabel('Dice Score')
    ax.set_title('Dice by Point Location')

    # Perturbation effect
    ax = axes[1, 1]
    sigmas = [p["sigma"] for p in perturbation_summary]
    dice_drops = [p["avg_dice_drop"] for p in perturbation_summary]
    ax.plot(sigmas, dice_drops, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Perturbation σ (pixels)')
    ax.set_ylabel('Average Dice Drop')
    ax.set_title('Effect of Point Perturbation on Performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Additional plot: Perturbation curve with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    sigmas = [0] + [p["sigma"] for p in perturbation_summary]
    avg_dices = [np.mean([r["dice_gt_point"] for r in results])]
    avg_dices += [p["avg_dice"] for p in perturbation_summary]

    ax.plot(sigmas, avg_dices, 'b-o', linewidth=2, markersize=10)
    ax.fill_between(sigmas, [d - 0.05 for d in avg_dices], [d + 0.05 for d in avg_dices],
                    alpha=0.2, color='blue')
    ax.set_xlabel('Perturbation σ (pixels)', fontsize=12)
    ax.set_ylabel('Average Dice Score', fontsize=12)
    ax.set_title('Point Prompt Quality vs UltraSAM Performance', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perturbation_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze point prompt effects on UltraSAM")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed data directory")
    parser.add_argument("--transunet_pred_dir", type=str, required=True,
                        help="TransUNet predictions directory")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="UltraSAM config file")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per fold (for quick testing)")
    parser.add_argument("--perturbation_sigmas", type=str, default="5,10,20,30,50",
                        help="Comma-separated perturbation sigma values")
    args = parser.parse_args()

    # Resolve paths
    config_path = os.path.join(ROOT_DIR, args.ultrasam_config) \
        if not os.path.isabs(args.ultrasam_config) else args.ultrasam_config
    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    perturbation_sigmas = [float(s) for s in args.perturbation_sigmas.split(",")]

    analyze_point_prompts(
        data_dir=args.data_dir,
        transunet_pred_dir=args.transunet_pred_dir,
        ultrasam_config=config_path,
        ultrasam_ckpt=ckpt_path,
        output_dir=args.output_dir,
        device=args.device,
        max_samples=args.max_samples,
        perturbation_sigmas=perturbation_sigmas,
    )


if __name__ == "__main__":
    main()
