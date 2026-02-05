"""Fine-tune UltraSAM on BUSI GT masks with perturbed box prompts.

This script trains UltraSAM directly on preprocessed npz files (ground-truth
masks) using K-fold cross-validation. During training, bounding box prompts
derived from GT masks are randomly perturbed to improve robustness to
imperfect box prompts at inference time.

Box perturbation strategies (each edge perturbed independently):
  - exact:          tight GT box, unchanged
  - slight_enlarge: each edge expanded by 5-20% of box size
  - slight_shrink:  each edge shrunk by 5-15% of box size
  - heavy_enlarge:  each edge expanded by 30-60% of box size
  - heavy_shrink:   each edge shrunk by 15-30% of box size

Usage:
    # Train all 5 folds (default perturbation ratios)
    python pipeline/08_finetune_ultrasam_gt_box.py \
        --data_dir outputs/preprocessed/busi \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam_gt_box/busi \
        --freeze_backbone \
        --epochs 50 --lr 1e-4 --device cuda:0

    # Single fold with custom perturbation ratios
    python pipeline/08_finetune_ultrasam_gt_box.py \
        --data_dir outputs/preprocessed/busi \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam_gt_box/busi \
        --fold 0 \
        --p_exact 0.20 --p_slight_enlarge 0.30 --p_slight_shrink 0.30 \
        --p_heavy_enlarge 0.10 --p_heavy_shrink 0.10 \
        --freeze_all_but_decoder_head \
        --epochs 100 --batch_size 2
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)

from utils.prompt_utils import mask_to_bbox, transform_coords


# ---------------------------------------------------------------------------
# Box perturbation
# ---------------------------------------------------------------------------

# Perturbation strategy names and default probabilities
PERTURBATION_STRATEGIES = [
    "exact",
    "slight_enlarge",
    "slight_shrink",
    "heavy_enlarge",
    "heavy_shrink",
]

DEFAULT_PERTURBATION_RATIOS = {
    "exact": 0.10,
    "slight_enlarge": 0.40,
    "slight_shrink": 0.40,
    "heavy_enlarge": 0.05,
    "heavy_shrink": 0.05,
}

# Range of perturbation magnitude (fraction of box width/height) per strategy
PERTURBATION_RANGES = {
    "slight_enlarge": (0.05, 0.20),
    "slight_shrink": (0.05, 0.15),
    "heavy_enlarge": (0.30, 0.60),
    "heavy_shrink": (0.15, 0.30),
}


def perturb_box(
    box: List[int],
    image_h: int,
    image_w: int,
    perturbation_ratios: Dict[str, float],
    rng: np.random.Generator,
    min_box_size: int = 2,
) -> Tuple[List[int], str]:
    """Perturb a bounding box with stochastic edge-level noise.

    Each edge is perturbed independently, producing asymmetric variations
    that simulate realistic detection errors.

    Args:
        box: [x1, y1, x2, y2] tight GT box in original pixel space.
        image_h: Image height for clamping.
        image_w: Image width for clamping.
        perturbation_ratios: Dict mapping strategy name to probability.
        rng: numpy random Generator.
        min_box_size: Minimum box dimension in pixels.

    Returns:
        (perturbed_box, strategy_name)
    """
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return box, "exact"

    # Sample strategy
    strategies = list(perturbation_ratios.keys())
    probs = [perturbation_ratios[s] for s in strategies]
    strategy = rng.choice(strategies, p=probs)

    if strategy == "exact":
        return box, strategy

    lo, hi = PERTURBATION_RANGES[strategy]

    # Sample independent perturbation for each edge
    dx1 = rng.uniform(lo, hi) * bw
    dy1 = rng.uniform(lo, hi) * bh
    dx2 = rng.uniform(lo, hi) * bw
    dy2 = rng.uniform(lo, hi) * bh

    if strategy in ("slight_enlarge", "heavy_enlarge"):
        # Expand: move x1/y1 outward (subtract), x2/y2 outward (add)
        nx1 = x1 - dx1
        ny1 = y1 - dy1
        nx2 = x2 + dx2
        ny2 = y2 + dy2
    else:
        # Shrink: move x1/y1 inward (add), x2/y2 inward (subtract)
        nx1 = x1 + dx1
        ny1 = y1 + dy1
        nx2 = x2 - dx2
        ny2 = y2 - dy2

    # Clamp to image bounds
    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(image_w - 1), nx2)
    ny2 = min(float(image_h - 1), ny2)

    # Enforce minimum box size
    if nx2 - nx1 < min_box_size:
        cx = (nx1 + nx2) / 2
        nx1 = max(0.0, cx - min_box_size / 2)
        nx2 = min(float(image_w - 1), nx1 + min_box_size)
        nx1 = max(0.0, nx2 - min_box_size)
    if ny2 - ny1 < min_box_size:
        cy = (ny1 + ny2) / 2
        ny1 = max(0.0, cy - min_box_size / 2)
        ny2 = min(float(image_h - 1), ny1 + min_box_size)
        ny1 = max(0.0, ny2 - min_box_size)

    perturbed = [int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))]
    return perturbed, strategy


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BUSIGTDataset(Dataset):
    """Dataset reading preprocessed BUSI npz files with GT masks.

    During training, box prompts derived from GT masks are randomly perturbed.
    During validation, exact GT boxes are used (no perturbation).

    Args:
        npz_paths: List of paths to .npz files.
        target_size: UltraSAM input size (default 1024).
        augment: Apply data augmentation (horizontal flip).
        perturb: Apply box perturbation (True for train, False for val).
        perturbation_ratios: Dict of strategy → probability.
        seed: Base random seed (combined with worker_id for per-worker RNG).
    """

    def __init__(
        self,
        npz_paths: List[str],
        target_size: int = 1024,
        augment: bool = True,
        perturb: bool = True,
        perturbation_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.npz_paths = npz_paths
        self.target_size = target_size
        self.augment = augment
        self.perturb = perturb
        self.perturbation_ratios = perturbation_ratios or DEFAULT_PERTURBATION_RATIOS
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> Dict:
        data = np.load(self.npz_paths[idx])
        image = data["image"]   # (H, W, 3) uint8 RGB
        label = data["label"]   # (H, W) uint8 binary

        # DetDataPreprocessor expects BGR input (bgr_to_rgb=True in config)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = image_bgr.shape[:2]

        # --- Augmentation: horizontal flip ---
        if self.augment and self.rng.random() > 0.5:
            image_bgr = cv2.flip(image_bgr, 1)
            label = cv2.flip(label, 1)

        # --- Extract box from (possibly flipped) mask ---
        mask_binary = (label > 0).astype(np.uint8)
        bbox = mask_to_bbox(mask_binary)

        if bbox is None:
            # Empty mask: use full image box, skip perturbation
            bbox = [0, 0, orig_w - 1, orig_h - 1]
            perturbed_bbox = bbox
            strategy = "exact"
        elif self.perturb:
            perturbed_bbox, strategy = perturb_box(
                bbox, orig_h, orig_w, self.perturbation_ratios, self.rng
            )
        else:
            perturbed_bbox = bbox
            strategy = "exact"

        # --- Compute box center point (from perturbed box) ---
        bx1, by1, bx2, by2 = perturbed_bbox
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0

        # --- Aspect-ratio preserving resize to target_size + zero-pad ---
        scale = self.target_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        resized_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        resized_mask = cv2.resize(mask_binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded_mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        padded_mask[:new_h, :new_w] = resized_mask.astype(np.float32)

        # --- Transform box & point to 1024-space ---
        box_1024 = transform_coords(perturbed_bbox, (orig_h, orig_w), self.target_size, is_box=True)
        point_1024 = transform_coords([cx, cy], (orig_h, orig_w), self.target_size, is_box=False)

        # --- Build tensors ---
        img_tensor = torch.from_numpy(padded_img.astype(np.float32).transpose(2, 0, 1))

        return {
            "image": img_tensor,                                         # (3, 1024, 1024)
            "mask": torch.from_numpy(padded_mask),                       # (1024, 1024)
            "mask_orig": torch.from_numpy(mask_binary.astype(np.float32)),  # (orig_h, orig_w)
            "box": torch.tensor(box_1024, dtype=torch.float32),          # (4,)
            "point": torch.tensor(point_1024, dtype=torch.float32),      # (2,)
            "prompt_type": torch.tensor(1, dtype=torch.long),            # always BOX
            "orig_shape": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "img_shape": torch.tensor([new_h, new_w], dtype=torch.long),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "case_name": os.path.splitext(os.path.basename(self.npz_paths[idx]))[0],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "mask_orig": [b["mask_orig"] for b in batch],  # variable sizes
        "box": torch.stack([b["box"] for b in batch]),
        "point": torch.stack([b["point"] for b in batch]),
        "prompt_type": torch.stack([b["prompt_type"] for b in batch]),
        "orig_shape": torch.stack([b["orig_shape"] for b in batch]),
        "img_shape": torch.stack([b["img_shape"] for b in batch]),
        "scale": torch.stack([b["scale"] for b in batch]),
        "case_name": [b["case_name"] for b in batch],
    }


def worker_init_fn(worker_id):
    """Seed each DataLoader worker's RNG differently."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.rng = np.random.default_rng(dataset.seed + worker_id)


# ---------------------------------------------------------------------------
# Model loading and freeze strategy
# ---------------------------------------------------------------------------

def load_ultrasam_model(config_path: str, ckpt_path: str, device: str):
    """Load UltraSAM model."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS

    register_all_modules()

    cfg = Config.fromfile(config_path)

    # Process custom imports
    if hasattr(cfg, "custom_imports"):
        import importlib
        for mod_name in cfg.custom_imports.get("imports", []):
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                print(f"Warning: Failed to import {mod_name}: {e}")

    # Apply MonkeyPatch for custom multi-head attention
    try:
        from endosam.models.utils.custom_functional import (
            multi_head_attention_forward as custom_mha_forward,
        )
        torch.nn.functional.multi_head_attention_forward = custom_mha_forward
        print("Applied UltraSAM MonkeyPatch")
    except ImportError as e:
        print(f"Warning: Could not apply MonkeyPatch: {e}")

    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.to(device)

    return model, cfg


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def apply_freeze_strategy(
    model: nn.Module,
    freeze_backbone: bool = False,
    unfreeze_backbone_layers: int = 0,
    freeze_prompt_encoder: bool = False,
    freeze_decoder: bool = False,
    freeze_mask_head: bool = False,
):
    """Apply freezing strategy to UltraSAM model."""
    print("\nApplying freeze strategy...")

    def freeze_module(module, name):
        count = 0
        for param in module.parameters():
            param.requires_grad = False
            count += param.numel()
        print(f"  Froze {name}: {count:,} parameters")

    if freeze_backbone:
        if unfreeze_backbone_layers > 0:
            for param in model.backbone.parameters():
                param.requires_grad = False
            if hasattr(model.backbone, "layers"):
                layers = model.backbone.layers
                num_layers = len(layers)
                for i in range(max(0, num_layers - unfreeze_backbone_layers), num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"  Backbone: frozen except last {unfreeze_backbone_layers} layers")
        else:
            freeze_module(model.backbone, "backbone")
    else:
        print("  Backbone: trainable")

    if freeze_prompt_encoder:
        freeze_module(model.prompt_encoder, "prompt_encoder")
    else:
        print("  Prompt encoder: trainable")

    if freeze_decoder:
        freeze_module(model.decoder, "decoder")
    else:
        print("  Decoder: trainable")

    if freeze_mask_head:
        freeze_module(model.bbox_head, "mask_head")
    else:
        print("  Mask head: trainable")

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    print(f"\nParameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")


# ---------------------------------------------------------------------------
# Data sample construction for UltraSAM
# ---------------------------------------------------------------------------

def build_data_samples_for_training(batch: Dict, device: str, target_size: int = 1024):
    """Build DetDataSample for training (includes GT masks for loss computation)."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks

    B = batch["image"].shape[0]
    data_samples = []

    for i in range(B):
        data_sample = DetDataSample()
        gt_instances = InstanceData()

        x1, y1, x2, y2 = batch["box"][i].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        gt_instances.points = torch.tensor(
            [[[cx, cy]]], dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
        gt_instances.prompt_types = torch.tensor([1], dtype=torch.long, device=device)
        gt_instances.labels = torch.tensor([0], dtype=torch.long, device=device)

        # GT mask in 1024-space for loss computation
        mask_np = batch["mask"][i].cpu().numpy()
        gt_instances.masks = BitmapMasks([mask_np], mask_np.shape[0], mask_np.shape[1])

        data_sample.gt_instances = gt_instances

        orig_h, orig_w = batch["orig_shape"][i].tolist()
        new_h, new_w = batch["img_shape"][i].tolist()
        sc = batch["scale"][i].item()

        data_sample.set_metainfo({
            "img_shape": (new_h, new_w),
            "ori_shape": (orig_h, orig_w),
            "scale_factor": (sc, sc),
            "pad_shape": (target_size, target_size),
            "batch_input_shape": (target_size, target_size),
        })

        data_samples.append(data_sample)

    return data_samples


def build_data_samples_for_inference(batch: Dict, device: str, target_size: int = 1024):
    """Build DetDataSample for inference (no GT masks — prevents data leakage)."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    B = batch["image"].shape[0]
    data_samples = []

    for i in range(B):
        data_sample = DetDataSample()
        gt_instances = InstanceData()

        x1, y1, x2, y2 = batch["box"][i].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        gt_instances.points = torch.tensor(
            [[[cx, cy]]], dtype=torch.float32, device=device)
        gt_instances.boxes = torch.tensor(
            [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
        gt_instances.prompt_types = torch.tensor([1], dtype=torch.long, device=device)
        gt_instances.labels = torch.tensor([0], dtype=torch.long, device=device)

        data_sample.gt_instances = gt_instances

        orig_h, orig_w = batch["orig_shape"][i].tolist()
        new_h, new_w = batch["img_shape"][i].tolist()
        sc = batch["scale"][i].item()

        data_sample.set_metainfo({
            "img_shape": (new_h, new_w),
            "ori_shape": (orig_h, orig_w),
            "scale_factor": (sc, sc),
            "pad_shape": (target_size, target_size),
            "batch_input_shape": (target_size, target_size),
        })

        data_samples.append(data_sample)

    return data_samples


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    fold_idx: Optional[int] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    fold_prefix = f"Fold {fold_idx} | " if fold_idx is not None else ""

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)

        data_samples = build_data_samples_for_training(batch, device)

        data = {"inputs": [img for img in images], "data_samples": data_samples}
        data = model.data_preprocessor(data, True)

        optimizer.zero_grad()
        losses = model(**data, mode="loss")

        loss = sum(v for k, v in losses.items() if "loss" in k)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=0.1
        )

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  {fold_prefix}Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return {"total": avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    fold_idx: Optional[int] = None,
) -> Dict[str, float]:
    """Validate with exact GT boxes, computing metrics at original resolution."""
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    fold_prefix = f"Fold {fold_idx} | " if fold_idx is not None else ""

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)

            data_samples = build_data_samples_for_inference(batch, device)

            data = {"inputs": [img for img in images], "data_samples": data_samples}
            data = model.data_preprocessor(data, False)
            results = model(**data, mode="predict")

            for i, result in enumerate(results):
                if not (hasattr(result, "pred_instances")
                        and hasattr(result.pred_instances, "masks")
                        and len(result.pred_instances.masks) > 0):
                    continue

                # Predicted mask (SAMHead rescales to ori_shape)
                pred_mask = result.pred_instances.masks[0].cpu().numpy().astype(np.float32)

                # GT mask at original resolution
                gt_mask = batch["mask_orig"][i].numpy()
                orig_h, orig_w = batch["orig_shape"][i].tolist()
                gt_mask = gt_mask[:orig_h, :orig_w]

                # Ensure predicted mask matches original resolution
                if pred_mask.shape != (orig_h, orig_w):
                    pred_mask = cv2.resize(
                        pred_mask, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                pred_binary = (pred_mask > 0.5).astype(np.float32)
                gt_binary = (gt_mask > 0.5).astype(np.float32)

                intersection = (pred_binary * gt_binary).sum()
                pred_sum = pred_binary.sum()
                gt_sum = gt_binary.sum()
                union = pred_sum + gt_sum - intersection

                # Handle empty masks
                if pred_sum == 0 and gt_sum == 0:
                    iou = 1.0
                    dice = 1.0
                elif union == 0:
                    iou = 0.0
                    dice = 0.0
                else:
                    iou = intersection / (union + 1e-8)
                    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)

                total_iou += iou
                total_dice += dice
                num_samples += 1

    if num_samples == 0:
        print(f"  {fold_prefix}Warning: no valid predictions during validation")
        return {"iou": 0.0, "dice": 0.0, "n_samples": 0}

    return {
        "iou": total_iou / num_samples,
        "dice": total_dice / num_samples,
        "n_samples": num_samples,
    }


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def finetune_fold(
    fold_idx: int,
    train_paths: List[str],
    val_paths: List[str],
    args,
    perturbation_ratios: Dict[str, float],
) -> Dict[str, float]:
    """Train and validate a single fold."""
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Save split info
    train_names = [os.path.splitext(os.path.basename(p))[0] for p in train_paths]
    val_names = [os.path.splitext(os.path.basename(p))[0] for p in val_paths]
    with open(os.path.join(fold_dir, "split.json"), "w") as f:
        json.dump({"train": train_names, "val": val_names}, f, indent=2)

    # Load model fresh for each fold (avoid weight leakage between folds)
    config_path = os.path.join(ROOT_DIR, args.ultrasam_config) \
        if not os.path.isabs(args.ultrasam_config) else args.ultrasam_config
    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    print(f"\nLoading model for fold {fold_idx}...")
    model, cfg = load_ultrasam_model(config_path, ckpt_path, args.device)

    # Apply freeze strategy
    apply_freeze_strategy(
        model,
        freeze_backbone=args.freeze_backbone,
        unfreeze_backbone_layers=args.unfreeze_backbone_layers,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
        freeze_decoder=args.freeze_decoder,
        freeze_mask_head=args.freeze_mask_head,
    )

    # Datasets
    train_dataset = BUSIGTDataset(
        train_paths,
        augment=True,
        perturb=True,
        perturbation_ratios=perturbation_ratios,
        seed=args.seed + fold_idx,  # Different seed per fold
    )
    val_dataset = BUSIGTDataset(
        val_paths,
        augment=False,
        perturb=False,  # Exact GT boxes for validation
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Perturbation ratios: {perturbation_ratios}")

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    best_val_iou = 0.0
    training_log = []

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx} | Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        start_time = time.time()

        train_losses = train_epoch(
            model, train_loader, optimizer, args.device, epoch, fold_idx
        )
        print(f"  Fold {fold_idx} | Train Loss: {train_losses['total']:.4f}")

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_losses["total"],
            "lr": scheduler.get_last_lr()[0],
        }

        # Validate
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, args.device, fold_idx)
            print(f"  Fold {fold_idx} | Val IoU: {val_metrics['iou']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f} "
                  f"({val_metrics['n_samples']} samples)")

            epoch_log["val_iou"] = val_metrics["iou"]
            epoch_log["val_dice"] = val_metrics["dice"]

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_iou": val_metrics["iou"],
                    "val_dice": val_metrics["dice"],
                    "perturbation_ratios": perturbation_ratios,
                }, os.path.join(fold_dir, "best.pth"))
                print(f"  -> Saved best model (IoU: {best_val_iou:.4f})")

        scheduler.step()

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(fold_dir, f"epoch_{epoch + 1}.pth"))

        epoch_time = time.time() - start_time
        epoch_log["epoch_time"] = epoch_time
        training_log.append(epoch_log)

        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "perturbation_ratios": perturbation_ratios,
    }, os.path.join(fold_dir, "final.pth"))

    # Save training log
    with open(os.path.join(fold_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nFold {fold_idx} complete. Best val IoU: {best_val_iou:.4f}")

    return {"best_val_iou": best_val_iou}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune UltraSAM on BUSI GT masks with perturbed box prompts"
    )

    # Data and model
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with preprocessed npz files "
                             "(e.g. outputs/preprocessed/busi)")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="Path to UltraSAM config")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for models")

    # Cross-validation
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train only this fold (0-indexed). Omit for all folds.")

    # Box perturbation ratios
    parser.add_argument("--p_exact", type=float, default=0.10,
                        help="Probability of exact (unperturbed) box")
    parser.add_argument("--p_slight_enlarge", type=float, default=0.40,
                        help="Probability of slightly enlarged box")
    parser.add_argument("--p_slight_shrink", type=float, default=0.40,
                        help="Probability of slightly shrunk box")
    parser.add_argument("--p_heavy_enlarge", type=float, default=0.05,
                        help="Probability of heavily enlarged box")
    parser.add_argument("--p_heavy_shrink", type=float, default=0.05,
                        help="Probability of heavily shrunk box")

    # Freeze options
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze image encoder")
    parser.add_argument("--unfreeze_backbone_layers", type=int, default=0,
                        help="Unfreeze last N backbone layers (requires --freeze_backbone)")
    parser.add_argument("--freeze_prompt_encoder", action="store_true",
                        help="Freeze prompt encoder")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze transformer decoder")
    parser.add_argument("--freeze_mask_head", action="store_true",
                        help="Freeze mask prediction head")
    parser.add_argument("--freeze_all_but_head", action="store_true",
                        help="Freeze everything except mask head")
    parser.add_argument("--freeze_all_but_decoder_head", action="store_true",
                        help="Freeze backbone and prompt encoder only")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Validate every N epochs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for perturbation")

    args = parser.parse_args()

    # --- Validate perturbation ratios ---
    perturbation_ratios = {
        "exact": args.p_exact,
        "slight_enlarge": args.p_slight_enlarge,
        "slight_shrink": args.p_slight_shrink,
        "heavy_enlarge": args.p_heavy_enlarge,
        "heavy_shrink": args.p_heavy_shrink,
    }
    ratio_sum = sum(perturbation_ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        print(f"Error: Perturbation ratios must sum to 1.0, got {ratio_sum:.6f}")
        print(f"  Ratios: {perturbation_ratios}")
        return

    # --- Apply freeze presets ---
    if args.freeze_all_but_head:
        args.freeze_backbone = True
        args.freeze_prompt_encoder = True
        args.freeze_decoder = True

    if args.freeze_all_but_decoder_head:
        args.freeze_backbone = True
        args.freeze_prompt_encoder = True

    # --- Gather npz files ---
    npz_files = sorted(glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"Error: No npz files found in {args.data_dir}")
        return
    print(f"Found {len(npz_files)} samples")

    # --- K-fold split (same seed as TransUNet for consistency) ---
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=123)
    splits = list(kf.split(npz_files))

    # Determine which folds to train
    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.n_folds:
            print(f"Error: --fold must be in [0, {args.n_folds - 1}]")
            return
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.n_folds))

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Print configuration ---
    print(f"\n{'='*60}")
    print("Fine-tune UltraSAM with Perturbed GT Box Prompts")
    print(f"{'='*60}")
    print(f"Data:       {args.data_dir} ({len(npz_files)} samples)")
    print(f"Config:     {args.ultrasam_config}")
    print(f"Checkpoint: {args.ultrasam_ckpt}")
    print(f"Output:     {args.output_dir}")
    print(f"Folds:      {args.n_folds} (training: {folds_to_train})")
    print(f"Perturbation ratios:")
    for k, v in perturbation_ratios.items():
        print(f"  {k}: {v:.0%}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Device:     {args.device}")
    print(f"{'='*60}")

    # --- Train each fold ---
    results = {}
    for fold_idx in folds_to_train:
        train_idx, val_idx = splits[fold_idx]
        train_paths = [npz_files[i] for i in train_idx]
        val_paths = [npz_files[i] for i in val_idx]

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}: {len(train_paths)} train, {len(val_paths)} val")
        print(f"{'='*60}")

        fold_result = finetune_fold(
            fold_idx, train_paths, val_paths, args, perturbation_ratios
        )
        results[fold_idx] = fold_result

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for fold_idx in sorted(results.keys()):
        iou = results[fold_idx]["best_val_iou"]
        print(f"  Fold {fold_idx}: Best Val IoU = {iou:.4f}")

    if len(results) > 1:
        ious = [r["best_val_iou"] for r in results.values()]
        print(f"  Mean: {np.mean(ious):.4f} +/- {np.std(ious):.4f}")

    # Save cross-fold summary
    summary = {
        "n_folds": args.n_folds,
        "perturbation_ratios": perturbation_ratios,
        "folds": {
            str(k): v for k, v in results.items()
        },
    }
    if len(results) > 1:
        ious = [r["best_val_iou"] for r in results.values()]
        summary["mean_val_iou"] = float(np.mean(ious))
        summary["std_val_iou"] = float(np.std(ious))

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nModels saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
