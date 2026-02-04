"""Finetune UltraSAM with LoRA and optional FFN training on cropped data.

This script uses LoRA (Low-Rank Adaptation) for efficient finetuning of the
UltraSAM backbone, with options to also train the FFN layers.

Usage:
    # LoRA only (most efficient)
    python pipeline/08_finetune_ultrasam_lora.py \
        --data_dir outputs/crop_data/busi/combined \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam_lora/busi \
        --lora_rank 16 \
        --epochs 50

    # LoRA + FFN training
    python pipeline/08_finetune_ultrasam_lora.py \
        --data_dir outputs/crop_data/busi/combined \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam_lora/busi \
        --lora_rank 16 \
        --train_ffn \
        --epochs 50
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

# Add UltraSam root to sys.path
ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)

from utils.prompt_utils import mask_to_bbox, mask_to_centroid, transform_coords


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for linear transformations."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ A^T @ B^T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha
        )
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora(x)


def inject_lora_layers(model: nn.Module, rank: int = 16, alpha: float = 16.0, target_modules: List[str] = None):
    """Inject LoRA layers into the model's attention modules."""
    if target_modules is None:
        target_modules = ['qkv', 'proj']

    lora_count = 0

    if hasattr(model, 'backbone'):
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules:
                    if target in name:
                        parts = name.split('.')
                        parent = model.backbone
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
                        setattr(parent, parts[-1], lora_linear)
                        lora_count += 1
                        break

    print(f"Injected {lora_count} LoRA layers (rank={rank}, alpha={alpha})")
    return lora_count


class CropDataset(Dataset):
    """Dataset for cropped UltraSAM training data."""

    def __init__(
        self,
        data_dir: str,
        target_size: int = 1024,
        augment: bool = True,
        prompt_type: str = "box",
    ):
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment
        self.prompt_type = prompt_type

        ann_path = os.path.join(data_dir, "annotations", "train.json")
        if not os.path.exists(ann_path):
            combined_path = os.path.join(data_dir, "combined", "annotations", "train.json")
            if os.path.exists(combined_path):
                raise FileNotFoundError(
                    f"Annotation file not found at: {ann_path}\n"
                    f"Did you mean to use: {os.path.join(data_dir, 'combined')}?\n"
                    f"Use --data_dir {os.path.join(data_dir, 'combined')} instead."
                )
            else:
                raise FileNotFoundError(
                    f"Annotation file not found at: {ann_path}\n"
                    f"Expected directory structure:\n"
                    f"  {data_dir}/\n"
                    f"    images/\n"
                    f"    annotations/\n"
                    f"      train.json\n"
                    f"Run 07_generate_crop_data.py first, then use the 'combined' subdirectory."
                )
        with open(ann_path) as f:
            coco_data = json.load(f)

        self.images = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.image_ids = list(self.img_to_anns.keys())
        print(f"Loaded {len(self.image_ids)} images with {len(self.annotations)} annotations")

    def __len__(self):
        return len(self.image_ids)

    def _decode_rle(self, rle: Dict) -> np.ndarray:
        from pycocotools import mask as mask_util
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_util.decode(rle)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.img_to_anns[img_id]

        img_path = os.path.join(self.data_dir, "images", img_info["file_name"])
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")

        orig_h, orig_w = image_bgr.shape[:2]

        ann = annotations[0]
        mask_orig = self._decode_rle(ann["segmentation"])

        # Augmentation (before any resizing)
        if self.augment and np.random.random() > 0.5:
            image_bgr = cv2.flip(image_bgr, 1)
            mask_orig = cv2.flip(mask_orig, 1)

        # Preprocess image: resize and pad to target_size
        scale = self.target_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        resized_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # Keep image as uint8 BGR - data_preprocessor will normalize
        img_tensor = torch.from_numpy(padded_img.astype(np.float32).transpose(2, 0, 1))

        # Preprocess mask the same way
        resized_mask = cv2.resize(mask_orig, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded_mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        padded_mask[:new_h, :new_w] = resized_mask
        mask_tensor = torch.from_numpy(padded_mask)

        # Generate prompts from original mask (before resize)
        mask_binary = (mask_orig > 0).astype(np.uint8)
        bbox = mask_to_bbox(mask_binary)
        if bbox is None:
            bbox = [0, 0, orig_w - 1, orig_h - 1]
        centroid = mask_to_centroid(mask_binary)

        # Transform prompts to 1024-space
        box_1024 = transform_coords(bbox, (orig_h, orig_w), self.target_size, is_box=True)
        point_1024 = transform_coords(centroid, (orig_h, orig_w), self.target_size, is_box=False)

        if self.prompt_type == "box":
            prompt_type = 1
        elif self.prompt_type == "point":
            prompt_type = 0
        else:
            prompt_type = np.random.randint(0, 2)

        return {
            "image": img_tensor,
            "mask": mask_tensor,  # Padded mask in 1024x1024 space
            "mask_orig": torch.from_numpy(mask_orig.astype(np.float32)),  # Original resolution mask
            "box": torch.tensor(box_1024, dtype=torch.float32),
            "point": torch.tensor(point_1024, dtype=torch.float32),
            "prompt_type": torch.tensor(prompt_type, dtype=torch.long),
            "orig_shape": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "img_shape": torch.tensor([new_h, new_w], dtype=torch.long),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "mask_orig": [b["mask_orig"] for b in batch],  # Keep as list (variable sizes)
        "box": torch.stack([b["box"] for b in batch]),
        "point": torch.stack([b["point"] for b in batch]),
        "prompt_type": torch.stack([b["prompt_type"] for b in batch]),
        "orig_shape": torch.stack([b["orig_shape"] for b in batch]),
        "img_shape": torch.stack([b["img_shape"] for b in batch]),
        "scale": torch.stack([b["scale"] for b in batch]),
    }


def build_ultrasam_model(config_path: str, ckpt_path: str, device: str = "cuda:0"):
    """Build and load standard UltraSAM model."""
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
            except ImportError as e:
                print(f"Warning: Failed to import {mod_name}: {e}")

    # Apply MonkeyPatch - CRITICAL for UltraSAM
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
    print(f"Loaded checkpoint from {ckpt_path}")

    model = model.to(device)
    return model, cfg


def apply_freeze_strategy(
    model: nn.Module,
    train_lora: bool = True,
    train_ffn: bool = False,
    train_decoder: bool = False,
    train_mask_head: bool = False,
):
    """Apply freezing strategy for LoRA finetuning."""
    print("\nApplying freeze strategy...")

    for param in model.parameters():
        param.requires_grad = False

    if train_lora:
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        print("  LoRA parameters: trainable")

    if train_ffn and hasattr(model, 'backbone'):
        for name, param in model.backbone.named_parameters():
            if 'mlp' in name or 'ffn' in name:
                param.requires_grad = True
        print("  Backbone FFN: trainable")

    if train_decoder and hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = True
        print("  Decoder: trainable")

    if train_mask_head and hasattr(model, 'bbox_head'):
        for param in model.bbox_head.parameters():
            param.requires_grad = True
        print("  Mask head: trainable")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100*trainable_params/total_params:.2f}%)")


def build_data_samples_for_training(batch: Dict, device: str, target_size: int = 1024):
    """Build DetDataSample list for UltraSAM training (loss mode)."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks

    B = batch["image"].shape[0]
    data_samples = []

    for i in range(B):
        data_sample = DetDataSample()
        gt_instances = InstanceData()

        box = batch["box"][i]
        point = batch["point"][i]
        prompt_type = batch["prompt_type"][i].item()

        x1, y1, x2, y2 = box.tolist()
        px, py = point.tolist()

        if prompt_type == 1:  # BOX
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            gt_instances.points = torch.tensor(
                [[[cx, cy]]], dtype=torch.float32, device=device)
            gt_instances.boxes = torch.tensor(
                [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
            gt_instances.prompt_types = torch.tensor([1], dtype=torch.long, device=device)
        else:  # POINT
            gt_instances.points = torch.tensor(
                [[[px, py]]], dtype=torch.float32, device=device)
            gt_instances.boxes = torch.tensor(
                [[[0.0, 0.0], [float(target_size), float(target_size)]]],
                dtype=torch.float32, device=device)
            gt_instances.prompt_types = torch.tensor([0], dtype=torch.long, device=device)

        gt_instances.labels = torch.tensor([0], dtype=torch.long, device=device)

        # Mask for loss computation - in padded 1024 space
        mask_np = batch["mask"][i].cpu().numpy()
        gt_instances.masks = BitmapMasks([mask_np], mask_np.shape[0], mask_np.shape[1])

        data_sample.gt_instances = gt_instances

        orig_h, orig_w = batch["orig_shape"][i].tolist()
        new_h, new_w = batch["img_shape"][i].tolist()
        scale = batch["scale"][i].item()

        data_sample.set_metainfo({
            "img_shape": (new_h, new_w),
            "ori_shape": (orig_h, orig_w),
            "scale_factor": (scale, scale),
            "pad_shape": (target_size, target_size),
            "batch_input_shape": (target_size, target_size),
        })

        data_samples.append(data_sample)

    return data_samples


def build_data_samples_for_inference(batch: Dict, device: str, target_size: int = 1024):
    """Build DetDataSample list for UltraSAM inference (predict mode).

    Key difference from training: we don't include masks in gt_instances.
    """
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    B = batch["image"].shape[0]
    data_samples = []

    for i in range(B):
        data_sample = DetDataSample()
        gt_instances = InstanceData()

        box = batch["box"][i]
        point = batch["point"][i]
        prompt_type = batch["prompt_type"][i].item()

        x1, y1, x2, y2 = box.tolist()
        px, py = point.tolist()

        if prompt_type == 1:  # BOX
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            gt_instances.points = torch.tensor(
                [[[cx, cy]]], dtype=torch.float32, device=device)
            gt_instances.boxes = torch.tensor(
                [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
            gt_instances.prompt_types = torch.tensor([1], dtype=torch.long, device=device)
        else:  # POINT
            gt_instances.points = torch.tensor(
                [[[px, py]]], dtype=torch.float32, device=device)
            gt_instances.boxes = torch.tensor(
                [[[0.0, 0.0], [float(target_size), float(target_size)]]],
                dtype=torch.float32, device=device)
            gt_instances.prompt_types = torch.tensor([0], dtype=torch.long, device=device)

        gt_instances.labels = torch.tensor([0], dtype=torch.long, device=device)

        data_sample.gt_instances = gt_instances

        orig_h, orig_w = batch["orig_shape"][i].tolist()
        new_h, new_w = batch["img_shape"][i].tolist()
        scale = batch["scale"][i].item()

        data_sample.set_metainfo({
            "img_shape": (new_h, new_w),
            "ori_shape": (orig_h, orig_w),
            "scale_factor": (scale, scale),
            "pad_shape": (target_size, target_size),
            "batch_input_shape": (target_size, target_size),
        })

        data_samples.append(data_sample)

    return data_samples


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        data_samples = build_data_samples_for_training(batch, device)

        # Prepare data for model
        data = {"inputs": [img for img in images], "data_samples": data_samples}
        data = model.data_preprocessor(data, True)

        optimizer.zero_grad()
        losses = model(**data, mode="loss")

        loss = sum(v for k, v in losses.items() if "loss" in k)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(model, dataloader, device):
    """Validate model by running inference and comparing to GT."""
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            data_samples = build_data_samples_for_inference(batch, device)

            # Run inference
            data = {"inputs": [img for img in images], "data_samples": data_samples}
            data = model.data_preprocessor(data, False)
            results = model(**data, mode="predict")

            # Process each result
            for i, result in enumerate(results):
                # Get GT mask at original resolution
                gt_mask_orig = batch["mask_orig"][i].numpy()
                orig_h, orig_w = batch["orig_shape"][i].tolist()

                # Extract predicted mask
                pred_mask = None
                if hasattr(result, "pred_instances"):
                    pred_inst = result.pred_instances
                    if hasattr(pred_inst, "masks") and len(pred_inst.masks) > 0:
                        # pred_inst.masks is at ori_shape after SAMHead rescaling
                        mask_tensor = pred_inst.masks[0]
                        if isinstance(mask_tensor, torch.Tensor):
                            pred_mask = mask_tensor.cpu().numpy()
                        else:
                            pred_mask = np.array(mask_tensor)

                if pred_mask is None:
                    if batch_idx == 0 and i == 0:
                        print("  Warning: No prediction mask for first sample")
                        if hasattr(result, "pred_instances"):
                            pi = result.pred_instances
                            print(f"    pred_instances attrs: {[a for a in dir(pi) if not a.startswith('_')]}")
                            if hasattr(pi, "masks"):
                                print(f"    masks: {type(pi.masks)}, len={len(pi.masks) if hasattr(pi.masks, '__len__') else 'N/A'}")
                    continue

                # Ensure pred_mask is at original resolution
                if pred_mask.shape != (orig_h, orig_w):
                    pred_mask = cv2.resize(
                        pred_mask.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Ensure gt_mask is at original resolution
                if gt_mask_orig.shape != (orig_h, orig_w):
                    gt_mask_orig = cv2.resize(
                        gt_mask_orig.astype(np.float32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Binarize
                pred_binary = (pred_mask > 0.5).astype(np.float32)
                gt_binary = (gt_mask_orig > 0.5).astype(np.float32)

                # Compute metrics
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum() - intersection
                iou = intersection / (union + 1e-6)
                dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-6)

                total_iou += iou
                total_dice += dice
                num_samples += 1

    if num_samples == 0:
        print("  Warning: No valid predictions during validation")
        return {"iou": 0, "dice": 0}

    return {"iou": total_iou / num_samples, "dice": total_dice / num_samples}


def finetune_ultrasam_lora(
    data_dir: str,
    ultrasam_config: str,
    ultrasam_ckpt: str,
    output_dir: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    train_ffn: bool = False,
    train_decoder: bool = False,
    train_mask_head: bool = False,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda:0",
    prompt_type: str = "box",
):
    """Finetune UltraSAM with LoRA."""
    os.makedirs(output_dir, exist_ok=True)

    # Build model
    print(f"\nBuilding UltraSAM model...")
    model, cfg = build_ultrasam_model(ultrasam_config, ultrasam_ckpt, device)

    # Inject LoRA layers
    print(f"\nInjecting LoRA layers (rank={lora_rank}, alpha={lora_alpha})")
    lora_count = inject_lora_layers(model, rank=lora_rank, alpha=lora_alpha)

    if lora_count == 0:
        print("Warning: No LoRA layers were injected. Falling back to regular finetuning.")

    # Move model to device again (LoRA layers are created on CPU)
    model = model.to(device)

    # Apply freeze strategy
    apply_freeze_strategy(
        model,
        train_lora=True,
        train_ffn=train_ffn,
        train_decoder=train_decoder,
        train_mask_head=train_mask_head,
    )

    # Create datasets
    print(f"\nLoading data from {data_dir}")
    train_dataset = CropDataset(data_dir, augment=True, prompt_type=prompt_type)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Validation dataset
    val_ann_path = os.path.join(data_dir, "annotations", "val.json")
    val_loader = None
    if os.path.exists(val_ann_path):
        val_dataset = CropDataset(data_dir, augment=False, prompt_type=prompt_type)
        with open(val_ann_path) as f:
            val_coco = json.load(f)
        val_dataset.images = {img["id"]: img for img in val_coco["images"]}
        val_dataset.annotations = val_coco["annotations"]
        val_dataset.img_to_anns = {}
        for ann in val_coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in val_dataset.img_to_anns:
                val_dataset.img_to_anns[img_id] = []
            val_dataset.img_to_anns[img_id].append(ann)
        val_dataset.image_ids = list(val_dataset.img_to_anns.keys())

        if len(val_dataset.image_ids) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
            print(f"Validation set: {len(val_dataset.image_ids)} samples")
        else:
            print("No validation samples found")

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {len(trainable_params)} tensors")

    if len(trainable_params) == 0:
        print("Error: No trainable parameters! Check freeze strategy.")
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training loop
    print(f"\nStarting finetuning for {epochs} epochs")
    best_val_iou = 0

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate every 5 epochs
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Val - IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "lora_config": {"rank": lora_rank, "alpha": lora_alpha},
                    "val_iou": val_metrics["iou"],
                }
                torch.save(save_dict, os.path.join(output_dir, "best.pth"))
                print(f"  Saved best model (IoU: {best_val_iou:.4f})")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(output_dir, f"epoch_{epoch + 1}.pth"))

        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "lora_config": {"rank": lora_rank, "alpha": lora_alpha},
    }, os.path.join(output_dir, "final.pth"))

    print(f"\nTraining complete. Models saved to: {output_dir}")
    print(f"Best validation IoU: {best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Finetune UltraSAM with LoRA")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with cropped data")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="Path to UltraSAM config file")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (default: 16)")

    parser.add_argument("--train_ffn", action="store_true",
                        help="Also train FFN layers in backbone")
    parser.add_argument("--train_decoder", action="store_true",
                        help="Also train transformer decoder")
    parser.add_argument("--train_mask_head", action="store_true",
                        help="Also train mask prediction head")

    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--prompt_type", type=str, default="box",
                        choices=["box", "point", "both"],
                        help="Prompt type for training")

    args = parser.parse_args()

    config_path = os.path.join(ROOT_DIR, args.ultrasam_config) \
        if not os.path.isabs(args.ultrasam_config) else args.ultrasam_config
    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    finetune_ultrasam_lora(
        data_dir=args.data_dir,
        ultrasam_config=config_path,
        ultrasam_ckpt=ckpt_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        train_ffn=args.train_ffn,
        train_decoder=args.train_decoder,
        train_mask_head=args.train_mask_head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        prompt_type=args.prompt_type,
    )


if __name__ == "__main__":
    main()
