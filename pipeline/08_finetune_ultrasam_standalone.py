"""Standalone UltraSAM finetuning script for cropped data.

This is a simpler, standalone finetuning script that doesn't require
mmengine's Runner. It provides more direct control over the training loop.

Usage:
    python pipeline/08_finetune_ultrasam_standalone.py \
        --data_dir outputs/crop_data/busi/fold_0 \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam/busi \
        --freeze_backbone \
        --epochs 50 \
        --lr 1e-4 \
        --device cuda:0
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
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

# Add UltraSam root to sys.path
ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)

from utils.prompt_utils import mask_to_bbox, mask_to_centroid, transform_coords


class CropDataset(Dataset):
    """Dataset for cropped UltraSAM training data."""

    def __init__(
        self,
        data_dir: str,
        target_size: int = 1024,
        augment: bool = True,
        prompt_type: str = "box",
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing images/ and annotations/train.json.
            target_size: Target image size for UltraSAM.
            augment: Whether to apply data augmentation.
            prompt_type: "box", "point", or "both".
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment
        self.prompt_type = prompt_type

        # Load COCO annotations
        ann_path = os.path.join(data_dir, "annotations", "train.json")
        with open(ann_path) as f:
            coco_data = json.load(f)

        self.images = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        # Group annotations by image
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
        """Decode RLE segmentation to binary mask."""
        from pycocotools import mask as mask_util
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_util.decode(rle)

    def _preprocess_image(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, float, int, int]:
        """Preprocess image: resize to target_size with aspect-ratio preserving.

        Returns:
            img_tensor: (3, target_size, target_size) float32 tensor
            scale: scaling factor applied
            new_h, new_w: actual content dimensions
        """
        h, w = image_bgr.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Convert to tensor (keep BGR for DetDataPreprocessor compatibility)
        img_tensor = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1))

        return img_tensor, scale, new_h, new_w

    def _preprocess_mask(self, mask: np.ndarray, scale: float, new_h: int, new_w: int) -> torch.Tensor:
        """Preprocess mask to match preprocessed image."""
        h, w = mask.shape[:2]
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create padded mask
        padded = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        padded[:new_h, :new_w] = resized

        return torch.from_numpy(padded)

    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample.

        Returns dict with:
            - image: (3, 1024, 1024) tensor
            - mask: (1024, 1024) tensor
            - box: (4,) tensor [x1, y1, x2, y2] in 1024 space
            - point: (2,) tensor [x, y] in 1024 space
            - prompt_type: int (0=POINT, 1=BOX)
            - meta: dict with image info
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.img_to_anns[img_id]

        # Load image
        img_path = os.path.join(self.data_dir, "images", img_info["file_name"])
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")

        orig_h, orig_w = image_bgr.shape[:2]

        # Apply augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image_bgr = cv2.flip(image_bgr, 1)
                for ann in annotations:
                    if "segmentation" in ann:
                        # We'll need to flip the mask after decoding
                        ann["_flip_h"] = True

        # Preprocess image
        img_tensor, scale, new_h, new_w = self._preprocess_image(image_bgr)

        # Process annotations (use first annotation for now)
        ann = annotations[0]

        # Decode mask
        mask = self._decode_rle(ann["segmentation"])
        if ann.get("_flip_h", False):
            mask = cv2.flip(mask, 1)

        mask_tensor = self._preprocess_mask(mask, scale, new_h, new_w)

        # Get prompts from mask
        mask_binary = (mask > 0).astype(np.uint8)
        bbox = mask_to_bbox(mask_binary)
        if bbox is None:
            bbox = [0, 0, orig_w - 1, orig_h - 1]

        centroid = mask_to_centroid(mask_binary)

        # Transform to 1024 space
        box_1024 = transform_coords(bbox, (orig_h, orig_w), self.target_size, is_box=True)
        point_1024 = transform_coords(centroid, (orig_h, orig_w), self.target_size, is_box=False)

        # Determine prompt type
        if self.prompt_type == "box":
            prompt_type = 1  # BOX
        elif self.prompt_type == "point":
            prompt_type = 0  # POINT
        else:  # both - random
            prompt_type = np.random.randint(0, 2)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "box": torch.tensor(box_1024, dtype=torch.float32),
            "point": torch.tensor(point_1024, dtype=torch.float32),
            "prompt_type": torch.tensor(prompt_type, dtype=torch.long),
            "meta": {
                "img_id": img_id,
                "file_name": img_info["file_name"],
                "orig_shape": (orig_h, orig_w),
                "img_shape": (new_h, new_w),
                "scale": scale,
            },
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "box": torch.stack([b["box"] for b in batch]),
        "point": torch.stack([b["point"] for b in batch]),
        "prompt_type": torch.stack([b["prompt_type"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


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

    # Apply MonkeyPatch
    try:
        from endosam.models.utils.custom_functional import (
            multi_head_attention_forward as custom_mha_forward,
        )
        torch.nn.functional.multi_head_attention_forward = custom_mha_forward
        print("Applied UltraSAM MonkeyPatch")
    except ImportError as e:
        print(f"Warning: Could not apply MonkeyPatch: {e}")

    # Build model
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

    # Backbone
    if freeze_backbone:
        if unfreeze_backbone_layers > 0:
            # Freeze all backbone first
            for param in model.backbone.parameters():
                param.requires_grad = False

            # Unfreeze last N layers
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

    # Prompt encoder
    if freeze_prompt_encoder:
        freeze_module(model.prompt_encoder, "prompt_encoder")
    else:
        print("  Prompt encoder: trainable")

    # Decoder
    if freeze_decoder:
        freeze_module(model.decoder, "decoder")
    else:
        print("  Decoder: trainable")

    # Mask head
    if freeze_mask_head:
        freeze_module(model.bbox_head, "mask_head")
    else:
        print("  Mask head: trainable")

    # Print summary
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    print(f"\nParameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")


def compute_loss(
    pred_masks: torch.Tensor,
    pred_ious: torch.Tensor,
    gt_masks: torch.Tensor,
    focal_weight: float = 20.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute training loss.

    Args:
        pred_masks: (B, num_masks, H, W) predicted mask logits
        pred_ious: (B, num_masks) predicted IoU scores
        gt_masks: (B, H, W) ground truth masks
        focal_weight: Weight for focal loss
        dice_weight: Weight for dice loss
        iou_weight: Weight for IoU loss

    Returns:
        total_loss, loss_dict
    """
    B, num_masks, H, W = pred_masks.shape

    # Resize GT masks to prediction size
    gt_masks_resized = F.interpolate(
        gt_masks.unsqueeze(1).float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # (B, H, W)

    # Compute IoU for each mask to select best
    pred_binary = (pred_masks.sigmoid() > 0.5).float()
    gt_expanded = gt_masks_resized.unsqueeze(1).expand(-1, num_masks, -1, -1)

    intersection = (pred_binary * gt_expanded).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + gt_expanded.sum(dim=(2, 3)) - intersection
    ious = intersection / (union + 1e-6)  # (B, num_masks)

    # Select best mask per sample
    best_idx = ious.argmax(dim=1)  # (B,)
    best_masks = pred_masks[torch.arange(B), best_idx]  # (B, H, W)
    best_ious_pred = pred_ious[torch.arange(B), best_idx]  # (B,)
    best_ious_gt = ious[torch.arange(B), best_idx]  # (B,)

    # Focal loss
    bce = F.binary_cross_entropy_with_logits(best_masks, gt_masks_resized, reduction="none")
    p = torch.sigmoid(best_masks)
    pt = p * gt_masks_resized + (1 - p) * (1 - gt_masks_resized)
    focal = bce * ((1 - pt) ** 2)
    loss_focal = focal.mean() * focal_weight

    # Dice loss
    p = torch.sigmoid(best_masks)
    intersection = (p * gt_masks_resized).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + gt_masks_resized.sum(dim=(1, 2))
    dice = 1 - (2 * intersection + 1) / (union + 1)
    loss_dice = dice.mean() * dice_weight

    # IoU loss (MSE between predicted and actual IoU)
    loss_iou = F.mse_loss(best_ious_pred, best_ious_gt) * iou_weight

    total_loss = loss_focal + loss_dice + loss_iou

    return total_loss, {
        "loss_focal": loss_focal.item(),
        "loss_dice": loss_dice.item(),
        "loss_iou": loss_iou.item(),
        "total": total_loss.item(),
        "mean_iou": best_ious_gt.mean().item(),
    }


def build_data_sample(batch: Dict, device: str, target_size: int = 1024):
    """Build DetDataSample for UltraSAM forward pass."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    B = batch["image"].shape[0]
    data_samples = []

    for i in range(B):
        data_sample = DetDataSample()
        gt_instances = InstanceData()

        # Get prompt info
        box = batch["box"][i]  # (4,)
        point = batch["point"][i]  # (2,)
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

        # Add mask for training
        from mmdet.structures.mask import BitmapMasks
        mask_np = batch["mask"][i].cpu().numpy()
        gt_instances.masks = BitmapMasks([mask_np], mask_np.shape[0], mask_np.shape[1])

        data_sample.gt_instances = gt_instances

        # Set metainfo
        meta = batch["meta"][i]
        data_sample.set_metainfo({
            "img_shape": meta["img_shape"],
            "ori_shape": meta["orig_shape"],
            "scale_factor": (meta["scale"], meta["scale"]),
            "pad_shape": (target_size, target_size),
            "batch_input_shape": (target_size, target_size),
        })

        data_samples.append(data_sample)

    return data_samples


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_dict_accum = {"loss_focal": 0, "loss_dice": 0, "loss_iou": 0, "mean_iou": 0}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Build data samples
        data_samples = build_data_sample(batch, device)

        # Forward pass through data preprocessor
        data = {"inputs": [img for img in images], "data_samples": data_samples}
        data = model.data_preprocessor(data, True)

        # Compute loss using model's loss method
        optimizer.zero_grad()
        losses = model(**data, mode="loss")

        # Aggregate losses
        loss = sum(v for k, v in losses.items() if "loss" in k)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for k, v in losses.items():
            if k in loss_dict_accum:
                loss_dict_accum[k] += v.item() if torch.is_tensor(v) else v
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    # Average losses
    avg_loss = total_loss / num_batches
    for k in loss_dict_accum:
        loss_dict_accum[k] /= num_batches

    return {"total": avg_loss, **loss_dict_accum}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Build data samples
            data_samples = build_data_sample(batch, device)

            # Forward pass
            data = {"inputs": [img for img in images], "data_samples": data_samples}
            data = model.data_preprocessor(data, False)
            results = model(**data, mode="predict")

            # Compute metrics
            for i, result in enumerate(results):
                if hasattr(result, "pred_instances") and hasattr(result.pred_instances, "masks"):
                    pred_mask = result.pred_instances.masks[0].float()
                    gt_mask = masks[i]

                    # Resize to same size
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = F.interpolate(
                            pred_mask.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape,
                            mode="nearest",
                        ).squeeze()

                    # IoU
                    intersection = (pred_mask * gt_mask).sum()
                    union = pred_mask.sum() + gt_mask.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    total_iou += iou.item()

                    # Dice
                    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
                    total_dice += dice.item()

                    num_samples += 1

    if num_samples == 0:
        return {"iou": 0, "dice": 0}

    return {
        "iou": total_iou / num_samples,
        "dice": total_dice / num_samples,
    }


def finetune_ultrasam(
    data_dir: str,
    ultrasam_config: str,
    ultrasam_ckpt: str,
    output_dir: str,
    freeze_backbone: bool = False,
    unfreeze_backbone_layers: int = 0,
    freeze_prompt_encoder: bool = False,
    freeze_decoder: bool = False,
    freeze_mask_head: bool = False,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda:0",
    prompt_type: str = "box",
):
    """Finetune UltraSAM on cropped data."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model from {ultrasam_ckpt}")
    model, cfg = load_ultrasam_model(ultrasam_config, ultrasam_ckpt, device)

    # Apply freeze strategy
    apply_freeze_strategy(
        model,
        freeze_backbone=freeze_backbone,
        unfreeze_backbone_layers=unfreeze_backbone_layers,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_decoder=freeze_decoder,
        freeze_mask_head=freeze_mask_head,
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

    # Check for validation data
    val_ann_path = os.path.join(data_dir, "annotations", "val.json")
    if os.path.exists(val_ann_path):
        val_dataset = CropDataset(data_dir, augment=False, prompt_type=prompt_type)
        # Override annotation path for validation
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

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )
    else:
        val_loader = None
        print("No validation data found")

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training loop
    print(f"\nStarting training for {epochs} epochs")
    best_val_iou = 0

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        start_time = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train - Loss: {train_losses['total']:.4f}")

        # Validate
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Val - IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")

            # Save best model
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_metrics["iou"],
                }, os.path.join(output_dir, "best.pth"))
                print(f"  Saved best model (IoU: {best_val_iou:.4f})")

        # Update LR
        scheduler.step()

        # Save checkpoint
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
    }, os.path.join(output_dir, "final.pth"))

    print(f"\nTraining complete. Models saved to: {output_dir}")
    print(f"Best validation IoU: {best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Finetune UltraSAM on cropped data (standalone)")

    # Data and model paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with cropped data")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="Path to UltraSAM config")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    # Freeze options
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze image encoder")
    parser.add_argument("--unfreeze_backbone_layers", type=int, default=0,
                        help="Unfreeze last N backbone layers")
    parser.add_argument("--freeze_prompt_encoder", action="store_true",
                        help="Freeze prompt encoder")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze transformer decoder")
    parser.add_argument("--freeze_mask_head", action="store_true",
                        help="Freeze mask head")

    # Presets
    parser.add_argument("--freeze_all_but_head", action="store_true",
                        help="Freeze everything except mask head")
    parser.add_argument("--freeze_all_but_decoder_head", action="store_true",
                        help="Freeze backbone and prompt encoder only")

    # Training parameters
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

    # Resolve paths
    config_path = os.path.join(ROOT_DIR, args.ultrasam_config) \
        if not os.path.isabs(args.ultrasam_config) else args.ultrasam_config
    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    # Apply presets
    freeze_backbone = args.freeze_backbone
    freeze_prompt_encoder = args.freeze_prompt_encoder
    freeze_decoder = args.freeze_decoder
    freeze_mask_head = args.freeze_mask_head

    if args.freeze_all_but_head:
        freeze_backbone = True
        freeze_prompt_encoder = True
        freeze_decoder = True

    if args.freeze_all_but_decoder_head:
        freeze_backbone = True
        freeze_prompt_encoder = True

    finetune_ultrasam(
        data_dir=args.data_dir,
        ultrasam_config=config_path,
        ultrasam_ckpt=ckpt_path,
        output_dir=args.output_dir,
        freeze_backbone=freeze_backbone,
        unfreeze_backbone_layers=args.unfreeze_backbone_layers,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_decoder=freeze_decoder,
        freeze_mask_head=freeze_mask_head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        prompt_type=args.prompt_type,
    )


if __name__ == "__main__":
    main()
