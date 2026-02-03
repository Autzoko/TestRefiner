"""Finetune UltraSAM with LoRA and optional FFN training on cropped data.

This script uses LoRA (Low-Rank Adaptation) for efficient finetuning of the
UltraSAM backbone, with options to also train the FFN layers.

LoRA Configuration:
    - Targets: qkv projections in the ViT backbone
    - Default rank: 16, alpha: 16
    - Can optionally train FFN layers in backbone (--train_ffn)

Usage:
    # LoRA only (most efficient)
    python pipeline/08_finetune_ultrasam_lora.py \
        --data_dir outputs/crop_data/busi/combined \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam_lora/busi \
        --lora_rank 16 \
        --epochs 50

    # LoRA + FFN training
    python pipeline/08_finetune_ultrasam_lora.py \
        --data_dir outputs/crop_data/busi/combined \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam_lora/busi \
        --lora_rank 16 \
        --train_ffn \
        --epochs 50
"""

import argparse
import json
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

        # Load COCO annotations
        ann_path = os.path.join(data_dir, "annotations", "train.json")
        if not os.path.exists(ann_path):
            # Provide helpful error message
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
        from pycocotools import mask as mask_util
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_util.decode(rle)

    def _preprocess_image(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, float, int, int]:
        h, w = image_bgr.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        img_tensor = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1))
        return img_tensor, scale, new_h, new_w

    def _preprocess_mask(self, mask: np.ndarray, scale: float, new_h: int, new_w: int) -> torch.Tensor:
        h, w = mask.shape[:2]
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        padded[:new_h, :new_w] = resized
        return torch.from_numpy(padded)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.img_to_anns[img_id]

        img_path = os.path.join(self.data_dir, "images", img_info["file_name"])
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")

        orig_h, orig_w = image_bgr.shape[:2]

        if self.augment and np.random.random() > 0.5:
            image_bgr = cv2.flip(image_bgr, 1)
            for ann in annotations:
                if "segmentation" in ann:
                    ann["_flip_h"] = True

        img_tensor, scale, new_h, new_w = self._preprocess_image(image_bgr)

        ann = annotations[0]
        mask = self._decode_rle(ann["segmentation"])
        if ann.get("_flip_h", False):
            mask = cv2.flip(mask, 1)

        mask_tensor = self._preprocess_mask(mask, scale, new_h, new_w)

        mask_binary = (mask > 0).astype(np.uint8)
        bbox = mask_to_bbox(mask_binary)
        if bbox is None:
            bbox = [0, 0, orig_w - 1, orig_h - 1]

        centroid = mask_to_centroid(mask_binary)

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
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "box": torch.stack([b["box"] for b in batch]),
        "point": torch.stack([b["point"] for b in batch]),
        "prompt_type": torch.stack([b["prompt_type"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def build_lora_model(
    base_ckpt: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_targets: List[str] = None,
    device: str = "cuda:0",
):
    """Build UltraSAM model with LoRA on the backbone.

    Args:
        base_ckpt: Path to pretrained UltraSAM checkpoint.
        lora_rank: LoRA rank (default: 16).
        lora_alpha: LoRA alpha scaling factor (default: 16).
        lora_dropout: Dropout rate for LoRA layers (default: 0.1).
        lora_targets: List of target layer types for LoRA (default: ['qkv']).
        device: Device to load model on.

    Returns:
        model, cfg tuple.
    """
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS

    register_all_modules()

    if lora_targets is None:
        lora_targets = [dict(type='qkv')]

    # Build config for LoRA model
    # We use the sam_lora.py config as reference
    base_config_path = os.path.join(
        ULTRASAM_ROOT, "configs/_base_/models/sam_mask_refinement.py")
    cfg = Config.fromfile(base_config_path)

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

    # Modify backbone config to use LoRA wrapper
    # Note: We don't use _delete_=True here since we're directly assigning,
    # not merging with config inheritance
    cfg.model.backbone = dict(
        type='mmpretrain.LoRAModel',
        module=dict(
            type='mmpretrain.ViTSAM',
            arch='base',
            img_size=1024,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        ),
        alpha=lora_alpha,
        rank=lora_rank,
        drop_rate=lora_dropout,
        targets=lora_targets,
    )

    # Build model
    model = MODELS.build(cfg.model)

    # Load pretrained weights (need to handle LoRA wrapper)
    # First load the base checkpoint
    ckpt = torch.load(base_ckpt, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Filter and rename keys for the LoRA-wrapped backbone
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            # Map backbone.X to backbone.module.X for LoRA wrapper
            new_key = k.replace("backbone.", "backbone.module.")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Load with strict=False to allow missing LoRA parameters
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # Filter out expected missing keys (LoRA parameters)
    unexpected_non_lora = [k for k in unexpected if 'lora' not in k.lower()]
    if unexpected_non_lora:
        print(f"Warning: Unexpected non-LoRA keys: {unexpected_non_lora[:5]}...")

    model = model.to(device)
    return model, cfg


def apply_freeze_strategy(
    model: nn.Module,
    train_lora: bool = True,
    train_ffn: bool = False,
    train_decoder: bool = False,
    train_mask_head: bool = False,
):
    """Apply freezing strategy for LoRA finetuning.

    Args:
        model: UltraSAM model with LoRA backbone.
        train_lora: Train LoRA parameters (default: True).
        train_ffn: Also train FFN layers in backbone (default: False).
        train_decoder: Train transformer decoder (default: False).
        train_mask_head: Train mask prediction head (default: False).
    """
    print("\nApplying freeze strategy...")

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0

    # Unfreeze LoRA parameters in backbone
    if train_lora and hasattr(model, 'backbone'):
        for name, param in model.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                trainable_count += param.numel()
        print(f"  LoRA parameters: trainable")

    # Unfreeze FFN layers in backbone
    if train_ffn and hasattr(model, 'backbone'):
        for name, param in model.backbone.named_parameters():
            # FFN in ViT is typically named 'mlp' or 'ffn'
            if 'mlp' in name or 'ffn' in name:
                param.requires_grad = True
                trainable_count += param.numel()
        print(f"  Backbone FFN: trainable")

    # Unfreeze decoder
    if train_decoder and hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        print(f"  Decoder: trainable")

    # Unfreeze mask head
    if train_mask_head and hasattr(model, 'bbox_head'):
        for param in model.bbox_head.parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        print(f"  Mask head: trainable")

    # Count total
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100*trainable_params/total_params:.2f}%)")


def build_data_sample(batch: Dict, device: str, target_size: int = 1024):
    """Build DetDataSample for UltraSAM forward pass."""
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

        mask_np = batch["mask"][i].cpu().numpy()
        gt_instances.masks = BitmapMasks([mask_np], mask_np.shape[0], mask_np.shape[1])

        data_sample.gt_instances = gt_instances

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


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        data_samples = build_data_sample(batch, device)

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
    """Validate model."""
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            data_samples = build_data_sample(batch, device)

            data = {"inputs": [img for img in images], "data_samples": data_samples}
            data = model.data_preprocessor(data, False)
            results = model(**data, mode="predict")

            for i, result in enumerate(results):
                if hasattr(result, "pred_instances") and hasattr(result.pred_instances, "masks"):
                    pred_mask = result.pred_instances.masks[0].float()
                    gt_mask = masks[i]

                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = F.interpolate(
                            pred_mask.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape,
                            mode="nearest",
                        ).squeeze()

                    intersection = (pred_mask * gt_mask).sum()
                    union = pred_mask.sum() + gt_mask.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    total_iou += iou.item()

                    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
                    total_dice += dice.item()

                    num_samples += 1

    if num_samples == 0:
        return {"iou": 0, "dice": 0}

    return {"iou": total_iou / num_samples, "dice": total_dice / num_samples}


def finetune_ultrasam_lora(
    data_dir: str,
    ultrasam_ckpt: str,
    output_dir: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
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

    # Build model with LoRA
    print(f"\nBuilding LoRA model (rank={lora_rank}, alpha={lora_alpha})")
    model, cfg = build_lora_model(
        ultrasam_ckpt, lora_rank, lora_alpha, lora_dropout, device=device)

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

    # Check for validation data
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

        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Setup optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training loop
    print(f"\nStarting LoRA finetuning for {epochs} epochs")
    best_val_iou = 0

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Val - IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                # Save only LoRA weights + other trainable weights
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "lora_config": {
                        "rank": lora_rank,
                        "alpha": lora_alpha,
                        "dropout": lora_dropout,
                    },
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
        "lora_config": {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout},
    }, os.path.join(output_dir, "final.pth"))

    print(f"\nTraining complete. Models saved to: {output_dir}")
    print(f"Best validation IoU: {best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Finetune UltraSAM with LoRA")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with cropped data")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout (default: 0.1)")

    # What to train
    parser.add_argument("--train_ffn", action="store_true",
                        help="Also train FFN layers in backbone")
    parser.add_argument("--train_decoder", action="store_true",
                        help="Also train transformer decoder")
    parser.add_argument("--train_mask_head", action="store_true",
                        help="Also train mask prediction head")

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

    ckpt_path = os.path.join(ROOT_DIR, args.ultrasam_ckpt) \
        if not os.path.isabs(args.ultrasam_ckpt) else args.ultrasam_ckpt

    finetune_ultrasam_lora(
        data_dir=args.data_dir,
        ultrasam_ckpt=ckpt_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
