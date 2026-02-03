"""Finetune UltraSAM on cropped data for improved crop-mode inference.

This script provides fine-grained control over which model components to
train/freeze during finetuning.

Model Architecture:
    - backbone: ViT-SAM image encoder (heavy, often frozen)
    - prompt_encoder: Encodes point/box prompts
    - decoder: SAM transformer decoder (2 layers)
    - bbox_head: SAMHead mask prediction head

Freezing Options:
    --freeze_backbone: Freeze entire image encoder
    --unfreeze_backbone_layers N: Unfreeze last N layers of backbone
    --freeze_prompt_encoder: Freeze prompt encoder
    --freeze_decoder: Freeze transformer decoder
    --freeze_mask_head: Freeze mask prediction head

Usage:
    python pipeline/08_finetune_ultrasam.py \
        --data_dir outputs/crop_data/busi/fold_0 \
        --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
        --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
        --output_dir outputs/finetuned_ultrasam/busi \
        --freeze_backbone \
        --max_iters 5000 \
        --lr 1e-4 \
        --device cuda:0
"""

import argparse
import copy
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "pipeline"))

# Add UltraSam root to sys.path
ULTRASAM_ROOT = os.path.join(ROOT_DIR, "UltraSam")
if ULTRASAM_ROOT not in sys.path:
    sys.path.insert(0, ULTRASAM_ROOT)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_module(module: nn.Module, name: str = "module"):
    """Freeze all parameters in a module."""
    count = 0
    for param in module.parameters():
        param.requires_grad = False
        count += param.numel()
    print(f"  Froze {name}: {count:,} parameters")


def unfreeze_module(module: nn.Module, name: str = "module"):
    """Unfreeze all parameters in a module."""
    count = 0
    for param in module.parameters():
        param.requires_grad = True
        count += param.numel()
    print(f"  Unfroze {name}: {count:,} parameters")


def freeze_backbone_except_last_n_layers(backbone: nn.Module, n_layers: int):
    """Freeze backbone except the last N layers.

    Args:
        backbone: ViT-SAM backbone.
        n_layers: Number of layers to keep unfrozen at the end.
    """
    # First freeze everything
    for param in backbone.parameters():
        param.requires_grad = False

    # Get the layers module
    if hasattr(backbone, "layers"):
        layers = backbone.layers
        num_layers = len(layers)

        if n_layers > 0 and n_layers <= num_layers:
            # Unfreeze last N layers
            for i in range(num_layers - n_layers, num_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            print(f"  Backbone: frozen except last {n_layers} layers")
        else:
            print(f"  Backbone: fully frozen (n_layers={n_layers}, total={num_layers})")

    # Also unfreeze the neck if present
    if hasattr(backbone, "neck") and n_layers > 0:
        for param in backbone.neck.parameters():
            param.requires_grad = True
        print("  Backbone neck: unfrozen")


def apply_freeze_strategy(
    model: nn.Module,
    freeze_backbone: bool = False,
    unfreeze_backbone_layers: int = 0,
    freeze_prompt_encoder: bool = False,
    freeze_decoder: bool = False,
    freeze_mask_head: bool = False,
):
    """Apply freezing strategy to UltraSAM model.

    Args:
        model: UltraSAM model.
        freeze_backbone: Freeze entire backbone.
        unfreeze_backbone_layers: If freeze_backbone, unfreeze last N layers.
        freeze_prompt_encoder: Freeze prompt encoder.
        freeze_decoder: Freeze transformer decoder.
        freeze_mask_head: Freeze mask prediction head.
    """
    print("\nApplying freeze strategy...")

    # Handle DataParallel or DistributedDataParallel
    if hasattr(model, "module"):
        model = model.module

    # Backbone
    if freeze_backbone:
        if unfreeze_backbone_layers > 0:
            freeze_backbone_except_last_n_layers(model.backbone, unfreeze_backbone_layers)
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

    # Mask head (bbox_head in mmdet terminology)
    if freeze_mask_head:
        freeze_module(model.bbox_head, "bbox_head (mask head)")
    else:
        print("  Mask head: trainable")

    # Print parameter summary
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    print(f"\nParameter summary:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")


def build_crop_config(
    base_config_path: str,
    data_dir: str,
    max_iters: int = 5000,
    batch_size: int = 4,
    lr: float = 1e-4,
    val_interval: int = 500,
):
    """Build config for cropped data finetuning.

    Args:
        base_config_path: Path to base UltraSAM config.
        data_dir: Directory with cropped data (fold_X/ or combined/).
        max_iters: Maximum training iterations.
        batch_size: Training batch size.
        lr: Learning rate.
        val_interval: Validation interval.

    Returns:
        mmengine Config object.
    """
    from mmengine.config import Config

    cfg = Config.fromfile(base_config_path)

    # Update data paths
    images_dir = os.path.join(data_dir, "images")
    train_ann = os.path.join(data_dir, "annotations", "train.json")
    val_ann = os.path.join(data_dir, "annotations", "val.json")

    # Check if val.json exists; if not, use train.json for validation
    if not os.path.exists(val_ann):
        val_ann = train_ann
        print(f"Warning: No val.json found, using train.json for validation")

    # Update dataloader configs
    cfg.train_dataloader = dict(
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(
            type='CocoDataset',
            metainfo={'classes': ('object',)},
            data_root=data_dir,
            data_prefix=dict(img='images/'),
            ann_file=train_ann,
            filter_cfg=dict(filter_empty_gt=True),
            pipeline=cfg.train_dataloader.dataset.pipeline if hasattr(cfg, 'train_dataloader') else None,
        ),
    )

    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            metainfo={'classes': ('object',)},
            data_root=data_dir,
            data_prefix=dict(img='images/'),
            ann_file=val_ann,
            test_mode=True,
        ),
    )

    cfg.test_dataloader = cfg.val_dataloader

    # Update evaluators
    cfg.val_evaluator = dict(
        type='CocoMetric',
        metric=['segm'],
        format_only=False,
        ann_file=val_ann,
    )
    cfg.test_evaluator = cfg.val_evaluator

    # Update training config
    cfg.train_cfg = dict(
        type='IterBasedTrainLoop',
        max_iters=max_iters,
        val_interval=val_interval,
    )

    # Update optimizer
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
        clip_grad=dict(max_norm=0.1, norm_type=2),
    )

    # Update learning rate schedule (cosine annealing for finetuning)
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.01,
            by_epoch=False,
            begin=0,
            end=min(500, max_iters // 10),
        ),
        dict(
            type='CosineAnnealingLR',
            begin=min(500, max_iters // 10),
            end=max_iters,
            by_epoch=False,
            eta_min=lr * 0.01,
        ),
    ]

    # Update default hooks
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=val_interval, max_keep_ckpts=3),
        sampler_seed=dict(type='DistSamplerSeedHook'),
    )

    return cfg


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
    max_iters: int = 5000,
    batch_size: int = 4,
    lr: float = 1e-4,
    val_interval: int = 500,
    device: str = "cuda:0",
    resume: Optional[str] = None,
):
    """Finetune UltraSAM on cropped data.

    Args:
        data_dir: Directory with cropped data.
        ultrasam_config: Path to UltraSAM config.
        ultrasam_ckpt: Path to UltraSAM checkpoint.
        output_dir: Output directory for finetuned model.
        freeze_backbone: Freeze image encoder.
        unfreeze_backbone_layers: Unfreeze last N backbone layers.
        freeze_prompt_encoder: Freeze prompt encoder.
        freeze_decoder: Freeze transformer decoder.
        freeze_mask_head: Freeze mask prediction head.
        max_iters: Maximum training iterations.
        batch_size: Training batch size.
        lr: Learning rate.
        val_interval: Validation interval.
        device: CUDA device.
        resume: Path to checkpoint to resume from.
    """
    from mmengine.config import Config
    from mmengine.runner import Runner

    # Register all modules
    from mmdet.utils import register_all_modules
    register_all_modules()

    # Load and process custom imports
    cfg = Config.fromfile(ultrasam_config)
    if hasattr(cfg, "custom_imports"):
        import importlib
        modules = cfg.custom_imports.get("imports", [])
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                print(f"Warning: Failed to import {mod_name}: {e}")

    # Apply MonkeyPatch for custom attention
    try:
        from endosam.models.utils.custom_functional import (
            multi_head_attention_forward as custom_mha_forward,
        )
        torch.nn.functional.multi_head_attention_forward = custom_mha_forward
        print("Applied UltraSAM MonkeyPatch for multi_head_attention_forward")
    except ImportError as e:
        print(f"Warning: Could not apply MonkeyPatch: {e}")

    # Build config for cropped data
    cfg = build_crop_config(
        ultrasam_config, data_dir, max_iters, batch_size, lr, val_interval)

    # Set load_from to pretrained checkpoint
    cfg.load_from = ultrasam_ckpt

    # Set work_dir
    cfg.work_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    cfg.device = device

    # Add custom freeze hook
    freeze_hook_config = dict(
        type='FreezeUltraSAMHook',
        freeze_backbone=freeze_backbone,
        unfreeze_backbone_layers=unfreeze_backbone_layers,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_decoder=freeze_decoder,
        freeze_mask_head=freeze_mask_head,
    )

    if 'custom_hooks' not in cfg or cfg.custom_hooks is None:
        cfg.custom_hooks = []
    cfg.custom_hooks.append(freeze_hook_config)

    # Resume from checkpoint if specified
    if resume:
        cfg.resume = True
        cfg.load_from = resume

    # Save config
    cfg.dump(os.path.join(output_dir, "config.py"))
    print(f"\nConfig saved to: {os.path.join(output_dir, 'config.py')}")

    # Build and run
    runner = Runner.from_cfg(cfg)
    runner.train()

    print(f"\nFinetuning complete. Model saved to: {output_dir}")


# Register custom freeze hook
from mmdet.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class FreezeUltraSAMHook(Hook):
    """Hook to apply freezing strategy to UltraSAM during training."""

    def __init__(
        self,
        freeze_backbone: bool = False,
        unfreeze_backbone_layers: int = 0,
        freeze_prompt_encoder: bool = False,
        freeze_decoder: bool = False,
        freeze_mask_head: bool = False,
    ):
        self.freeze_backbone = freeze_backbone
        self.unfreeze_backbone_layers = unfreeze_backbone_layers
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_mask_head = freeze_mask_head

    def before_train(self, runner):
        """Apply freeze strategy before training starts."""
        apply_freeze_strategy(
            runner.model,
            freeze_backbone=self.freeze_backbone,
            unfreeze_backbone_layers=self.unfreeze_backbone_layers,
            freeze_prompt_encoder=self.freeze_prompt_encoder,
            freeze_decoder=self.freeze_decoder,
            freeze_mask_head=self.freeze_mask_head,
        )

    def after_load_checkpoint(self, runner, checkpoint):
        """Reapply freeze strategy after loading checkpoint."""
        self.before_train(runner)


def main():
    parser = argparse.ArgumentParser(description="Finetune UltraSAM on cropped data")

    # Data and model paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with cropped data (fold_X/ or combined/)")
    parser.add_argument("--ultrasam_config", type=str, required=True,
                        help="Path to UltraSAM config")
    parser.add_argument("--ultrasam_ckpt", type=str, required=True,
                        help="Path to UltraSAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for finetuned model")

    # Freeze options
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze image encoder (backbone)")
    parser.add_argument("--unfreeze_backbone_layers", type=int, default=0,
                        help="If --freeze_backbone, unfreeze last N layers (default: 0)")
    parser.add_argument("--freeze_prompt_encoder", action="store_true",
                        help="Freeze prompt encoder")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze transformer decoder")
    parser.add_argument("--freeze_mask_head", action="store_true",
                        help="Freeze mask prediction head")

    # Convenience presets
    parser.add_argument("--freeze_all_but_head", action="store_true",
                        help="Freeze everything except mask head (common for finetuning)")
    parser.add_argument("--freeze_all_but_decoder_head", action="store_true",
                        help="Freeze backbone and prompt encoder only")

    # Training parameters
    parser.add_argument("--max_iters", type=int, default=5000,
                        help="Maximum training iterations (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--val_interval", type=int, default=500,
                        help="Validation interval (default: 500)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device (default: cuda:0)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

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
        freeze_mask_head = False

    if args.freeze_all_but_decoder_head:
        freeze_backbone = True
        freeze_prompt_encoder = True
        freeze_decoder = False
        freeze_mask_head = False

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
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        val_interval=args.val_interval,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
