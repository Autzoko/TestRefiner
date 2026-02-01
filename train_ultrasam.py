#!/usr/bin/env python
"""Phase 2: Train/evaluate UltraSAM with TransUNet-based cropping and prompts.

Usage (evaluation only):
    python train_ultrasam.py \
        --dataset busi --data_dir /path/to/BUSI \
        --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
        --ultrasam_ckpt weights/ultrasam_standalone.pth \
        --fold 0 --use_crop --crop_expand 0.3 \
        --freeze_ultrasam

Usage (fine-tune):
    python train_ultrasam.py \
        --dataset busi --data_dir /path/to/BUSI \
        --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
        --ultrasam_ckpt weights/ultrasam_standalone.pth \
        --fold 0 --use_crop --crop_expand 0.3 \
        --max_epoch 100 --batch_size 2 --lr 1e-4
"""

import os
import sys
import random
import argparse
import logging

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.transunet import VisionTransformer, CONFIGS
from models.ultrasam import UltraSAM, build_ultrasam_vit_b
from datasets import get_dataset
from datasets.transforms import get_transunet_val_transforms
from utils.losses import CombinedUltraSAMLoss
from utils.metrics import (
    compute_dice, compute_iou, compute_hd95, compute_mask_iou_batch,
)
from utils.crop_utils import (
    crop_and_generate_prompts, uncrop_mask, preprocess_for_ultrasam,
)


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser(description='Phase 2: UltraSAM refinement')
    # Dataset
    parser.add_argument('--dataset', type=str, default='busi',
                        choices=['busi', 'busbra', 'bus'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=224,
                        help='TransUNet input size')
    parser.add_argument('--num_classes', type=int, default=2)
    # Model checkpoints
    parser.add_argument('--transunet_ckpt', type=str, required=True,
                        help='Path to trained TransUNet checkpoint')
    parser.add_argument('--ultrasam_ckpt', type=str, default=None,
                        help='Path to UltraSAM standalone weights')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--n_skip', type=int, default=3)
    # Prompt config
    parser.add_argument('--no_point', action='store_true',
                        help='Disable point prompt')
    parser.add_argument('--no_box', action='store_true',
                        help='Disable box prompt')
    parser.add_argument('--use_crop', action='store_true',
                        help='Crop image based on TransUNet prediction')
    parser.add_argument('--crop_expand', type=float, default=0.3,
                        help='Expand ratio for crop bounding box')
    # Training
    parser.add_argument('--freeze_ultrasam', action='store_true',
                        help='Freeze UltraSAM (evaluation-only mode)')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    # Output
    parser.add_argument('--output_dir', type=str, default='output/ultrasam')
    return parser.parse_args()


def load_transunet(args, device):
    """Load frozen TransUNet for coarse prediction."""
    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / config_vit.patches.size[0]),
            int(args.img_size / config_vit.patches.size[1]),
        )

    model = VisionTransformer(
        config_vit, img_size=args.img_size, num_classes=args.num_classes,
    )

    ckpt = torch.load(args.transunet_ckpt, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"Loaded TransUNet from {args.transunet_ckpt}")
    return model


def load_ultrasam(args, device):
    """Load UltraSAM model."""
    model = build_ultrasam_vit_b()

    if args.ultrasam_ckpt and os.path.exists(args.ultrasam_ckpt):
        state_dict = torch.load(args.ultrasam_ckpt, map_location='cpu',
                                weights_only=False)
        result = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded UltraSAM from {args.ultrasam_ckpt}")
        if result.missing_keys:
            logging.warning(f"  Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            logging.warning(f"  Unexpected keys: {len(result.unexpected_keys)}")
    else:
        logging.info("UltraSAM initialized with random weights")

    model = model.to(device)

    if args.freeze_ultrasam:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        logging.info("UltraSAM is FROZEN (evaluation mode)")
    else:
        model.train()
        logging.info("UltraSAM is TRAINABLE")

    return model


def get_coarse_prediction(transunet, image_np, img_size, device):
    """Run TransUNet inference on a single image.

    Args:
        transunet: frozen TransUNet model
        image_np: (H, W, 3) uint8 numpy image (original size)
        img_size: TransUNet input size (224)
        device: torch device

    Returns:
        coarse_mask: (H, W) binary numpy array at original resolution
    """
    h_orig, w_orig = image_np.shape[:2]

    resized = cv2.resize(image_np, (img_size, img_size),
                         interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = transunet(tensor)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        pred = pred.squeeze(0).cpu().numpy()

    coarse_mask = cv2.resize(
        pred.astype(np.uint8), (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST,
    )
    return coarse_mask


def process_single_sample(transunet, image_np, gt_mask_np, args, device):
    """Process one sample through the full pipeline (TransUNet -> crop -> prompts).

    Returns dict with all tensors needed for UltraSAM forward.
    """
    use_point = not args.no_point
    use_box = not args.no_box

    coarse_mask = get_coarse_prediction(
        transunet, image_np, args.img_size, device,
    )

    (cropped_image, cropped_mask,
     point_prompt, point_label,
     box_prompt, box_label,
     crop_info) = crop_and_generate_prompts(
        image_np, gt_mask_np, coarse_mask,
        use_crop=args.use_crop, crop_expand=args.crop_expand,
        use_point=use_point, use_box=use_box,
        target_size=1024,
    )

    sam_input = preprocess_for_ultrasam(cropped_image)

    point_coords_t = None
    point_labels_t = None
    box_coords_t = None

    if point_prompt is not None:
        point_coords_t = torch.from_numpy(point_prompt).float()
        point_labels_t = torch.from_numpy(point_label).long()

    if box_prompt is not None:
        bp = box_prompt[0]  # (2, 2): [[x1,y1],[x2,y2]]
        box_coords_t = torch.tensor(
            [bp[0, 0], bp[0, 1], bp[1, 0], bp[1, 1]], dtype=torch.float32,
        )

    gt_256 = cv2.resize(
        cropped_mask.astype(np.uint8), (256, 256),
        interpolation=cv2.INTER_NEAREST,
    )
    gt_256_t = torch.from_numpy(gt_256).float()

    return {
        'sam_input': sam_input,
        'point_coords': point_coords_t,
        'point_labels': point_labels_t,
        'box_coords': box_coords_t,
        'gt_256': gt_256_t,
        'crop_info': crop_info,
        'coarse_mask': coarse_mask,
    }


def collate_samples(samples, device):
    """Collate processed samples into batched tensors."""
    sam_inputs = torch.stack([s['sam_input'] for s in samples]).to(device)
    gt_256 = torch.stack([s['gt_256'] for s in samples]).to(device)

    has_points = samples[0]['point_coords'] is not None
    point_coords = None
    point_labels = None
    if has_points:
        point_coords = torch.stack(
            [s['point_coords'] for s in samples]
        ).to(device)
        point_labels = torch.stack(
            [s['point_labels'] for s in samples]
        ).to(device)

    has_boxes = samples[0]['box_coords'] is not None
    box_coords = None
    if has_boxes:
        box_coords = torch.stack(
            [s['box_coords'] for s in samples]
        ).to(device)

    return {
        'sam_inputs': sam_inputs,
        'gt_256': gt_256,
        'point_coords': point_coords,
        'point_labels': point_labels,
        'box_coords': box_coords,
    }


def run_epoch(
    transunet, ultrasam, dataloader, args, device,
    optimizer=None, criterion=None, is_train=False,
):
    """Run one epoch (train or val)."""
    if is_train and optimizer is not None:
        ultrasam.train()
    else:
        ultrasam.eval()

    total_loss = 0.0
    all_dice = []
    all_iou = []
    all_hd95 = []
    n_batches = 0

    pbar = tqdm(dataloader, desc='Train' if is_train else 'Val')
    for batch in pbar:
        images_np = []
        gt_masks_np = []

        for i in range(batch['image'].shape[0]):
            img_t = batch['image'][i]
            mask_t = batch['mask'][i]
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_np = mask_t.numpy().astype(np.uint8)
            images_np.append(img_np)
            gt_masks_np.append(mask_np)

        processed = []
        for img_np, gt_np in zip(images_np, gt_masks_np):
            sample = process_single_sample(
                transunet, img_np, gt_np, args, device,
            )
            processed.append(sample)

        collated = collate_samples(processed, device)

        context = torch.enable_grad() if is_train and optimizer else torch.no_grad()
        with context:
            masks_pred, iou_pred = ultrasam.forward_with_prompts(
                images=collated['sam_inputs'],
                point_coords=collated['point_coords'],
                point_labels=collated['point_labels'],
                box_coords=collated['box_coords'],
                multimask_output=False,
            )
            masks_pred = masks_pred.squeeze(1)
            iou_pred = iou_pred.squeeze(1)

            if is_train and criterion is not None and optimizer is not None:
                gt_iou = compute_mask_iou_batch(
                    masks_pred.unsqueeze(1), collated['gt_256'],
                ).squeeze(1)
                loss = criterion(
                    masks_pred, collated['gt_256'],
                    iou_pred, gt_iou,
                )
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        ultrasam.parameters(), args.grad_clip,
                    )
                optimizer.step()
                total_loss += loss.item()

        masks_sigmoid = torch.sigmoid(masks_pred).cpu().numpy()
        for i in range(len(processed)):
            pred_256 = masks_sigmoid[i]
            full_pred = uncrop_mask(pred_256, processed[i]['crop_info'])
            full_gt = gt_masks_np[i]

            all_dice.append(compute_dice(full_pred, full_gt))
            all_iou.append(compute_iou(full_pred, full_gt))
            hd = compute_hd95(full_pred, full_gt)
            if hd != float('inf'):
                all_hd95.append(hd)

        n_batches += 1
        if is_train and criterion is not None:
            pbar.set_postfix(loss=f'{total_loss/n_batches:.4f}')
        else:
            pbar.set_postfix(dice=f'{np.mean(all_dice):.4f}')

    results = {
        'loss': total_loss / max(n_batches, 1),
        'dice': np.mean(all_dice) if all_dice else 0.0,
        'iou': np.mean(all_iou) if all_iou else 0.0,
        'hd95': np.mean(all_hd95) if all_hd95 else float('inf'),
    }
    return results


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.no_point and args.no_box:
        logging.warning("Both --no_point and --no_box set. Using center point as fallback.")
        args.no_point = False

    # Output directory
    suffix = ''
    if args.freeze_ultrasam:
        suffix = '_frozen'
    if args.use_crop:
        suffix += f'_crop{args.crop_expand}'
    if args.no_point:
        suffix += '_nopoint'
    if args.no_box:
        suffix += '_nobox'

    exp_dir = os.path.join(
        args.output_dir, args.dataset, f'fold_{args.fold}{suffix}',
    )
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'model'), exist_ok=True)

    # Logging
    logging.basicConfig(
        filename=os.path.join(exp_dir, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    writer = SummaryWriter(os.path.join(exp_dir, 'tb_logs'))

    logging.info(f"Args: {args}")
    logging.info(f"Device: {device}")

    # Datasets — keep original size, no transform (per-sample processing)
    train_ds = get_dataset(
        args.dataset, args.data_dir, split='train',
        fold=args.fold, n_folds=args.n_folds,
        img_size=None, transform=None,
    )
    val_ds = get_dataset(
        args.dataset, args.data_dir, split='val',
        fold=args.fold, n_folds=args.n_folds,
        img_size=None, transform=None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    logging.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Models
    transunet = load_transunet(args, device)
    ultrasam = load_ultrasam(args, device)

    # Optimizer and loss (only if not frozen)
    optimizer = None
    criterion = None
    if not args.freeze_ultrasam:
        optimizer = optim.AdamW(
            ultrasam.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = CombinedUltraSAMLoss()
        logging.info("Using AdamW optimizer + Combined UltraSAM loss")

    # Training / evaluation loop
    best_dice = 0.0
    for epoch in range(args.max_epoch):
        logging.info(f"\n--- Epoch {epoch+1}/{args.max_epoch} ---")

        if not args.freeze_ultrasam:
            train_results = run_epoch(
                transunet, ultrasam, train_loader, args, device,
                optimizer=optimizer, criterion=criterion, is_train=True,
            )
            writer.add_scalar('train/loss', train_results['loss'], epoch)
            writer.add_scalar('train/dice', train_results['dice'], epoch)
            logging.info(
                f"Train - Loss: {train_results['loss']:.4f}, "
                f"Dice: {train_results['dice']:.4f}"
            )

        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.max_epoch:
            val_results = run_epoch(
                transunet, ultrasam, val_loader, args, device,
                is_train=False,
            )
            writer.add_scalar('val/dice', val_results['dice'], epoch)
            writer.add_scalar('val/iou', val_results['iou'], epoch)
            if val_results['hd95'] != float('inf'):
                writer.add_scalar('val/hd95', val_results['hd95'], epoch)

            logging.info(
                f"Val - Dice: {val_results['dice']:.4f}, "
                f"IoU: {val_results['iou']:.4f}, "
                f"HD95: {val_results['hd95']:.2f}"
            )

            if val_results['dice'] > best_dice:
                best_dice = val_results['dice']
                if not args.freeze_ultrasam:
                    save_path = os.path.join(exp_dir, 'model', 'best.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': ultrasam.state_dict(),
                        'best_dice': best_dice,
                        'args': vars(args),
                    }, save_path)
                    logging.info(f"  Saved best model (Dice={best_dice:.4f})")

        # For frozen mode, only run one epoch
        if args.freeze_ultrasam:
            break

    # Save final model
    if not args.freeze_ultrasam:
        save_path = os.path.join(exp_dir, 'model', 'final.pth')
        torch.save({
            'epoch': args.max_epoch,
            'model_state_dict': ultrasam.state_dict(),
            'best_dice': best_dice,
            'args': vars(args),
        }, save_path)

    writer.close()
    logging.info(f"Done. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    args = get_args()
    train(args)
