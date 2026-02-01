#!/usr/bin/env python
"""Phase 1: Train TransUNet for coarse segmentation.

Usage:
    python train_transunet.py \
        --dataset busi \
        --data_dir /path/to/BUSI \
        --fold 0 --n_folds 5 \
        --img_size 224 --num_classes 2 \
        --max_epoch 1000 --batch_size 4 \
        --base_lr 0.01 \
        --vit_name R50-ViT-B_16 \
        --pretrained_path weights/R50+ViT-B_16.npz \
        --output_dir output/transunet
"""

import os
import sys
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.transunet import VisionTransformer, CONFIGS
from datasets import get_dataset
from datasets.transforms import get_transunet_train_transforms, get_transunet_val_transforms
from utils.losses import CombinedTransUNetLoss
from utils.metrics import compute_dice, compute_iou, compute_hd95


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Phase 1: Train TransUNet')
    # Dataset
    parser.add_argument('--dataset', type=str, default='busi',
                        choices=['busi', 'busbra', 'bus'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=2)
    # Model
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained ViT .npz weights')
    # Training
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    # Output
    parser.add_argument('--output_dir', type=str, default='output/transunet')
    return parser.parse_args()


def build_model(args):
    """Build TransUNet model."""
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

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.load_from(weights=np.load(args.pretrained_path))
        logging.info(f"Loaded pretrained weights from {args.pretrained_path}")

    return model


def validate(model, val_loader, device, num_classes=2):
    """Run validation and compute metrics."""
    model.eval()
    dice_scores = []
    iou_scores = []
    hd95_scores = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].numpy()

            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            preds = preds.cpu().numpy()

            for i in range(preds.shape[0]):
                pred_i = (preds[i] == 1).astype(np.uint8)
                gt_i = (masks[i] > 0).astype(np.uint8)
                dice_scores.append(compute_dice(pred_i, gt_i))
                iou_scores.append(compute_iou(pred_i, gt_i))
                hd = compute_hd95(pred_i, gt_i)
                if hd != float('inf'):
                    hd95_scores.append(hd)

    results = {
        'dice': np.mean(dice_scores) if dice_scores else 0.0,
        'iou': np.mean(iou_scores) if iou_scores else 0.0,
        'hd95': np.mean(hd95_scores) if hd95_scores else float('inf'),
    }
    return results


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output directory
    exp_dir = os.path.join(args.output_dir, args.dataset, f'fold_{args.fold}')
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

    # Datasets
    train_transform = get_transunet_train_transforms(args.img_size)
    val_transform = get_transunet_val_transforms(args.img_size)

    train_ds = get_dataset(
        args.dataset, args.data_dir, split='train',
        fold=args.fold, n_folds=args.n_folds,
        img_size=args.img_size, transform=train_transform,
    )
    val_ds = get_dataset(
        args.dataset, args.data_dir, split='val',
        fold=args.fold, n_folds=args.n_folds,
        img_size=args.img_size, transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    logging.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    model = build_model(args)
    model = model.to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer
    criterion = CombinedTransUNetLoss(n_classes=args.num_classes)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Training loop
    max_iterations = args.max_epoch * len(train_loader)
    best_dice = 0.0
    global_step = 0

    for epoch in range(args.max_epoch):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.max_epoch}')
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Polynomial LR decay
            lr = args.base_lr * (1.0 - global_step / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr:.6f}')

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        logging.info(f'Epoch {epoch+1}/{args.max_epoch} - Loss: {avg_loss:.4f}, LR: {lr:.6f}')

        # Validation
        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.max_epoch:
            results = validate(model, val_loader, device, args.num_classes)
            writer.add_scalar('val/dice', results['dice'], epoch)
            writer.add_scalar('val/iou', results['iou'], epoch)
            if results['hd95'] != float('inf'):
                writer.add_scalar('val/hd95', results['hd95'], epoch)

            logging.info(
                f"  Val - Dice: {results['dice']:.4f}, "
                f"IoU: {results['iou']:.4f}, "
                f"HD95: {results['hd95']:.2f}"
            )

            if results['dice'] > best_dice:
                best_dice = results['dice']
                save_path = os.path.join(exp_dir, 'model', 'best.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'args': vars(args),
                }, save_path)
                logging.info(f"  Saved best model (Dice={best_dice:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 100 == 0:
            save_path = os.path.join(exp_dir, 'model', f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args),
            }, save_path)

    # Save final model
    save_path = os.path.join(exp_dir, 'model', 'final.pth')
    torch.save({
        'epoch': args.max_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
        'args': vars(args),
    }, save_path)

    writer.close()
    logging.info(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    args = get_args()
    train(args)
