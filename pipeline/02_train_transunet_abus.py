"""Train TransUNet on ABUS dataset with predefined train/val/test splits.

Usage:
    python pipeline/02_train_transunet_abus.py \
        --data_dir outputs/preprocessed/abus \
        --output_dir outputs/transunet_models/abus \
        --max_epochs 1000 --batch_size 4 --base_lr 0.01 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add TransUNet to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "TransUNet"))

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class UltrasoundDataset(Dataset):
    """Dataset that reads npz files, resizes to target size, and applies augmentation."""

    def __init__(self, npz_paths, img_size=224, augment=False):
        self.npz_paths = npz_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        data = np.load(self.npz_paths[idx])
        image = data["image"]  # H, W, 3 uint8
        label = data["label"]  # H, W uint8

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0).copy()
                label = np.flip(label, axis=0).copy()
            if np.random.rand() > 0.5:
                k = np.random.choice([1, 2, 3])
                image = np.rot90(image, k).copy()
                label = np.rot90(label, k).copy()

        # Normalize to [0, 1] and convert to CHW tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))  # 3, H, W
        label = torch.from_numpy(label.astype(np.int64))     # H, W

        return image, label


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.softmax(pred_logits, dim=1)[:, 1]  # foreground probability
        target_f = target.float()
        intersection = (pred * target_f).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def build_model(config, img_size=224, n_classes=2, pretrained_path=None):
    """Build TransUNet model."""
    config_vit = CONFIGS_ViT_seg[config]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3

    if config.find("R50") != -1:
        config_vit.patches.grid = (
            int(img_size / config_vit.patches.size[0]),
            int(img_size / config_vit.patches.size[1]),
        )

    model = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_from(weights=np.load(pretrained_path))
        print(f"Loaded pretrained weights from {pretrained_path}")

    return model


def train_model(train_paths, val_paths, output_dir, args):
    """Train the model."""
    os.makedirs(output_dir, exist_ok=True)

    # Save split info
    train_names = [os.path.splitext(os.path.basename(p))[0] for p in train_paths]
    val_names = [os.path.splitext(os.path.basename(p))[0] for p in val_paths]
    with open(os.path.join(output_dir, "split.json"), "w") as f:
        json.dump({"train": train_names, "val": val_names}, f, indent=2)

    # Datasets and loaders
    train_ds = UltrasoundDataset(train_paths, img_size=args.img_size, augment=True)
    val_ds = UltrasoundDataset(val_paths, img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Model
    pretrained_path = os.path.join(
        ROOT_DIR, "TransUNet", "model", "vit_checkpoint", "imagenet21k", "R50+ViT-B_16.npz"
    )
    model = build_model(
        args.vit_name, img_size=args.img_size, n_classes=args.n_classes,
        pretrained_path=pretrained_path,
    )
    model = model.to(args.device)

    # Loss, optimizer
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4
    )

    best_val_dice = 0.0
    max_iter = args.max_epochs * len(train_loader)
    iter_count = 0

    for epoch in range(args.max_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Poly learning rate decay
            lr = args.base_lr * (1 - iter_count / max_iter) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(images)
            loss = 0.5 * ce_loss(logits, labels) + 0.5 * dice_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter_count += 1

        avg_loss = epoch_loss / len(train_loader)

        # --- Validate ---
        val_interval = max(1, args.max_epochs // 20)
        if (epoch + 1) % val_interval == 0 or epoch == args.max_epochs - 1:
            model.eval()
            val_dice_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    # Compute dice per sample
                    for b in range(preds.shape[0]):
                        p = preds[b].cpu().numpy().astype(bool)
                        g = labels[b].cpu().numpy().astype(bool)
                        if not p.any() and not g.any():
                            val_dice_sum += 1.0
                        elif not p.any() or not g.any():
                            val_dice_sum += 0.0
                        else:
                            inter = np.logical_and(p, g).sum()
                            val_dice_sum += 2.0 * inter / (p.sum() + g.sum())
                        val_count += 1

            val_dice = val_dice_sum / max(val_count, 1)
            print(f"Epoch {epoch+1}/{args.max_epochs} | "
                  f"Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {lr:.6f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
                print(f"  -> New best model saved (Dice={val_dice:.4f})")
        else:
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{args.max_epochs} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.6f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final.pth"))
    print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Train TransUNet on ABUS dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with preprocessed ABUS data (train/val/test splits)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for models")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Check for train and val directories
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir):
        print(f"Error: Train directory not found: {train_dir}")
        print("Make sure to run preprocessing first:")
        print("  python pipeline/01_preprocess.py --dataset abus --data_dir /path/to/ABUS --output_dir outputs/preprocessed/abus")
        return

    if not os.path.isdir(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        return

    # Gather npz files
    train_paths = sorted(glob(os.path.join(train_dir, "*.npz")))
    val_paths = sorted(glob(os.path.join(val_dir, "*.npz")))

    if not train_paths:
        print(f"Error: No training npz files found in {train_dir}")
        return

    if not val_paths:
        print(f"Error: No validation npz files found in {val_dir}")
        return

    print(f"Found {len(train_paths)} training samples")
    print(f"Found {len(val_paths)} validation samples")

    print(f"\n{'='*60}")
    print("Training TransUNet on ABUS dataset")
    print(f"{'='*60}")

    best_dice = train_model(train_paths, val_paths, args.output_dir, args)

    print(f"\n{'='*60}")
    print(f"Training Summary: Best Val Dice = {best_dice:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
