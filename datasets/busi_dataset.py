"""BUSI (Breast Ultrasound Images) dataset."""
import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


class BUSIDataset(Dataset):
    def __init__(self, root_dir, split='train', fold=0, n_folds=5,
                 img_size=224, transform=None, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # Collect samples from benign and malignant folders
        samples = []
        for category in ['benign', 'malignant']:
            cat_dir = os.path.join(root_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            # Find all images (not masks)
            for f in sorted(os.listdir(cat_dir)):
                if f.endswith('.png') and '_mask' not in f:
                    img_path = os.path.join(cat_dir, f)
                    base = os.path.splitext(f)[0]
                    # Find all corresponding masks
                    mask_paths = sorted(glob.glob(os.path.join(cat_dir, f"{base}_mask*.png")))
                    if mask_paths:
                        samples.append({'image': img_path, 'masks': mask_paths, 'name': f"{category}/{base}"})

        # K-fold split
        indices = np.arange(len(samples))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(indices))
        train_idx, val_idx = splits[fold]

        if split == 'train':
            self.samples = [samples[i] for i in train_idx]
        else:
            self.samples = [samples[i] for i in val_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image'])
        if image is None:
            raise FileNotFoundError(f"Cannot read {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and merge masks
        mask = None
        for mp in sample['masks']:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = (m > 127).astype(np.uint8)
                mask = m if mask is None else np.maximum(mask, m)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        original_size = image.shape[:2]

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))  # (3, H, W)
        mask = torch.from_numpy(mask.astype(np.int64))  # (H, W)

        return {
            'image': image,
            'mask': mask,
            'name': sample['name'],
            'original_size': original_size,
        }
