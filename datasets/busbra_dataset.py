"""BUSBRA (Brazilian Breast Ultrasound) dataset."""
import os
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


class BUSBRADataset(Dataset):
    def __init__(self, root_dir, split='train', fold=0, n_folds=5,
                 img_size=224, transform=None, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform

        img_dir = os.path.join(root_dir, 'Images')
        mask_dir = os.path.join(root_dir, 'Masks')

        # Collect all image-mask pairs
        samples = []
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if not f.endswith('.png'):
                    continue
                img_path = os.path.join(img_dir, f)
                # bus_0001-l.png -> mask_0001-l.png
                mask_name = f.replace('bus_', 'mask_')
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    samples.append({'image': img_path, 'mask': mask_path, 'name': os.path.splitext(f)[0]})

        # Try to use predefined CSV splits
        csv_path = os.path.join(root_dir, '5-fold-cv.csv')
        if os.path.exists(csv_path) and n_folds == 5:
            df = pd.read_csv(csv_path)
            sample_names = {s['name']: s for s in samples}
            train_samples = []
            val_samples = []
            for _, row in df.iterrows():
                name = str(row.iloc[0]) if 'name' not in df.columns else str(row['name'])
                fold_col = f'fold_{fold}' if f'fold_{fold}' in df.columns else df.columns[fold + 1] if len(df.columns) > fold + 1 else None
                if fold_col and fold_col in df.columns:
                    if name in sample_names:
                        if row[fold_col] == 'train':
                            train_samples.append(sample_names[name])
                        else:
                            val_samples.append(sample_names[name])
            if train_samples or val_samples:
                self.samples = train_samples if split == 'train' else val_samples
                return

        # Fallback: K-fold split
        indices = np.arange(len(samples))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(indices))
        train_idx, val_idx = splits[fold]
        self.samples = [samples[i] for i in (train_idx if split == 'train' else val_idx)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(sample['image'])
        if image is None:
            raise FileNotFoundError(f"Cannot read {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = (mask > 127).astype(np.uint8)

        original_size = image.shape[:2]

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.astype(np.int64))

        return {
            'image': image,
            'mask': mask,
            'name': sample['name'],
            'original_size': original_size,
        }
