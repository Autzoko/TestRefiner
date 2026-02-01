"""Shared data augmentation transforms for ultrasound segmentation."""
import numpy as np
import cv2
from scipy.ndimage import rotate as scipy_rotate


class RandomFlip:
    """Random horizontal and/or vertical flip."""
    def __init__(self, h_flip=True, v_flip=False, p=0.5):
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.p = p

    def __call__(self, image, mask):
        if self.h_flip and np.random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if self.v_flip and np.random.random() < self.p:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask


class RandomRotation:
    """Random rotation by a random angle."""
    def __init__(self, max_angle=20, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            image = scipy_rotate(image, angle, axes=(0, 1), reshape=False, order=3, mode='reflect')
            mask = scipy_rotate(mask.astype(np.float32), angle, axes=(0, 1), reshape=False, order=0, mode='constant')
            mask = (mask > 0.5).astype(np.uint8)
        return image, mask


class Resize:
    """Resize image and mask to target size."""
    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, image, mask):
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        return image, mask


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_transunet_train_transforms(img_size=224):
    return Compose([
        Resize((img_size, img_size)),
        RandomFlip(h_flip=True, v_flip=False, p=0.5),
        RandomRotation(max_angle=20, p=0.5),
    ])


def get_transunet_val_transforms(img_size=224):
    return Compose([
        Resize((img_size, img_size)),
    ])
