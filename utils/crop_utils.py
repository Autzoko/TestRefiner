"""Cropping, prompt generation, and coordinate transform utilities."""
import numpy as np
import cv2
import torch


def get_bbox_from_mask(mask, expand_ratio=0.0):
    """Compute bounding box from binary mask with optional expansion.
    Args:
        mask: (H, W) binary numpy array
        expand_ratio: fraction to expand bbox on each side
    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clipped to image bounds.
        Returns None if mask is empty.
    """
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = mask.shape
    rh = rmax - rmin
    rw = cmax - cmin

    rmin = max(0, int(rmin - rh * expand_ratio))
    rmax = min(h - 1, int(rmax + rh * expand_ratio))
    cmin = max(0, int(cmin - rw * expand_ratio))
    cmax = min(w - 1, int(cmax + rw * expand_ratio))

    return cmin, rmin, cmax, rmax  # x1, y1, x2, y2


def get_centroid_from_mask(mask):
    """Compute centroid of binary mask.
    Returns: (cx, cy) in pixel coordinates, or image center if mask is empty.
    """
    if mask.sum() == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    rows, cols = np.where(mask > 0)
    return float(cols.mean()), float(rows.mean())


def crop_and_generate_prompts(image, gt_mask, coarse_mask,
                               use_crop=True, crop_expand=0.3,
                               use_point=True, use_box=True,
                               target_size=1024):
    """Core pipeline: crop image using coarse mask, generate prompts, transform coords.

    Args:
        image: (H, W, 3) or (H, W) numpy array, original image
        gt_mask: (H, W) binary numpy array, ground truth
        coarse_mask: (H, W) binary numpy array, TransUNet prediction
        use_crop: whether to crop based on coarse mask
        crop_expand: bbox expansion ratio
        use_point: generate point prompt
        use_box: generate box prompt
        target_size: UltraSAM input size (1024)

    Returns:
        cropped_image: (target_size, target_size, 3) resized crop
        cropped_mask: (target_size, target_size) resized GT mask
        point_prompt: (1, 2) point in target_size space, or None
        point_label: (1,) label (EmbeddingIndex.POS.value=2), or None
        box_prompt: (1, 2, 2) box corners in target_size space, or None
        box_label: (1, 2) labels (3, 4), or None
        crop_info: dict for uncropping
    """
    h_orig, w_orig = image.shape[:2]

    # Ensure image is 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if use_crop:
        bbox = get_bbox_from_mask(coarse_mask, expand_ratio=crop_expand)
        if bbox is None:
            # Empty mask: use full image
            x1, y1, x2, y2 = 0, 0, w_orig - 1, h_orig - 1
        else:
            x1, y1, x2, y2 = bbox
    else:
        x1, y1, x2, y2 = 0, 0, w_orig - 1, h_orig - 1

    # Crop
    cropped_image = image[y1:y2+1, x1:x2+1].copy()
    cropped_gt = gt_mask[y1:y2+1, x1:x2+1].copy()
    cropped_coarse = coarse_mask[y1:y2+1, x1:x2+1].copy()
    crop_h, crop_w = cropped_image.shape[:2]

    # Scale factors
    scale_x = target_size / crop_w
    scale_y = target_size / crop_h

    # Generate point prompt (centroid of coarse mask in crop space)
    point_prompt = None
    point_label = None
    if use_point:
        cx, cy = get_centroid_from_mask(cropped_coarse)
        point_prompt = np.array([[cx * scale_x, cy * scale_y]], dtype=np.float32)
        point_label = np.array([2], dtype=np.int64)  # EmbeddingIndex.POS

    # Generate box prompt (tight bbox of coarse mask in crop space)
    box_prompt = None
    box_label = None
    if use_box:
        tight_bbox = get_bbox_from_mask(cropped_coarse, expand_ratio=0.0)
        if tight_bbox is None:
            # Use full crop as box
            bx1, by1, bx2, by2 = 0, 0, crop_w - 1, crop_h - 1
        else:
            bx1, by1, bx2, by2 = tight_bbox
        box_prompt = np.array([[[bx1 * scale_x, by1 * scale_y],
                                [bx2 * scale_x, by2 * scale_y]]], dtype=np.float32)
        box_label = np.array([[3, 4]], dtype=np.int64)  # BOX_CORNER_A, BOX_CORNER_B

    # Resize crop to target size
    cropped_image = cv2.resize(cropped_image, (target_size, target_size),
                               interpolation=cv2.INTER_LINEAR)
    cropped_mask = cv2.resize(cropped_gt.astype(np.uint8), (target_size, target_size),
                              interpolation=cv2.INTER_NEAREST)

    crop_info = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'orig_h': h_orig, 'orig_w': w_orig,
        'crop_h': crop_h, 'crop_w': crop_w,
    }

    return cropped_image, cropped_mask, point_prompt, point_label, box_prompt, box_label, crop_info


def uncrop_mask(pred_mask, crop_info, threshold=0.5):
    """Transform predicted mask from cropped+resized space back to original.
    Args:
        pred_mask: (H_pred, W_pred) numpy array (probabilities or binary)
        crop_info: dict from crop_and_generate_prompts
        threshold: binarization threshold
    Returns:
        full_mask: (orig_h, orig_w) binary uint8 mask
    """
    crop_h = crop_info['y2'] - crop_info['y1'] + 1
    crop_w = crop_info['x2'] - crop_info['x1'] + 1

    # Resize back to crop dimensions
    resized = cv2.resize(pred_mask.astype(np.float32), (crop_w, crop_h),
                         interpolation=cv2.INTER_LINEAR)
    binary = (resized > threshold).astype(np.uint8)

    # Place in full image
    full_mask = np.zeros((crop_info['orig_h'], crop_info['orig_w']), dtype=np.uint8)
    full_mask[crop_info['y1']:crop_info['y2']+1,
              crop_info['x1']:crop_info['x2']+1] = binary

    return full_mask


def preprocess_for_ultrasam(image):
    """Normalize image for UltraSAM (ImageNet normalization).
    Args:
        image: (H, W, 3) uint8 or float numpy array
    Returns:
        tensor: (3, H, W) normalized float32 tensor
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    # ImageNet normalization (BGR→RGB handled by default, but our images are already RGB)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image = (image - mean) / std
    # HWC → CHW
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float()
