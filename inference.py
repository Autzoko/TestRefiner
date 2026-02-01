#!/usr/bin/env python
"""Full pipeline inference: TransUNet (coarse) -> crop -> UltraSAM (refine).

Usage:
    # Single image
    python inference.py \
        --image path/to/image.png \
        --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
        --ultrasam_ckpt weights/ultrasam_standalone.pth \
        --use_crop --crop_expand 0.3 \
        --save_dir output/predictions

    # Directory of images
    python inference.py \
        --image_dir path/to/images/ \
        --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
        --ultrasam_ckpt weights/ultrasam_standalone.pth \
        --use_crop --crop_expand 0.3 \
        --save_dir output/predictions

    # Evaluate against ground truth
    python inference.py \
        --dataset busi --data_dir /path/to/BUSI \
        --fold 0 \
        --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
        --ultrasam_ckpt weights/ultrasam_standalone.pth \
        --use_crop --crop_expand 0.3 \
        --save_dir output/predictions
"""

import os
import sys
import argparse

import numpy as np
import cv2
import torch
from tqdm import tqdm

from models.transunet import VisionTransformer, CONFIGS
from models.ultrasam import build_ultrasam_vit_b
from utils.crop_utils import (
    crop_and_generate_prompts, uncrop_mask, preprocess_for_ultrasam,
)
from utils.metrics import compute_dice, compute_iou, compute_hd95


def get_args():
    parser = argparse.ArgumentParser(description='UltraRefiner Inference')
    # Input
    parser.add_argument('--image', type=str, default=None, help='Single image path')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory of images')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['busi', 'busbra', 'bus'],
                        help='Use a dataset for evaluation')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    # Model
    parser.add_argument('--transunet_ckpt', type=str, required=True)
    parser.add_argument('--ultrasam_ckpt', type=str, required=True)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=2)
    # Pipeline config
    parser.add_argument('--no_point', action='store_true')
    parser.add_argument('--no_box', action='store_true')
    parser.add_argument('--use_crop', action='store_true')
    parser.add_argument('--crop_expand', type=float, default=0.3)
    # Output
    parser.add_argument('--save_dir', type=str, default='output/predictions')
    parser.add_argument('--save_overlay', action='store_true',
                        help='Save overlay visualization')
    return parser.parse_args()


def load_transunet(args, device):
    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / config_vit.patches.size[0]),
            int(args.img_size / config_vit.patches.size[1]),
        )
    model = VisionTransformer(config_vit, img_size=args.img_size,
                               num_classes=args.num_classes)
    ckpt = torch.load(args.transunet_ckpt, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    return model


def load_ultrasam(args, device):
    model = build_ultrasam_vit_b()
    state_dict = torch.load(args.ultrasam_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


def predict_coarse(transunet, image_np, img_size, device):
    """Run TransUNet to get coarse mask."""
    h, w = image_np.shape[:2]
    resized = cv2.resize(image_np, (img_size, img_size))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = transunet(tensor)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def predict_refined(ultrasam, image_np, coarse_mask, args, device):
    """Run UltraSAM refinement pipeline."""
    use_point = not args.no_point
    use_box = not args.no_box

    (cropped_image, _, point_prompt, point_label,
     box_prompt, box_label, crop_info) = crop_and_generate_prompts(
        image_np, coarse_mask, coarse_mask,
        use_crop=args.use_crop, crop_expand=args.crop_expand,
        use_point=use_point, use_box=use_box,
        target_size=1024,
    )

    sam_input = preprocess_for_ultrasam(cropped_image).unsqueeze(0).to(device)

    point_coords = None
    point_labels = None
    box_coords = None

    if point_prompt is not None:
        point_coords = torch.from_numpy(point_prompt).float().unsqueeze(0).to(device)
        point_labels = torch.from_numpy(point_label).long().unsqueeze(0).to(device)

    if box_prompt is not None:
        bp = box_prompt[0]
        box_coords = torch.tensor(
            [[bp[0, 0], bp[0, 1], bp[1, 0], bp[1, 1]]],
            dtype=torch.float32, device=device,
        )

    with torch.no_grad():
        masks_pred, iou_pred = ultrasam.forward_with_prompts(
            images=sam_input,
            point_coords=point_coords,
            point_labels=point_labels,
            box_coords=box_coords,
            multimask_output=False,
        )

    mask_sigmoid = torch.sigmoid(masks_pred[0, 0]).cpu().numpy()
    full_mask = uncrop_mask(mask_sigmoid, crop_info)
    return full_mask, iou_pred[0, 0].item()


def save_prediction(save_dir, name, pred_mask, image_np=None, save_overlay=False):
    """Save prediction mask and optional overlay."""
    os.makedirs(save_dir, exist_ok=True)
    mask_path = os.path.join(save_dir, f'{name}_pred.png')
    cv2.imwrite(mask_path, pred_mask * 255)

    if save_overlay and image_np is not None:
        overlay = image_np.copy()
        overlay[pred_mask > 0] = (
            overlay[pred_mask > 0] * 0.5 +
            np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)
        overlay_path = os.path.join(save_dir, f'{name}_overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    if args.no_point and args.no_box:
        args.no_point = False

    print("Loading TransUNet...")
    transunet = load_transunet(args, device)
    print("Loading UltraSAM...")
    ultrasam = load_ultrasam(args, device)

    # Collect inputs
    if args.dataset and args.data_dir:
        from datasets import get_dataset
        ds = get_dataset(
            args.dataset, args.data_dir, split='val',
            fold=args.fold, n_folds=args.n_folds,
            img_size=None, transform=None,
        )
        items = []
        for i in range(len(ds)):
            sample = ds[i]
            img = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt = sample['mask'].numpy().astype(np.uint8)
            items.append((sample['name'], img, gt))
    elif args.image:
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(args.image))[0]
        items = [(name, img, None)]
    elif args.image_dir:
        items = []
        for fname in sorted(os.listdir(args.image_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                fpath = os.path.join(args.image_dir, fname)
                img = cv2.imread(fpath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                name = os.path.splitext(fname)[0]
                items.append((name, img, None))
    else:
        print("Error: provide --image, --image_dir, or --dataset + --data_dir")
        sys.exit(1)

    # Run inference
    dice_scores = []
    iou_scores = []
    hd95_scores = []

    for name, image_np, gt_mask in tqdm(items, desc='Inference'):
        coarse_mask = predict_coarse(transunet, image_np, args.img_size, device)
        refined_mask, iou_score = predict_refined(
            ultrasam, image_np, coarse_mask, args, device,
        )

        save_prediction(args.save_dir, name, refined_mask,
                        image_np, args.save_overlay)

        if gt_mask is not None:
            d = compute_dice(refined_mask, gt_mask)
            iou = compute_iou(refined_mask, gt_mask)
            hd = compute_hd95(refined_mask, gt_mask)
            dice_scores.append(d)
            iou_scores.append(iou)
            if hd != float('inf'):
                hd95_scores.append(hd)

    if dice_scores:
        print(f"\nResults ({len(dice_scores)} samples):")
        print(f"  Dice: {np.mean(dice_scores):.4f} +/- {np.std(dice_scores):.4f}")
        print(f"  IoU:  {np.mean(iou_scores):.4f} +/- {np.std(iou_scores):.4f}")
        if hd95_scores:
            print(f"  HD95: {np.mean(hd95_scores):.2f} +/- {np.std(hd95_scores):.2f}")
    else:
        print(f"\nSaved {len(items)} predictions to {args.save_dir}")


if __name__ == '__main__':
    args = get_args()
    run_inference(args)
