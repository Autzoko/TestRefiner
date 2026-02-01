# UltraRefiner

Two-phase ultrasound image segmentation pipeline:

1. **Phase 1 (TransUNet)**: Train a coarse segmentation model on ultrasound datasets
2. **Phase 2 (UltraSAM)**: Refine predictions by cropping around the coarse mask, generating point/box prompts, and feeding into UltraSAM

UltraSAM is re-implemented as standalone PyTorch with no mmdet/mmcv dependencies.

## Project Structure

```
UltraRefiner/
├── models/
│   ├── transunet/                  # R50-ViT-B/16 + UNet decoder
│   │   ├── vit_seg_modeling.py
│   │   ├── vit_seg_modeling_resnet_skip.py
│   │   └── vit_seg_configs.py
│   └── ultrasam/                   # Standalone SAM-based model
│       ├── image_encoder.py        # ViT-B (1024x1024 -> 256x64x64)
│       ├── prompt_encoder.py       # Point/box prompt encoding
│       ├── transformer.py          # Two-way cross-attention decoder
│       ├── mask_decoder.py         # Mask prediction + IoU head
│       ├── common.py               # LayerNorm2d, MLP, positional encoding
│       └── build_ultrasam.py       # Top-level model + factory
├── datasets/
│   ├── busi_dataset.py             # BUSI loader
│   ├── busbra_dataset.py           # BUSBRA loader
│   ├── bus_dataset.py              # BUS loader
│   └── transforms.py               # Augmentations
├── utils/
│   ├── losses.py                   # CE+Dice, Focal+Dice+IoU losses
│   ├── metrics.py                  # Dice, IoU, HD95
│   └── crop_utils.py               # Cropping, prompts, coordinate transforms
├── scripts/
│   ├── download_weights.sh         # Download pretrained weights
│   └── convert_ultrasam_weights.py # mmdet -> standalone weight conversion
├── train_transunet.py              # Phase 1 training
├── train_ultrasam.py               # Phase 2 training / evaluation
├── inference.py                    # Full pipeline inference
└── requirements.txt
```

## 0. Environment Setup

```bash
# Create conda environment (recommended)
conda create -n ultrarefiner python=3.10 -y
conda activate ultrarefiner

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch >= 1.13, CUDA GPU recommended.

## 1. Data Preparation

### 1.1 BUSI (Breast Ultrasound Images Dataset)

Download from: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

Expected directory structure:

```
/path/to/BUSI/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   ├── benign (2).png
│   ├── benign (2)_mask.png
│   └── ...
└── malignant/
    ├── malignant (1).png
    ├── malignant (1)_mask.png
    └── ...
```

- Images are RGB `.png` files
- Masks follow the naming pattern `<name>_mask.png` (or `<name>_mask_1.png` for multiple masks)
- Multiple masks per image are OR-merged automatically
- The `normal/` category (no lesions) is excluded

### 1.2 BUSBRA (Brazilian Breast Ultrasound Dataset)

Expected directory structure:

```
/path/to/BUSBRA/
├── Images/
│   ├── bus_0001-l.png
│   ├── bus_0001-r.png
│   └── ...
├── Masks/
│   ├── mask_0001-l.png
│   ├── mask_0001-r.png
│   └── ...
└── 5-fold-cv.csv          # (optional) predefined fold assignments
```

- Image `bus_XXXX-X.png` maps to mask `mask_XXXX-X.png`
- If `5-fold-cv.csv` is present, it is used for fold splits; otherwise K-fold is applied

### 1.3 BUS (Breast Ultrasound Dataset)

Expected directory structure:

```
/path/to/BUS/
├── original/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── GT/
    ├── 000001.png
    ├── 000002.png
    └── ...
```

- Image filenames in `original/` are matched to the same filename in `GT/`
- Extension fallback is supported (`.png`, `.jpg`, `.bmp`)

### 1.4 No Preprocessing Required

The dataset loaders handle all preprocessing internally:
- Images are loaded as RGB and normalized to `[0, 1]`
- Masks are binarized at threshold 127
- Resizing to the target size (224x224 for TransUNet, 1024x1024 for UltraSAM) is done automatically
- K-fold cross-validation splitting is built into the dataset classes

## 2. Download Pretrained Weights

```bash
# Download TransUNet ViT weights + instructions for UltraSAM
bash scripts/download_weights.sh
```

This script:
1. Downloads `R50+ViT-B_16.npz` (ImageNet-21k pretrained) into `weights/`
2. Prompts you to manually download `UltraSam.pth` from the [UltraSam repo](https://github.com/CAMMA-public/UltraSam)

After placing `UltraSam.pth` in `weights/`, convert it to standalone format:

```bash
python scripts/convert_ultrasam_weights.py \
    --input weights/UltraSam.pth \
    --output weights/ultrasam_standalone.pth \
    --verify
```

The `--verify` flag checks that all converted keys match the standalone model architecture.

After this step, `weights/` should contain:

```
weights/
├── R50+ViT-B_16.npz           # TransUNet pretrained backbone
├── UltraSam.pth                # Original mmdet checkpoint
└── ultrasam_standalone.pth     # Converted standalone weights
```

## 3. Phase 1: Train TransUNet

TransUNet is trained as a 2-class segmentation model (background + lesion) at 224x224 resolution.

### 3.1 Basic Training

```bash
python train_transunet.py \
    --dataset busi \
    --data_dir /path/to/BUSI \
    --fold 0 \
    --n_folds 5 \
    --pretrained_path weights/R50+ViT-B_16.npz \
    --output_dir output/transunet
```

### 3.2 Full Argument Reference

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `busi` | Dataset name: `busi`, `busbra`, or `bus` |
| `--data_dir` | (required) | Path to dataset root directory |
| `--fold` | `0` | K-fold index (0 to n_folds-1) |
| `--n_folds` | `5` | Number of cross-validation folds |
| `--img_size` | `224` | Input image size |
| `--num_classes` | `2` | Number of segmentation classes |
| `--vit_name` | `R50-ViT-B_16` | ViT backbone variant |
| `--n_skip` | `3` | Number of skip connections |
| `--pretrained_path` | `None` | Path to `.npz` pretrained ViT weights |
| `--max_epoch` | `1000` | Total training epochs |
| `--batch_size` | `4` | Training batch size |
| `--base_lr` | `0.01` | Initial learning rate |
| `--weight_decay` | `1e-4` | L2 regularization |
| `--momentum` | `0.9` | SGD momentum |
| `--val_every` | `5` | Validate every N epochs |
| `--num_workers` | `4` | DataLoader workers |
| `--seed` | `123` | Random seed |
| `--output_dir` | `output/transunet` | Output directory |

### 3.3 Training Details

- **Optimizer**: SGD (momentum=0.9, weight_decay=1e-4)
- **Loss**: 0.5 * CrossEntropy + 0.5 * Dice
- **LR schedule**: Polynomial decay `lr = base_lr * (1 - iter/max_iter)^0.9`
- **Augmentation**: Random horizontal flip (p=0.5) + random rotation up to 20 degrees (p=0.5)
- **Validation metrics**: Dice, IoU, HD95 (95th percentile Hausdorff distance)

### 3.4 Output Structure

```
output/transunet/busi/fold_0/
├── model/
│   ├── best.pth        # Best model by validation Dice
│   ├── epoch_100.pth   # Periodic checkpoints every 100 epochs
│   └── final.pth       # Final epoch model
├── tb_logs/            # TensorBoard logs
└── train.log           # Training log
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir output/transunet/busi/fold_0/tb_logs
```

### 3.5 Train All Folds

```bash
for fold in 0 1 2 3 4; do
    python train_transunet.py \
        --dataset busi \
        --data_dir /path/to/BUSI \
        --fold $fold \
        --pretrained_path weights/R50+ViT-B_16.npz \
        --output_dir output/transunet
done
```

## 4. Phase 2: UltraSAM Refinement

Phase 2 uses the trained TransUNet to generate coarse predictions, then crops the image around the prediction, generates point/box prompts, and feeds everything into UltraSAM.

### 4.1 Evaluation Only (Frozen UltraSAM)

Use pretrained UltraSAM weights without any fine-tuning:

```bash
python train_ultrasam.py \
    --dataset busi \
    --data_dir /path/to/BUSI \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --fold 0 \
    --use_crop --crop_expand 0.3 \
    --freeze_ultrasam
```

This runs a single validation epoch and prints Dice/IoU/HD95 metrics.

### 4.2 Fine-tune UltraSAM

```bash
python train_ultrasam.py \
    --dataset busi \
    --data_dir /path/to/BUSI \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --fold 0 \
    --use_crop --crop_expand 0.3 \
    --max_epoch 100 --batch_size 2 --lr 1e-4
```

### 4.3 Prompt Configuration

Control which prompts are sent to UltraSAM:

```bash
# Point only (no box prompt)
python train_ultrasam.py ... --no_box

# Box only (no point prompt)
python train_ultrasam.py ... --no_point

# Both point + box (default)
python train_ultrasam.py ...

# Without cropping (full image, prompts in full-image space)
python train_ultrasam.py ...  # (omit --use_crop)
```

Note: If both `--no_point` and `--no_box` are set, the script falls back to using a center point prompt.

### 4.4 Full Argument Reference

| Argument | Default | Description |
|---|---|---|
| `--transunet_ckpt` | (required) | Path to trained TransUNet checkpoint |
| `--ultrasam_ckpt` | `None` | Path to UltraSAM standalone weights |
| `--no_point` | `False` | Disable point prompt |
| `--no_box` | `False` | Disable box prompt |
| `--use_crop` | `False` | Crop image around TransUNet prediction |
| `--crop_expand` | `0.3` | Expand crop bbox by this ratio on each side |
| `--freeze_ultrasam` | `False` | Freeze UltraSAM (eval-only, no backprop) |
| `--max_epoch` | `100` | Training epochs (1 if frozen) |
| `--batch_size` | `2` | Batch size |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--weight_decay` | `1e-4` | Weight decay |
| `--grad_clip` | `0.1` | Gradient clipping norm |
| `--val_every` | `5` | Validate every N epochs |

### 4.5 Training Details

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss**: 20 * FocalLoss + DiceLoss + IoU MSE
- **Gradient clipping**: max norm 0.1
- **TransUNet**: Always frozen, used for coarse prediction only
- **Per-sample processing**: Each sample is individually cropped and prompted (crops vary in size)

### 4.6 Pipeline Flow (Per Sample)

```
Original image (H x W)
    |
    v
TransUNet (224x224) --> coarse mask --> resize to (H x W)
    |
    v
Crop image around coarse mask bbox (expanded by crop_expand)
    |
    v
Resize crop to 1024x1024, compute prompts in 1024x1024 space:
  - Point: centroid of coarse mask within crop
  - Box: tight bbox of coarse mask within crop
    |
    v
UltraSAM (1024x1024 + prompts) --> 256x256 mask logits
    |
    v
Resize to crop size --> paste into full-size mask (H x W)
```

## 5. Inference

Run the full two-stage pipeline on new images.

### 5.1 Single Image

```bash
python inference.py \
    --image path/to/image.png \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --use_crop --crop_expand 0.3 \
    --save_dir output/predictions
```

### 5.2 Directory of Images

```bash
python inference.py \
    --image_dir path/to/images/ \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --use_crop --crop_expand 0.3 \
    --save_dir output/predictions
```

### 5.3 Evaluate on Dataset (with Ground Truth)

```bash
python inference.py \
    --dataset busi --data_dir /path/to/BUSI \
    --fold 0 \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --use_crop --crop_expand 0.3 \
    --save_dir output/predictions
```

This prints per-dataset Dice, IoU, and HD95 with standard deviations.

### 5.4 Save Overlay Visualizations

Add `--save_overlay` to save green-tinted overlay images alongside prediction masks:

```bash
python inference.py \
    --image_dir path/to/images/ \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --use_crop --crop_expand 0.3 \
    --save_dir output/predictions \
    --save_overlay
```

Output:
```
output/predictions/
├── image1_pred.png       # Binary mask (0/255)
├── image1_overlay.png    # Image with green overlay
├── image2_pred.png
└── image2_overlay.png
```

## 6. Complete Example: BUSI from Scratch

```bash
# 1. Setup
pip install -r requirements.txt
bash scripts/download_weights.sh
# Place UltraSam.pth in weights/, then:
python scripts/convert_ultrasam_weights.py \
    --input weights/UltraSam.pth \
    --output weights/ultrasam_standalone.pth --verify

# 2. Phase 1: Train TransUNet (fold 0)
python train_transunet.py \
    --dataset busi \
    --data_dir /path/to/BUSI \
    --fold 0 \
    --pretrained_path weights/R50+ViT-B_16.npz \
    --max_epoch 1000 --batch_size 4 \
    --output_dir output/transunet

# 3. Phase 2: Evaluate with frozen UltraSAM
python train_ultrasam.py \
    --dataset busi \
    --data_dir /path/to/BUSI \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --fold 0 \
    --use_crop --crop_expand 0.3 \
    --freeze_ultrasam

# 4. (Optional) Fine-tune UltraSAM
python train_ultrasam.py \
    --dataset busi \
    --data_dir /path/to/BUSI \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --fold 0 \
    --use_crop --crop_expand 0.3 \
    --max_epoch 100 --batch_size 2 --lr 1e-4

# 5. Inference on new images
python inference.py \
    --image_dir /path/to/new_images/ \
    --transunet_ckpt output/transunet/busi/fold_0/model/best.pth \
    --ultrasam_ckpt weights/ultrasam_standalone.pth \
    --use_crop --crop_expand 0.3 \
    --save_dir output/predictions --save_overlay
```

## 7. Weight Conversion Details

The `convert_ultrasam_weights.py` script maps mmdet checkpoint keys to standalone model keys:

| mmdet prefix | Standalone prefix | Component |
|---|---|---|
| `backbone.*` | `image_encoder.*` | ViT-B image encoder |
| `prompt_encoder.*` | `prompt_encoder.*` | Prompt encoder |
| `decoder.*` | `transformer_decoder.*` | Two-way transformer |
| `bbox_head.*` | `mask_decoder.*` | Mask decoder + IoU head |

To inspect the key mapping:

```bash
# Print source checkpoint keys
python scripts/convert_ultrasam_weights.py \
    --input weights/UltraSam.pth \
    --output weights/ultrasam_standalone.pth \
    --print-src-keys

# Print converted keys
python scripts/convert_ultrasam_weights.py \
    --input weights/UltraSam.pth \
    --output weights/ultrasam_standalone.pth \
    --print-dst-keys
```
