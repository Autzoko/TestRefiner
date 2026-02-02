# UltraRefiner

A cross-validation pipeline that trains **TransUNet** on ultrasound segmentation datasets (BUSI, BUSBRA, etc.), then uses the predictions as prompts to drive **UltraSAM** inference — comparing both methods on Dice, IoU, and HD95.

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 0 | `pipeline/00_setup.sh` | Clone TransUNet & UltraSAM repos, download pretrained weights |
| 1 | `pipeline/01_preprocess.py` | Convert raw PNG datasets to `.npz` format |
| 2 | `pipeline/02_train_transunet.py` | Train TransUNet with 5-fold cross-validation |
| 3 | `pipeline/03_infer_transunet.py` | Run TransUNet inference on all validation folds |
| 4 | `pipeline/04_generate_prompts.py` | Extract bounding-box / point prompts from predictions |
| 5 | `pipeline/05_infer_ultrasam.py` | Run frozen UltraSAM with generated prompts |
| 6 | `pipeline/06_compare.py` | Compare metrics and optionally visualize results |

## Directory Structure

```
UltraRefiner/
├── pipeline/
│   ├── 00_setup.sh
│   ├── 01_preprocess.py
│   ├── 02_train_transunet.py
│   ├── 03_infer_transunet.py
│   ├── 04_generate_prompts.py
│   ├── 05_infer_ultrasam.py
│   ├── 06_compare.py
│   └── utils/
│       ├── metrics.py          # Dice, IoU, HD95
│       ├── prompt_utils.py     # mask → bbox / centroid, coordinate transforms
│       └── vis_utils.py        # 4-panel comparison figures
├── TransUNet/                  # cloned at setup (git-ignored)
├── UltraSam/                   # cloned at setup (git-ignored)
└── outputs/                    # all runtime outputs (git-ignored)
```

## Quick Start

### 1. Setup

```bash
bash pipeline/00_setup.sh
```

This clones both repositories and downloads the R50+ViT-B_16 pretrained encoder. You will need to manually download `UltraSam.pth` from the [UltraSam releases](https://github.com/CAMMA-public/UltraSam#pretrained-models) and place it at `UltraSam/weights/UltraSam.pth`.

### 2. Preprocess

```bash
# BUSI dataset
python pipeline/01_preprocess.py \
    --dataset busi \
    --data_dir /path/to/Dataset_BUSI_with_GT \
    --output_dir outputs/preprocessed/busi

# BUSBRA dataset
python pipeline/01_preprocess.py \
    --dataset busbra \
    --data_dir /path/to/BUSBRA \
    --output_dir outputs/preprocessed/busbra
```

### 3. Train TransUNet (5-fold CV)

```bash
# All 5 folds
python pipeline/02_train_transunet.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/transunet_models/busi \
    --n_folds 5 --max_epochs 1000 --batch_size 4 --base_lr 0.01 \
    --device cuda:0

# Single fold (for testing)
python pipeline/02_train_transunet.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/transunet_models/busi \
    --fold 0 --max_epochs 5 --device cuda:0
```

### 4. TransUNet Inference

```bash
python pipeline/03_infer_transunet.py \
    --data_dir outputs/preprocessed/busi \
    --model_dir outputs/transunet_models/busi \
    --output_dir outputs/transunet_preds/busi \
    --device cuda:0
```

### 5. Generate Prompts

```bash
python pipeline/04_generate_prompts.py \
    --pred_dir outputs/transunet_preds/busi \
    --output_dir outputs/prompts/busi
```

### 6. UltraSAM Inference

```bash
python pipeline/05_infer_ultrasam.py \
    --prompt_dir outputs/prompts/busi \
    --image_dir outputs/preprocessed/busi/images_fullres \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/ultrasam_preds/busi \
    --prompt_type box \
    --device cuda:0
```

### 7. Compare Results

```bash
python pipeline/06_compare.py \
    --transunet_dir outputs/transunet_preds/busi \
    --ultrasam_dir outputs/ultrasam_preds/busi \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/comparison/busi \
    --vis
```

## Key Design Decisions

- **Intermediate npz format**: All preprocessing produces `{image, label, original_size}` npz files for a consistent interface.
- **Original-resolution prompts**: Bounding boxes and points are stored in original pixel coordinates. Transformation to UltraSAM's 1024x1024 input space happens at inference time only.
- **Deterministic splits**: `KFold(shuffle=True, random_state=123)` ensures reproducibility across runs.
- **UltraSAM via mmengine**: Direct `model.predict()` calls give full control over prompt injection rather than relying on CLI wrappers.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- scikit-learn
- matplotlib
- numpy
- medpy (optional, for HD95)
- mmengine, mmdet (for UltraSAM inference)

## Metrics

| Metric | Description |
|--------|-------------|
| Dice | 2 x intersection / (pred + gt), 1.0 if both empty |
| IoU | intersection / union, 1.0 if both empty |
| HD95 | 95th-percentile Hausdorff distance (pixels), inf if either empty |
