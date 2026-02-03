# UltraRefiner

A cross-validation pipeline that trains **TransUNet** on ultrasound segmentation datasets (BUSI, BUSBRA, etc.), then uses the predictions as prompts to drive **UltraSAM** inference — comparing both methods on Dice, IoU, and HD95.

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 0 | `pipeline/00_setup.sh` | Clone TransUNet & UltraSAM repos, download pretrained weights |
| 1 | `pipeline/01_preprocess.py` | Convert raw PNG datasets to `.npz` format |
| 2 | `pipeline/02_train_transunet.py` | Train TransUNet with 5-fold cross-validation |
| 3 | `pipeline/03_infer_transunet.py` | Run TransUNet inference on all validation folds |
| 4 | `pipeline/04_generate_prompts.py` | Extract bounding-box / point prompts from TransUNet predictions |
| 5 | `pipeline/05_infer_ultrasam.py` | Run UltraSAM with generated prompts (supports crop mode) |
| 6 | `pipeline/06_compare.py` | Compare metrics and optionally visualize results |
| 7 | `pipeline/07_generate_crop_data.py` | Generate cropped training data for UltraSAM finetuning |
| 8 | `pipeline/08_finetune_ultrasam_standalone.py` | Finetune UltraSAM on cropped data |

## Data Flow

```
                     ┌──────────────────┐
Raw Dataset ──────►  │ 01_preprocess.py │ ──► .npz files + full-res PNGs
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────────┐
                     │ 02_train_transunet.py │ ──► best.pth per fold
                     └──────────────────────┘
                              │
                              ▼
                     ┌──────────────────────┐
                     │ 03_infer_transunet.py │ ──► prediction masks + GT masks
                     └──────────────────────┘
                              │
                              ▼
                     ┌────────────────────────┐
                     │ 04_generate_prompts.py  │ ──► box & point prompts (from predictions)
                     └────────────────────────┘
                              │
                              ▼
                     ┌──────────────────────┐
                     │ 05_infer_ultrasam.py  │ ──► UltraSAM prediction masks
                     └──────────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  06_compare.py   │ ──► metrics CSV + visualizations
                     └──────────────────┘
```

Prompts are generated entirely from TransUNet predictions — ground truth is never used for prompt generation. GT masks are only used in the final comparison step for metric computation.

## Directory Structure

```
UltraRefiner/
├── install_env.sh              # One-command environment setup
├── pipeline/
│   ├── 00_setup.sh
│   ├── 01_preprocess.py
│   ├── 02_train_transunet.py
│   ├── 03_infer_transunet.py
│   ├── 04_generate_prompts.py
│   ├── 05_infer_ultrasam.py
│   ├── 06_compare.py
│   ├── 07_generate_crop_data.py        # Generate cropped data for finetuning
│   ├── 08_finetune_ultrasam_standalone.py  # Finetune UltraSAM
│   ├── run_crop_finetune.sh            # Convenience script for full pipeline
│   └── utils/
│       ├── metrics.py          # Dice, IoU, HD95
│       ├── prompt_utils.py     # mask → bbox / centroid, coordinate transforms
│       └── vis_utils.py        # 4-panel comparison figures
├── TransUNet/                  # cloned at setup (git-ignored)
├── UltraSam/                   # cloned at setup (git-ignored)
└── outputs/                    # all runtime outputs (git-ignored)
```

## Environment Setup

```bash
conda activate <your_env>
bash install_env.sh                # CUDA 11.8 (default)
bash install_env.sh --cuda 12.1    # CUDA 12.1
bash install_env.sh --cpu          # CPU-only
```

This installs PyTorch 2.0.0, OpenMMLab suite (mmengine, mmcv, mmdet, mmpretrain), and all pipeline dependencies into the current conda environment. It also clones both repos and downloads pretrained weights.

**Note**: Requires Python 3.8+ and `numpy<2.0` (pinned automatically by the install script for PyTorch 2.0.0 compatibility).

## Quick Start

### 1. Setup

```bash
bash pipeline/00_setup.sh
```

This clones both repositories and downloads the R50+ViT-B_16 pretrained encoder. The UltraSam checkpoint is downloaded automatically from `https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth`.

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

- BUSI: merges multiple masks per image via logical OR, skips the `normal/` category
- BUSBRA: matches `Images/<id>.png` to `Masks/<id>.png`
- Output: `{image, label, original_size}` npz files + full-resolution PNGs

### 3. Train TransUNet (5-fold CV)

```bash
# All 5 folds
python pipeline/02_train_transunet.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/transunet_models/busi \
    --n_folds 5 --max_epochs 1000 --batch_size 4 --base_lr 0.01 \
    --device cuda:0

# Single fold (for quick testing)
python pipeline/02_train_transunet.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/transunet_models/busi \
    --fold 0 --max_epochs 5 --device cuda:0
```

- Config: R50-ViT-B_16, 224x224 input, 2 classes, SGD with poly LR decay
- Loss: 0.5 x CrossEntropy + 0.5 x Dice
- Splits: `KFold(n_splits=5, shuffle=True, random_state=123)`
- Saves `best.pth` + `split.json` per fold

### 4. TransUNet Inference

```bash
python pipeline/03_infer_transunet.py \
    --data_dir outputs/preprocessed/busi \
    --model_dir outputs/transunet_models/busi \
    --output_dir outputs/transunet_preds/busi \
    --device cuda:0
```

- Inferences at 224x224, resizes predictions to original resolution (nearest-neighbor)
- Saves binary prediction PNGs + GT PNGs + per-fold `metrics.json`

### 5. Generate Prompts

```bash
# From TransUNet predictions (default)
python pipeline/04_generate_prompts.py \
    --pred_dir outputs/transunet_preds/busi \
    --output_dir outputs/prompts/busi

# Expanded bounding box (20% on each side)
python pipeline/04_generate_prompts.py \
    --pred_dir outputs/transunet_preds/busi \
    --output_dir outputs/prompts/busi_expand20 \
    --box_expand 0.2

# From ground-truth masks (oracle / upper-bound comparison)
python pipeline/04_generate_prompts.py \
    --pred_dir outputs/transunet_preds/busi \
    --output_dir outputs/prompts/busi_gt \
    --use_gt --box_expand 0.2
```

For each mask, generates:
- **Box prompt**: tight bounding box `[x1, y1, x2, y2]` around foreground, optionally expanded by `--box_expand` ratio (clamped to image bounds)
- **Point prompt**: centroid `[x, y]` of foreground region
- Fallback for empty masks: full-image box, image-center point

By default, prompts come from TransUNet predictions (`fold_*/*_pred.png`). With `--use_gt`, prompts are extracted from ground-truth masks (`fold_*/gt/*_gt.png`) instead — useful for measuring UltraSAM's upper-bound performance with perfect prompts.

All coordinates are in original pixel space. Transformation to UltraSAM's 1024x1024 input space happens at inference time.

### 6. UltraSAM Inference

```bash
# Box prompt only (full image)
python pipeline/05_infer_ultrasam.py \
    --prompt_dir outputs/prompts/busi \
    --image_dir outputs/preprocessed/busi/images_fullres \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/ultrasam_preds/busi \
    --prompt_type box \
    --device cuda:0

# Point prompt only
python pipeline/05_infer_ultrasam.py \
    ... --prompt_type point --output_dir outputs/ultrasam_preds/busi_point

# Both prompts (two instances: one POINT + one BOX, merged via OR)
python pipeline/05_infer_ultrasam.py \
    ... --prompt_type both --output_dir outputs/ultrasam_preds/busi_both

# Crop mode: crop image around prediction bbox, then run inference on the crop
python pipeline/05_infer_ultrasam.py \
    --prompt_dir outputs/prompts/busi \
    --image_dir outputs/preprocessed/busi/images_fullres \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/ultrasam_preds/busi_crop \
    --prompt_type box \
    --crop --crop_expand 0.5 \
    --device cuda:0
```

**Prompt modes:**

| Mode | Behavior |
|------|----------|
| `box` | Single instance with BOX prompt type; box center used as required point placeholder |
| `point` | Single instance with POINT prompt type; full-image box as required box placeholder |
| `both` | Two instances sent to UltraSAM: one POINT + one BOX; output masks merged with logical OR |

**Crop mode (`--crop`):**

Instead of feeding the full image to UltraSAM, crops a region around TransUNet's predicted bounding box and feeds only the crop. This gives UltraSAM higher effective resolution on the region of interest. The `--crop_expand` ratio (default 0.5) controls how much context beyond the prompt box is included — e.g. 0.5 adds 50% of the box width/height on each side. Crop mode can be combined with any `--prompt_type`.

Coordinate transform chain in crop mode:
```
Original pixel space  ──(subtract crop origin)──►  Crop space  ──(scale to 1024)──►  UltraSAM
UltraSAM output mask  ──(SAMHead rescales to crop size)──►  Paste back into full image
```

`--crop_expand` is independent from `--box_expand` (used in step 5): `box_expand` controls how loose the prompt box given to UltraSAM is; `crop_expand` controls how much surrounding image context the model sees.

### 7. Compare Results

```bash
python pipeline/06_compare.py \
    --transunet_dir outputs/transunet_preds/busi \
    --ultrasam_dir outputs/ultrasam_preds/busi \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/comparison/busi \
    --vis
```

- Computes Dice, IoU, HD95 per sample for both TransUNet and UltraSAM
- Aggregates per-fold (mean +/- std) and overall
- Outputs `per_sample.csv` and `metrics_summary.csv`
- `--vis` generates 4-panel figures: image+GT, TransUNet pred+prompts, UltraSAM pred+prompts, prompts only (requires `--prompt_dir`)

## Key Design Decisions

- **Prompts from predictions or GT**: By default, prompts are derived from TransUNet's predicted masks. Use `--use_gt` to generate prompts from ground-truth masks instead, providing an oracle upper-bound for UltraSAM performance.
- **Intermediate npz format**: All preprocessing produces `{image, label, original_size}` npz files for a consistent interface between steps.
- **Original-resolution prompts**: Bounding boxes and points are stored in original pixel coordinates. Transformation to UltraSAM's 1024x1024 input space happens at inference time only.
- **Deterministic splits**: `KFold(shuffle=True, random_state=123)` ensures reproducibility across runs.
- **UltraSAM via mmengine**: Direct `model.predict()` calls with manual module registration and MonkeyPatch application, giving full control over prompt injection.
- **Box expansion**: Optional `--box_expand` ratio loosens the tight bounding box, which can help UltraSAM when TransUNet predictions are slightly misaligned.
- **Crop mode**: Optional `--crop` feeds only the region around the prediction to UltraSAM at higher effective resolution. Prompts are transformed from original space to crop space before being scaled to 1024-space. The output mask is pasted back into a full-size canvas.

## Requirements

- Python 3.8+
- PyTorch 2.0.0
- numpy < 2.0
- OpenCV (`opencv-python`)
- scikit-learn
- matplotlib
- ml-collections
- medpy (optional, for HD95)
- mmengine, mmcv==2.1.0, mmdet, mmpretrain (for UltraSAM inference)

Use `bash install_env.sh` to install all dependencies automatically.

## Metrics

| Metric | Description |
|--------|-------------|
| Dice | 2 x intersection / (pred + gt), 1.0 if both empty |
| IoU | intersection / union, 1.0 if both empty |
| HD95 | 95th-percentile Hausdorff distance (pixels), inf if either empty |

---

## UltraSAM Crop Finetuning

When using crop mode for inference, UltraSAM sees cropped regions that differ from its original training distribution. Finetuning on cropped data can improve performance.

### Finetuning Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 7 | `pipeline/07_generate_crop_data.py` | Generate cropped training data from preprocessed images |
| 8 | `pipeline/08_finetune_ultrasam_standalone.py` | Finetune UltraSAM on cropped data |

### Quick Start (Finetuning)

```bash
# 1. Generate cropped training data
python pipeline/07_generate_crop_data.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/crop_data/busi \
    --crop_expand 0.5 \
    --n_folds 5

# 2. Finetune UltraSAM (freeze backbone, train decoder + head)
python pipeline/08_finetune_ultrasam_standalone.py \
    --data_dir outputs/crop_data/busi/combined \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/finetuned_ultrasam/busi \
    --freeze_backbone \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda:0

# 3. Run inference with finetuned model
python pipeline/05_infer_ultrasam.py \
    --prompt_dir outputs/prompts/busi \
    --image_dir outputs/preprocessed/busi/images_fullres \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt outputs/finetuned_ultrasam/busi/best.pth \
    --output_dir outputs/ultrasam_preds_finetuned/busi \
    --prompt_type box --crop --crop_expand 0.5
```

Or use the convenience script:

```bash
./pipeline/run_crop_finetune.sh busi
```

### Cropped Data Generation

The `07_generate_crop_data.py` script:
- Takes preprocessed images and GT masks
- Computes crop regions around each GT mask (using `compute_crop_box()`)
- Saves cropped images and masks in COCO format
- Creates per-fold splits (same as TransUNet training) + a combined dataset

```bash
python pipeline/07_generate_crop_data.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/crop_data/busi \
    --crop_expand 0.5 \
    --n_folds 5 \
    --min_crop_size 64
```

Output structure:
```
outputs/crop_data/busi/
├── fold_0/
│   ├── images/           # Cropped images
│   ├── annotations/
│   │   ├── train.json    # COCO format
│   │   └── val.json
│   └── split.json
├── fold_1/ ...
├── combined/             # All data combined (no fold split)
│   ├── images/
│   └── annotations/
│       └── train.json
```

### Finetuning Script

The `08_finetune_ultrasam_standalone.py` script provides fine-grained control over which model components to train/freeze.

**Model Architecture:**
- `backbone`: ViT-SAM image encoder (~90M params, often frozen)
- `prompt_encoder`: Encodes point/box prompts (~6M params)
- `decoder`: SAM transformer decoder (2 layers, ~4M params)
- `bbox_head`: SAMHead mask prediction head (~4M params)

**Freezing Options:**

| Flag | Effect |
|------|--------|
| `--freeze_backbone` | Freeze entire image encoder |
| `--unfreeze_backbone_layers N` | If backbone frozen, unfreeze last N layers |
| `--freeze_prompt_encoder` | Freeze prompt encoder |
| `--freeze_decoder` | Freeze transformer decoder |
| `--freeze_mask_head` | Freeze mask prediction head |

**Convenience Presets:**

| Preset | Effect |
|--------|--------|
| `--freeze_all_but_head` | Freeze backbone + prompt_encoder + decoder; train only mask head |
| `--freeze_all_but_decoder_head` | Freeze backbone + prompt_encoder; train decoder + mask head |

**Recommended Strategies:**

1. **Full finetune** (most expensive, best results):
   ```bash
   # No freeze flags - train everything
   python pipeline/08_finetune_ultrasam_standalone.py ...
   ```

2. **Freeze backbone** (recommended for most cases):
   ```bash
   python pipeline/08_finetune_ultrasam_standalone.py ... --freeze_backbone
   ```

3. **Freeze backbone + unfreeze last N layers** (gradual unfreezing):
   ```bash
   python pipeline/08_finetune_ultrasam_standalone.py ... --freeze_backbone --unfreeze_backbone_layers 2
   ```

4. **Train only mask head** (fastest, for small datasets):
   ```bash
   python pipeline/08_finetune_ultrasam_standalone.py ... --freeze_all_but_head
   ```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 4 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--prompt_type` | box | Prompt type for training (box/point/both) |
| `--device` | cuda:0 | CUDA device |

**Outputs:**
- `best.pth`: Best model by validation IoU
- `final.pth`: Final model after all epochs
- `epoch_*.pth`: Checkpoints every 10 epochs

### Using Finetuned Model for Inference

The inference script automatically handles both original and finetuned checkpoint formats:

```bash
# With finetuned model
python pipeline/05_infer_ultrasam.py \
    --prompt_dir outputs/prompts/busi \
    --image_dir outputs/preprocessed/busi/images_fullres \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt outputs/finetuned_ultrasam/busi/best.pth \
    --output_dir outputs/ultrasam_preds_finetuned/busi \
    --prompt_type box --crop --crop_expand 0.5

# Compare with original model
python pipeline/06_compare.py \
    --transunet_dir outputs/transunet_preds/busi \
    --ultrasam_dir outputs/ultrasam_preds_finetuned/busi \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/comparison/busi_finetuned \
    --vis
```

### Finetuning Tips

1. **Start with frozen backbone**: The ViT-SAM backbone is well-pretrained. Start by freezing it and training only the decoder + head. If performance plateaus, try unfreezing the last few backbone layers.

2. **Use appropriate learning rate**: For frozen backbone, use 1e-4. For full finetuning, use 1e-5 or lower.

3. **Match crop_expand values**: Use the same `--crop_expand` value for data generation, finetuning, and inference.

4. **Monitor validation metrics**: The script saves `best.pth` based on validation IoU. Use `--val_interval` to control validation frequency.

5. **Data augmentation**: The standalone script applies random horizontal flips during training. The mmengine-based script uses the full UltraSAM augmentation pipeline.
