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
| 8a | `pipeline/08_finetune_ultrasam_standalone.py` | Finetune UltraSAM on cropped data (full finetuning) |
| 8b | `pipeline/08_finetune_ultrasam_lora.py` | Finetune UltraSAM with LoRA + optional FFN training |
| 9 | `pipeline/09_infer_ultrasam_abus.py` | Direct UltraSAM inference on ABUS dataset |
| 10 | `pipeline/10_generate_crop_data_abus.py` | Generate crop data from ABUS for finetuning |
| - | `pipeline/02_train_transunet_abus.py` | Train TransUNet on ABUS (predefined splits) |
| - | `pipeline/03_infer_transunet_abus.py` | TransUNet inference on ABUS |

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
│   ├── 08_finetune_ultrasam_standalone.py  # Finetune UltraSAM (full)
│   ├── 08_finetune_ultrasam_lora.py    # Finetune UltraSAM with LoRA
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

# Crop mode with fixed square aspect ratio
python pipeline/05_infer_ultrasam.py \
    ... --crop --crop_expand 0.5 --square

# Crop mode with custom aspect ratio (W/H)
python pipeline/05_infer_ultrasam.py \
    ... --crop --crop_expand 0.5 --aspect_ratio 1.5
```

**Prompt modes:**

| Mode | Behavior |
|------|----------|
| `box` | Single instance with BOX prompt type; box center used as required point placeholder |
| `point` | Single instance with POINT prompt type; full-image box as required box placeholder |
| `both` | Two instances sent to UltraSAM: one POINT + one BOX; output masks merged with logical OR |

**Crop mode (`--crop`):**

Instead of feeding the full image to UltraSAM, crops a region around TransUNet's predicted bounding box and feeds only the crop. This gives UltraSAM higher effective resolution on the region of interest. The `--crop_expand` ratio (default 0.5) controls how much context beyond the prompt box is included — e.g. 0.5 adds 50% of the box width/height on each side. Crop mode can be combined with any `--prompt_type`.

**Fixed aspect ratio options:**
- `--square`: Force square crops (W/H = 1.0)
- `--aspect_ratio N`: Force fixed W/H ratio (e.g., 1.5 for 3:2 crops)

When a fixed aspect ratio is specified, the crop region is expanded from its center to satisfy the ratio while staying within image bounds. This ensures consistent input shapes during finetuning and inference.

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
| 8a | `pipeline/08_finetune_ultrasam_standalone.py` | Finetune UltraSAM on cropped data (full finetuning) |
| 8b | `pipeline/08_finetune_ultrasam_lora.py` | Finetune UltraSAM with LoRA + optional FFN (parameter-efficient) |

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

---

## LoRA Finetuning (Parameter-Efficient)

For efficient finetuning with minimal GPU memory, use LoRA (Low-Rank Adaptation). This method trains only small adapter matrices in the ViT backbone while keeping the original weights frozen.

### Why LoRA?

- **Memory efficient**: Only ~1-15% of parameters are trainable
- **Fast training**: Fewer gradients to compute
- **Less overfitting**: Constrains the model to stay close to pretrained weights
- **Modular**: LoRA weights can be merged back or swapped easily

### LoRA + FFN Strategy

The recommended approach combines LoRA adapters on attention layers with full training of FFN (Feed-Forward Network) layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    ViT-SAM Backbone                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Transformer Block (x12)                             │   │
│  │  ┌───────────────┐    ┌───────────────┐             │   │
│  │  │  Attention    │    │     FFN       │             │   │
│  │  │  Q K V ← LoRA │    │  (trainable)  │             │   │
│  │  └───────────────┘    └───────────────┘             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

- **LoRA on QKV**: Low-rank adapters modify attention without changing pretrained weights
- **FFN trainable**: Feed-forward layers learn domain-specific features

### LoRA Quick Start

```bash
# 1. Generate cropped data with fixed square aspect ratio
python pipeline/07_generate_crop_data.py \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/crop_data_square/busi \
    --crop_expand 0.5 --square \
    --n_folds 5

# 2. LoRA-only finetuning (most efficient)
python pipeline/08_finetune_ultrasam_lora.py \
    --data_dir outputs/crop_data_square/busi/combined \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/finetuned_ultrasam_lora/busi \
    --lora_rank 16 \
    --epochs 50

# 3. LoRA + FFN training (better adaptation)
python pipeline/08_finetune_ultrasam_lora.py \
    --data_dir outputs/crop_data_square/busi/combined \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/finetuned_ultrasam_lora_ffn/busi \
    --lora_rank 16 \
    --train_ffn \
    --epochs 50

# 4. Inference with LoRA model
python pipeline/05_infer_ultrasam.py \
    --prompt_dir outputs/prompts/busi \
    --image_dir outputs/preprocessed/busi/images_fullres \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt outputs/finetuned_ultrasam_lora/busi/best.pth \
    --output_dir outputs/ultrasam_preds_lora/busi \
    --prompt_type box --crop --crop_expand 0.5 --square

# 5. Compare results
python pipeline/06_compare.py \
    --transunet_dir outputs/transunet_preds/busi \
    --ultrasam_dir outputs/ultrasam_preds_lora/busi \
    --data_dir outputs/preprocessed/busi \
    --output_dir outputs/comparison/busi_lora \
    --vis
```

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_rank` | 16 | LoRA rank (higher = more capacity, more params) |
| `--lora_alpha` | 16 | LoRA scaling factor (typically equal to rank) |
| `--lora_dropout` | 0.1 | Dropout rate for LoRA layers |

### What to Train with LoRA

| Flag | Effect | Params Added |
|------|--------|--------------|
| (default) | LoRA on qkv projections only | ~1.2M |
| `--train_ffn` | Also train FFN (mlp) layers in backbone | ~15M |
| `--train_decoder` | Also train transformer decoder | ~4M |
| `--train_mask_head` | Also train mask prediction head | ~4M |

**Recommended configurations:**

1. **LoRA only** (smallest footprint, ~1% of model params):
   ```bash
   python pipeline/08_finetune_ultrasam_lora.py ... --lora_rank 16
   ```

2. **LoRA + FFN** (recommended for domain adaptation):
   ```bash
   python pipeline/08_finetune_ultrasam_lora.py ... --lora_rank 16 --train_ffn
   ```

3. **LoRA + FFN + decoder** (for significant domain shift):
   ```bash
   python pipeline/08_finetune_ultrasam_lora.py ... --lora_rank 16 --train_ffn --train_decoder
   ```

### LoRA vs Full Finetuning

| Aspect | LoRA | Full Finetuning |
|--------|------|-----------------|
| Trainable params | ~1-15M (~1-15%) | ~100M (100%) |
| GPU memory | Low | High |
| Training speed | Fast | Slow |
| Risk of catastrophic forgetting | Low | Higher |
| Best for | Domain adaptation | Large dataset, significant distribution shift |

### Fixed Aspect Ratio for Consistent Crops

When using LoRA finetuning, it's recommended to use a fixed aspect ratio for crops to ensure consistent input shapes:

```bash
# Generate square crops
python pipeline/07_generate_crop_data.py ... --square

# Or specify custom W/H ratio
python pipeline/07_generate_crop_data.py ... --aspect_ratio 1.0

# Use same settings for inference
python pipeline/05_infer_ultrasam.py ... --crop --square
```

The aspect ratio is enforced by:
1. Computing the initial crop box (bbox + expansion)
2. Expanding the smaller dimension to match the target ratio
3. Clamping to image bounds
4. If clamping breaks the ratio, shrinking the larger dimension

### LoRA Finetuning Tips

1. **Start with LoRA + FFN**: This combination provides good balance between efficiency and adaptation capability:
   ```bash
   python pipeline/08_finetune_ultrasam_lora.py ... --lora_rank 16 --train_ffn
   ```

2. **Use consistent settings**: Match `--crop_expand` and `--square`/`--aspect_ratio` between data generation and inference.

3. **Monitor training**: The script validates every 5 epochs and saves `best.pth` based on IoU.

4. **Adjust LoRA rank**: Higher rank (32, 64) for more complex domain shifts; lower rank (8, 16) for subtle adaptations.

5. **Learning rate**: Default 1e-4 works well for LoRA. Reduce to 1e-5 if training is unstable.

### Checkpoint Format

LoRA finetuned checkpoints contain:
```python
{
    "epoch": int,
    "model_state_dict": {...},  # Full model including LoRA weights
    "lora_config": {
        "rank": 16,
        "alpha": 16,
        "dropout": 0.1
    },
    "val_iou": float
}
```

The inference script (`05_infer_ultrasam.py`) automatically detects and loads both original UltraSAM checkpoints and LoRA-finetuned checkpoints

---

## ABUS Dataset (3D→2D Slices)

The ABUS (Automated Breast Ultrasound System) dataset uses predefined Train/Validation/Test splits instead of k-fold cross-validation.

### Dataset Structure

```
ABUS/
├── Train/
│   ├── malignant (020)_342.png       # Image
│   ├── malignant (020)_342_mask.png  # Mask
│   └── ...
├── Validation/
│   └── ...
├── Test/
│   └── ...
├── Train_metadata.csv
├── Validation_metadata.csv
└── Test_metadata.csv
```

### ABUS Quick Start

```bash
# 1. Preprocess ABUS dataset
python pipeline/01_preprocess.py \
    --dataset abus \
    --data_dir /path/to/ABUS \
    --output_dir outputs/preprocessed/abus
```

### TransUNet on ABUS

```bash
# Train TransUNet
python pipeline/02_train_transunet_abus.py \
    --data_dir outputs/preprocessed/abus \
    --output_dir outputs/transunet_models/abus \
    --max_epochs 1000 --batch_size 4 --base_lr 0.01 \
    --device cuda:0

# Run TransUNet inference on test set
python pipeline/03_infer_transunet_abus.py \
    --data_dir outputs/preprocessed/abus \
    --model_path outputs/transunet_models/abus/best.pth \
    --output_dir outputs/transunet_preds/abus \
    --split test \
    --device cuda:0
```

### UltraSAM on ABUS (Direct Inference)

```bash
# 2. Direct inference (without finetuning) on test set
python pipeline/09_infer_ultrasam_abus.py \
    --data_dir outputs/preprocessed/abus \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/ultrasam_preds/abus \
    --split test \
    --prompt_type box \
    --device cuda:0

# 3. Direct inference with crop mode
python pipeline/09_infer_ultrasam_abus.py \
    --data_dir outputs/preprocessed/abus \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/ultrasam_preds/abus_crop \
    --split test \
    --prompt_type box \
    --crop --crop_expand 0.5 --square \
    --device cuda:0
```

### ABUS Finetuning

```bash
# 1. Generate cropped training data
python pipeline/10_generate_crop_data_abus.py \
    --data_dir outputs/preprocessed/abus \
    --output_dir outputs/crop_data/abus \
    --crop_expand 0.5 --square

# 2. LoRA finetuning on ABUS
python pipeline/08_finetune_ultrasam_lora.py \
    --data_dir outputs/crop_data/abus/combined \
    --ultrasam_ckpt UltraSam/weights/UltraSam.pth \
    --output_dir outputs/finetuned_ultrasam_lora/abus \
    --lora_rank 16 --train_ffn \
    --epochs 50 \
    --device cuda:0

# 3. Inference with finetuned model
python pipeline/09_infer_ultrasam_abus.py \
    --data_dir outputs/preprocessed/abus \
    --ultrasam_config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py \
    --ultrasam_ckpt outputs/finetuned_ultrasam_lora/abus/best.pth \
    --output_dir outputs/ultrasam_preds/abus_finetuned \
    --split test \
    --prompt_type box \
    --crop --crop_expand 0.5 --square \
    --device cuda:0
```

### ABUS Output Structure

After preprocessing:
```
outputs/preprocessed/abus/
├── train/
│   ├── *.npz                # {image, label, original_size}
│   └── images_fullres/*.png
├── val/
│   ├── *.npz
│   └── images_fullres/*.png
└── test/
    ├── *.npz
    └── images_fullres/*.png
```

After inference:
```
outputs/ultrasam_preds/abus/
└── test/
    ├── predictions/*_pred.png
    ├── gt/*_gt.png
    ├── metrics.json
    └── summary.json
```

### ABUS vs BUSI/BUSBRA Pipeline

| Aspect | ABUS | BUSI/BUSBRA |
|--------|------|-------------|
| Splits | Predefined Train/Val/Test | K-fold CV |
| Preprocessing | `01_preprocess.py --dataset abus` | `01_preprocess.py --dataset busi/busbra` |
| TransUNet train | `02_train_transunet_abus.py` | `02_train_transunet.py` |
| TransUNet infer | `03_infer_transunet_abus.py` | `03_infer_transunet.py` |
| UltraSAM infer | `09_infer_ultrasam_abus.py` | `05_infer_ultrasam.py` |
| Crop data | `10_generate_crop_data_abus.py` | `07_generate_crop_data.py` |
| Finetuning | Same (`08_finetune_ultrasam_lora.py`) | Same |
