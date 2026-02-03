#!/bin/bash
# Run the full crop finetuning pipeline for UltraSAM
#
# This script:
# 1. Generates cropped training data from preprocessed images
# 2. Finetunes UltraSAM on the cropped data
# 3. Runs inference with the finetuned model
#
# Usage:
#   ./pipeline/run_crop_finetune.sh <dataset> [options]
#
# Example:
#   ./pipeline/run_crop_finetune.sh busi --freeze_backbone --epochs 50

set -e

# Default parameters
DATASET=${1:-"busi"}
CROP_EXPAND=${CROP_EXPAND:-0.5}
N_FOLDS=${N_FOLDS:-5}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-4}
DEVICE=${DEVICE:-"cuda:0"}
FREEZE_OPTS=${FREEZE_OPTS:-"--freeze_backbone"}

# Paths
ROOT_DIR=$(dirname $(dirname $(realpath $0)))
DATA_DIR="${ROOT_DIR}/outputs/preprocessed/${DATASET}"
CROP_DATA_DIR="${ROOT_DIR}/outputs/crop_data/${DATASET}"
FINETUNE_OUTPUT_DIR="${ROOT_DIR}/outputs/finetuned_ultrasam/${DATASET}"
ULTRASAM_CONFIG="${ROOT_DIR}/UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py"
ULTRASAM_CKPT="${ROOT_DIR}/UltraSam/weights/UltraSam.pth"

echo "=========================================="
echo "UltraSAM Crop Finetuning Pipeline"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Crop expand: ${CROP_EXPAND}"
echo "Folds: ${N_FOLDS}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Device: ${DEVICE}"
echo "Freeze options: ${FREEZE_OPTS}"
echo "=========================================="

# Step 1: Generate cropped training data
echo ""
echo "Step 1: Generating cropped training data..."
python "${ROOT_DIR}/pipeline/07_generate_crop_data.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${CROP_DATA_DIR}" \
    --crop_expand ${CROP_EXPAND} \
    --n_folds ${N_FOLDS}

# Step 2: Finetune on combined data (or per-fold)
echo ""
echo "Step 2: Finetuning UltraSAM on cropped data..."

# Check if we should train per fold or combined
if [ "${TRAIN_PER_FOLD:-false}" = "true" ]; then
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        echo ""
        echo "Training fold ${fold}..."
        python "${ROOT_DIR}/pipeline/08_finetune_ultrasam_standalone.py" \
            --data_dir "${CROP_DATA_DIR}/fold_${fold}" \
            --ultrasam_config "${ULTRASAM_CONFIG}" \
            --ultrasam_ckpt "${ULTRASAM_CKPT}" \
            --output_dir "${FINETUNE_OUTPUT_DIR}/fold_${fold}" \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --device ${DEVICE} \
            ${FREEZE_OPTS}
    done
else
    # Train on combined data
    python "${ROOT_DIR}/pipeline/08_finetune_ultrasam_standalone.py" \
        --data_dir "${CROP_DATA_DIR}/combined" \
        --ultrasam_config "${ULTRASAM_CONFIG}" \
        --ultrasam_ckpt "${ULTRASAM_CKPT}" \
        --output_dir "${FINETUNE_OUTPUT_DIR}/combined" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --device ${DEVICE} \
        ${FREEZE_OPTS}
fi

echo ""
echo "=========================================="
echo "Finetuning complete!"
echo "=========================================="
echo ""
echo "To run inference with finetuned model:"
echo ""
echo "python pipeline/05_infer_ultrasam.py \\"
echo "    --prompt_dir outputs/prompts/${DATASET} \\"
echo "    --image_dir ${DATA_DIR}/images_fullres \\"
echo "    --ultrasam_config ${ULTRASAM_CONFIG} \\"
echo "    --ultrasam_ckpt ${FINETUNE_OUTPUT_DIR}/combined/best.pth \\"
echo "    --output_dir outputs/ultrasam_preds_finetuned/${DATASET} \\"
echo "    --prompt_type box --crop --crop_expand ${CROP_EXPAND}"
