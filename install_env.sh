#!/usr/bin/env bash
# ==============================================================================
# UltraRefiner Environment Installation
#
# Installs all dependencies into the current conda environment for:
#   - TransUNet (training & inference)
#   - UltraSAM  (mmengine/mmdet-based inference)
#   - Pipeline utilities (metrics, visualization, etc.)
#
# Usage:
#   conda activate <your_env>
#   bash install_env.sh                # default: CUDA 11.8
#   bash install_env.sh --cuda 12.1    # CUDA 12.1
#   bash install_env.sh --cpu          # CPU-only (no CUDA)
# ==============================================================================
set -euo pipefail

CUDA_VERSION="11.8"
CPU_ONLY=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)  CUDA_VERSION="$2";  shift 2 ;;
        --cpu)   CPU_ONLY=true;      shift   ;;
        -h|--help)
            echo "Usage: bash install_env.sh [--cuda VERSION] [--cpu]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Check conda env is active ────────────────────────────────────────────────
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Error: No conda environment is active."
    echo "Please run: conda activate <your_env>"
    exit 1
fi
echo "=== Installing into conda env: $CONDA_DEFAULT_ENV ==="

# ── Map CUDA version to pip index URL ─────────────────────────────────────────
if $CPU_ONLY; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "=== Mode: CPU-only ==="
else
    case "$CUDA_VERSION" in
        11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        *)
            echo "Unsupported CUDA version: $CUDA_VERSION (supported: 11.8, 12.1, 12.4)"
            exit 1 ;;
    esac
    echo "=== Mode: CUDA $CUDA_VERSION ==="
fi

# ── Step 1: Install PyTorch ──────────────────────────────────────────────────
echo ""
echo "=== [1/5] Installing PyTorch 2.0.0 ==="
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url "$TORCH_INDEX"

# ── Step 2: Install OpenMMLab (for UltraSAM) ────────────────────────────────
echo ""
echo "=== [2/5] Installing OpenMMLab suite (mmengine, mmcv, mmdet, mmpretrain) ==="
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmpretrain

# ── Step 3: Install TransUNet dependencies ───────────────────────────────────
echo ""
echo "=== [3/5] Installing TransUNet dependencies ==="
pip install ml-collections

# ── Step 4: Install shared / pipeline dependencies ───────────────────────────
echo ""
echo "=== [4/5] Installing pipeline dependencies ==="
pip install \
    "numpy<2.0" \
    scipy \
    scikit-learn \
    scikit-image \
    opencv-python \
    matplotlib \
    medpy \
    SimpleITK \
    h5py \
    tqdm \
    tensorboard

# ── Step 5: Clone repos & download weights ───────────────────────────────────
echo ""
echo "=== [5/5] Cloning repos and downloading weights ==="
bash "$SCRIPT_DIR/pipeline/00_setup.sh"

# Download UltraSAM checkpoint
ULTRASAM_WEIGHT="$SCRIPT_DIR/UltraSam/weights/UltraSam.pth"
if [ ! -f "$ULTRASAM_WEIGHT" ]; then
    echo "Downloading UltraSam.pth ..."
    wget -q --show-progress -O "$ULTRASAM_WEIGHT" \
        "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Installation complete!"
echo "============================================================"
echo ""
echo " Environment: $CONDA_DEFAULT_ENV"
echo ""
echo " Before running UltraSAM scripts, set PYTHONPATH:"
echo "   export PYTHONPATH=\$PYTHONPATH:$SCRIPT_DIR/UltraSam"
echo ""
echo " Verify installation:"
echo "   python -c \"import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())\""
echo "   python -c \"import mmengine; print('mmengine', mmengine.__version__)\""
echo "   python -c \"import mmcv; print('mmcv', mmcv.__version__)\""
echo "   python -c \"import mmdet; print('mmdet', mmdet.__version__)\""
echo "============================================================"
