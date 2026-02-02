#!/usr/bin/env bash
# ==============================================================================
# UltraRefiner Environment Installation
#
# Creates a conda environment with all dependencies for:
#   - TransUNet (training & inference)
#   - UltraSAM  (mmengine/mmdet-based inference)
#   - Pipeline utilities (metrics, visualization, etc.)
#
# Usage:
#   bash install_env.sh                # default: env name "ultrarefiner", CUDA 11.8
#   bash install_env.sh --name myenv   # custom env name
#   bash install_env.sh --cuda 12.1    # CUDA 12.1
#   bash install_env.sh --cpu          # CPU-only (no CUDA)
# ==============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ENV_NAME="ultrarefiner"
CUDA_VERSION="11.8"
CPU_ONLY=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)  ENV_NAME="$2";      shift 2 ;;
        --cuda)  CUDA_VERSION="$2";  shift 2 ;;
        --cpu)   CPU_ONLY=true;      shift   ;;
        -h|--help)
            echo "Usage: bash install_env.sh [--name ENV_NAME] [--cuda VERSION] [--cpu]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Map CUDA version to pip index URL ─────────────────────────────────────────
if $CPU_ONLY; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "=== Installing CPU-only build ==="
else
    case "$CUDA_VERSION" in
        11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        *)
            echo "Unsupported CUDA version: $CUDA_VERSION (supported: 11.8, 12.1, 12.4)"
            exit 1 ;;
    esac
    echo "=== Installing with CUDA $CUDA_VERSION ==="
fi

# ── Step 1: Create conda environment ─────────────────────────────────────────
echo ""
echo "=== [1/6] Creating conda environment: $ENV_NAME (Python 3.8) ==="
conda create --name "$ENV_NAME" python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Step 2: Install PyTorch ──────────────────────────────────────────────────
echo ""
echo "=== [2/6] Installing PyTorch 2.0.0 ==="
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url "$TORCH_INDEX"

# ── Step 3: Install OpenMMLab (for UltraSAM) ────────────────────────────────
echo ""
echo "=== [3/6] Installing OpenMMLab suite (mmengine, mmcv, mmdet, mmpretrain) ==="
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmpretrain

# ── Step 4: Install TransUNet dependencies ───────────────────────────────────
echo ""
echo "=== [4/6] Installing TransUNet dependencies ==="
pip install ml-collections

# ── Step 5: Install shared / pipeline dependencies ───────────────────────────
echo ""
echo "=== [5/6] Installing pipeline dependencies ==="
pip install \
    numpy \
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

# ── Step 6: Clone repos & download weights ───────────────────────────────────
echo ""
echo "=== [6/6] Cloning repos and downloading weights ==="
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
echo " Environment:  $ENV_NAME"
echo " Activate:     conda activate $ENV_NAME"
echo ""
echo " Before running UltraSAM scripts, set PYTHONPATH:"
echo "   export PYTHONPATH=\$PYTHONPATH:$(pwd)/UltraSam"
echo ""
echo " Verify installation:"
echo "   python -c \"import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())\""
echo "   python -c \"import mmengine; print('mmengine', mmengine.__version__)\""
echo "   python -c \"import mmcv; print('mmcv', mmcv.__version__)\""
echo "   python -c \"import mmdet; print('mmdet', mmdet.__version__)\""
echo "============================================================"
