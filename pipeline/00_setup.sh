#!/usr/bin/env bash
# Clone TransUNet and UltraSAM repos, download pretrained weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=== Cloning TransUNet ==="
if [ ! -d "TransUNet" ]; then
    git clone https://github.com/Beckschen/TransUNet.git TransUNet
else
    echo "TransUNet already exists, skipping clone."
fi

echo "=== Cloning UltraSam ==="
if [ ! -d "UltraSam" ]; then
    git clone https://github.com/CAMMA-public/UltraSam.git UltraSam
else
    echo "UltraSam already exists, skipping clone."
fi

# --- Download TransUNet pretrained encoder (R50+ViT-B_16) ---
TRANSUNET_WEIGHT_DIR="TransUNet/model/vit_checkpoint/imagenet21k"
mkdir -p "$TRANSUNET_WEIGHT_DIR"
TRANSUNET_WEIGHT="$TRANSUNET_WEIGHT_DIR/R50+ViT-B_16.npz"
if [ ! -f "$TRANSUNET_WEIGHT" ]; then
    echo "=== Downloading R50+ViT-B_16.npz ==="
    wget -q --show-progress -O "$TRANSUNET_WEIGHT" \
        "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
else
    echo "TransUNet weights already exist, skipping download."
fi

# --- Download UltraSAM checkpoint ---
ULTRASAM_WEIGHT_DIR="UltraSam/weights"
mkdir -p "$ULTRASAM_WEIGHT_DIR"
ULTRASAM_WEIGHT="$ULTRASAM_WEIGHT_DIR/UltraSam.pth"
if [ ! -f "$ULTRASAM_WEIGHT" ]; then
    echo "=== Downloading UltraSam.pth ==="
    echo "NOTE: UltraSam.pth must be downloaded manually."
    echo "Please download it from the UltraSam repository releases or model zoo"
    echo "and place it at: $ULTRASAM_WEIGHT"
    echo "See: https://github.com/CAMMA-public/UltraSam#pretrained-models"
else
    echo "UltraSam weights already exist, skipping download."
fi

echo ""
echo "=== Setup complete ==="
echo "TransUNet repo:   $ROOT_DIR/TransUNet"
echo "UltraSam repo:    $ROOT_DIR/UltraSam"
echo "TransUNet weight: $TRANSUNET_WEIGHT"
echo "UltraSam weight:  $ULTRASAM_WEIGHT"
