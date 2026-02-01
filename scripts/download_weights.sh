#!/bin/bash
# Download pretrained weights for UltraRefiner pipeline.
# Usage: bash scripts/download_weights.sh

set -e
WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

echo "=== UltraRefiner Weight Download ==="

# 1. TransUNet pretrained ViT (R50+ViT-B_16)
VIT_URL="https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
VIT_FILE="$WEIGHTS_DIR/R50+ViT-B_16.npz"
if [ ! -f "$VIT_FILE" ]; then
    echo "Downloading R50+ViT-B_16 pretrained weights..."
    wget -q --show-progress -O "$VIT_FILE" "$VIT_URL"
    echo "  Saved to $VIT_FILE"
else
    echo "R50+ViT-B_16 already exists at $VIT_FILE"
fi

# 2. UltraSAM pretrained weights
ULTRASAM_FILE="$WEIGHTS_DIR/UltraSam.pth"
if [ ! -f "$ULTRASAM_FILE" ]; then
    echo ""
    echo "UltraSAM weights not found at $ULTRASAM_FILE"
    echo "Please download UltraSam.pth from: https://github.com/CAMMA-public/UltraSam"
    echo "Then place it at: $ULTRASAM_FILE"
    echo ""
    echo "After downloading, convert to standalone format:"
    echo "  python scripts/convert_ultrasam_weights.py \\"
    echo "    --input $ULTRASAM_FILE \\"
    echo "    --output $WEIGHTS_DIR/ultrasam_standalone.pth \\"
    echo "    --verify"
else
    echo "UltraSAM checkpoint found at $ULTRASAM_FILE"
    STANDALONE_FILE="$WEIGHTS_DIR/ultrasam_standalone.pth"
    if [ ! -f "$STANDALONE_FILE" ]; then
        echo "Converting to standalone format..."
        python scripts/convert_ultrasam_weights.py \
            --input "$ULTRASAM_FILE" \
            --output "$STANDALONE_FILE" \
            --verify
    else
        echo "Standalone weights already exist at $STANDALONE_FILE"
    fi
fi

echo ""
echo "=== Done ==="
echo "Weight files:"
ls -lh "$WEIGHTS_DIR/"
