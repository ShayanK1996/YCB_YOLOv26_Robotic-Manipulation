#!/bin/bash
# Fine-tune existing model on new Roboflow dataset

set -e  # Exit on error

echo "=========================================="
echo "Fine-tuning YCB YOLO on New Dataset"
echo "=========================================="

# Configuration
PRETRAINED_WEIGHTS="runs/segment/runs/ycb_berkeley_yolo26_seg2/weights/best.pt"
DATA_YAML="data/New_dataset/yolo26_ycb.v2i.yolo26/data.yaml"
PROJECT="runs/segment"
NAME="finetune_new_dataset"

# Check if pretrained weights exist
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "Error: Pretrained weights not found at $PRETRAINED_WEIGHTS"
    exit 1
fi

# Check if data config exists
if [ ! -f "$DATA_YAML" ]; then
    echo "Error: Data config not found at $DATA_YAML"
    exit 1
fi

echo "Pretrained weights: $PRETRAINED_WEIGHTS"
echo "Dataset config: $DATA_YAML"
echo "Output: $PROJECT/$NAME"
echo "=========================================="

# Run training using YOLO CLI
yolo segment train \
    model="$PRETRAINED_WEIGHTS" \
    data="$DATA_YAML" \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    device=0 \
    workers=8 \
    patience=20 \
    project="$PROJECT" \
    name="$NAME" \
    lr0=0.001 \
    lrf=0.01 \
    optimizer=auto \
    amp=True \
    plots=True \
    val=True

echo "=========================================="
echo "Fine-tuning complete!"
echo "Results saved to: $PROJECT/$NAME"
echo "=========================================="
