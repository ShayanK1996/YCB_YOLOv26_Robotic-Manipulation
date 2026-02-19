# Fine-tuning Guide: New Roboflow Dataset

This guide explains how to fine-tune your existing YCB Berkeley YOLO model on the new Roboflow dataset.

## Overview

- **Existing Model**: `runs/segment/runs/ycb_berkeley_yolo26_seg2/weights/best.pt`
  - Trained on 17 YCB Berkeley classes
- **New Dataset**: `data/New_dataset/yolo26_ycb.v2i.yolo26/`
  - 13 classes from Roboflow
  - 984 training images, 30 validation images, 15 test images

## Class Comparison

### Original Model (17 classes):
chips_can, master_chef_can, cracker_box, sugar_box, tomato_soup_can, mustard_bottle, gelatin_box, potted_meat_can, bleach_cleanser, windex_bottle, bowl, mug, sponge, wood_block, medium_clamp, tennis_ball, rubiks_cube

### New Dataset (13 classes):
bowl, cleanser, coffee_can, jaco, meat_can, mug, mustard_bottle, rubik, soup_can, tennis_ball, windex, wooden_block, workspace

**Overlapping classes**: bowl, mug, mustard_bottle, tennis_ball, windex (cleanser), rubik (rubiks_cube), wooden_block (wood_block), meat_can (potted_meat_can)

## How to Fine-tune

### Option 1: Using Python Script (Recommended)

```bash
python scripts/finetune_new_dataset.py
```

### Option 2: Using Shell Script

```bash
bash scripts/finetune_new_dataset.sh
```

### Option 3: Using YOLO CLI Directly

```bash
yolo segment train \
    model=runs/segment/runs/ycb_berkeley_yolo26_seg2/weights/best.pt \
    data=data/New_dataset/yolo26_ycb.v2i.yolo26/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    lr0=0.001
```

## Fine-tuning Settings Explained

### Learning Rate
- **lr0=0.001** (vs 0.01 for training from scratch)
  - Lower learning rate preserves learned features
  - Prevents catastrophic forgetting

### Epochs
- **50 epochs** with early stopping (patience=20)
  - Fine-tuning typically requires fewer epochs
  - Early stopping prevents overfitting

### Batch Size
- **batch=8** (adjust based on GPU memory)
  - Increase if you have more GPU memory
  - Decrease if you get CUDA out-of-memory errors

### Optional: Freeze Backbone Layers
Uncomment `freeze=10` in the Python script to freeze the first 10 layers. This:
- Speeds up training
- Preserves low-level features
- Good when new dataset is similar to original

## Expected Output

The fine-tuned model will be saved to:
```
runs/segment/finetune_new_dataset/
├── weights/
│   ├── best.pt      # Best checkpoint (use this for inference)
│   └── last.pt      # Last epoch checkpoint
├── results.csv      # Training metrics
├── results.png      # Training curves
└── confusion_matrix.png
```

## After Fine-tuning

Test your fine-tuned model on livestream:

```python
from ultralytics import YOLO

# Load fine-tuned model
model = YOLO('runs/segment/finetune_new_dataset/weights/best.pt')

# Test on webcam
results = model.predict(source=0, show=True, conf=0.25)
```

## Monitoring Training

Watch training progress in real-time:

```bash
# In another terminal
tail -f runs/segment/finetune_new_dataset/results.csv
```

Or use TensorBoard (if installed):

```bash
tensorboard --logdir runs/segment/finetune_new_dataset
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `batch=4` or `batch=2`
- Reduce image size: `imgsz=416`

### Poor Performance
- Increase epochs: `epochs=100`
- Adjust learning rate: `lr0=0.0001` (lower) or `lr0=0.005` (higher)
- Enable layer freezing: `freeze=10`

### Overfitting (val loss increases)
- Increase augmentation
- Add dropout: `dropout=0.1`
- Reduce epochs
- Early stopping will handle this automatically

## Next Steps

1. Run fine-tuning script
2. Monitor training progress
3. Evaluate on validation set
4. Test on livestream camera
5. Compare with original model performance

Good luck! 🚀
