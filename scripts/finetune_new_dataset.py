#!/usr/bin/env python3
"""
Fine-tune the existing YCB Berkeley YOLO model on the new Roboflow dataset.

This script loads the pretrained weights from your previous training run and
continues training on the new dataset with adjusted classes.
"""

from ultralytics import YOLO
import torch

def main():
    # Path to your trained model weights
    pretrained_weights = "runs/segment/runs/ycb_berkeley_yolo26_seg2/weights/best.pt"
    
    # Path to your new dataset configuration (v3 - no Roboflow augmentation, auto-orient only)
    data_yaml = "data/New_dataset/yolo26_ycb.v3i.yolo26/data.yaml"
    
    print("=" * 60)
    print("Fine-tuning YCB Berkeley YOLO Model on New Dataset")
    print("=" * 60)
    print(f"Pretrained weights: {pretrained_weights}")
    print(f"New dataset config: {data_yaml}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    
    # Load the model with pretrained weights
    model = YOLO(pretrained_weights)
    
    # Fine-tune on new dataset
    results = model.train(
        data=data_yaml,
        epochs=50,              # Adjust based on your needs
        imgsz=640,              # Image size
        batch=8,                # Batch size (adjust based on GPU memory)
        device=0,               # Use GPU 0
        workers=8,              # Number of workers
        patience=20,            # Early stopping patience
        save=True,              # Save checkpoints
        project="runs/segment", # Save location
        name="finetune_v3_yolo_aug",  # Experiment name (v3 = no Roboflow aug)
        exist_ok=False,         # Don't overwrite existing runs
        
        # Learning rate settings (lower for fine-tuning)
        lr0=0.001,              # Initial learning rate (lower than training from scratch)
        lrf=0.01,               # Final learning rate fraction
        
        # Optimizer
        optimizer="auto",       # AdamW or SGD
        
        # Regularization
        weight_decay=0.0005,
        dropout=0.0,
        
        # Augmentation: use YOLO26 defaults (no overrides) - handles mosaic, mixup,
        # hsv, degrees, translate, scale, fliplr, erasing, etc. from built-in config
        
        # Validation
        val=True,
        plots=True,             # Generate training plots
        
        # Mixed precision training
        amp=True,
        
        # Freeze layers (optional - uncomment to freeze backbone)
        # freeze=10,            # Freeze first 10 layers for faster fine-tuning
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    print("=" * 60)
    
    # Validate the model
    print("\nValidating the fine-tuned model...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, 'seg'):
        print(f"  Mask mAP50: {metrics.seg.map50:.4f}")
        print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")

if __name__ == "__main__":
    main()
