# YCB Berkeley — YOLO26 Segmentation (Fine-tuned)

Fine-tuned **YOLO26n-seg** model for segmenting YCB objects. Trained on our own annotated data with YOLO26 augmentations, starting from a model pretrained on the YCB Berkeley dataset.

## Model

- **Weights:** `weights/best_finetuned_latest.pt`
- **Architecture:** YOLO26n-seg (Ultralytics)
- **Input size:** 640×640
- **Task:** Instance segmentation

## Classes (15)

`bowl`, `cleanser`, `coffee_can`, `cracker_box`, `gelatin_box`, `jaco`, `meat_can`, `mug`, `mustard_bottle`, `rubik`, `soup_can`, `tennis_ball`, `windex`, `wooden_block`, `workspace`

## Usage

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("weights/best_finetuned_latest.pt")

# Image
results = model.predict("image.jpg")

# Webcam
results = model.predict(source=0, show=True, conf=0.25)
```

## Results

Training and validation figures are in `figures/`:

- `results.png` — training curves
- `confusion_matrix.png` / `confusion_matrix_normalized.png` — classification confusion
- `BoxF1_curve.png`, `BoxPR_curve.png` — box metrics
- `MaskF1_curve.png`, `MaskPR_curve.png` — mask metrics
- `val_batch0_pred.jpg`, `val_batch1_pred.jpg` — sample validation predictions

## License

Weights and figures are shared for research use. Dataset and training code are not included.
