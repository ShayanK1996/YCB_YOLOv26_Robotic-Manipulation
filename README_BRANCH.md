## Branch: `with-masks-seg`

Train **YOLO26 segmentation** on the YCB Berkeley subset using the provided **PBM masks**.

This branch produces a single segmentation model (`yolo26*-seg.pt`) that outputs:
- **Detection**: class + confidence + bounding box
- **Segmentation**: instance mask (usable for grasp pose + orientation)

---

### Data layout

**Raw (not committed):**
- `data/raw/ycb/berkeley/<object>/...`
  - images: `N*_*.jpg`
  - masks: `masks/N*_mask.pbm`

**Prepared YOLO-seg dataset (not committed):**
- `data/processed/ybc_seg/`
  - `images/{train,val,test}/`
  - `labels/{train,val,test}/`
    - label format: `class x1 y1 x2 y2 ...` (normalized polygon points)

**Split lists (committed):**
- `data/splits/ycb_berkeley_{train,val,test}.txt`

---

### Environment setup

```bash
cd /home/tnlab_sk/Desktop/YBC_YOLO
source .venv/bin/activate
pip install -e .
```

---

### 1) Extract archives (only if needed)

```bash
bash scripts/extract_berkeley.sh
```

---

### 2) Prepare YOLO-seg labels from PBM masks

This converts PBM masks into simplified external-contour polygons and writes the segmentation dataset:

```bash
ybc-yolo prepare-seg
```

Quick sanity checks:

```bash
ls data/processed/ybc_seg/images/train | head
ls data/processed/ybc_seg/labels/train | head
head -n 1 data/processed/ybc_seg/labels/train/*.txt | head
```

Each label line should contain **many numbers** (polygon points), not just 5 values.

---

### 3) Train YOLO26 segmentation

Config:
- `configs/train/seg.yaml`
- model: `yolo26n-seg.pt` (downloads if missing)

Run:

```bash
ybc-yolo train --config configs/train/seg.yaml
```

Outputs go to `runs/segment/...` and include:
- `weights/best.pt`, `weights/last.pt`
- `results.csv` + plots
- `val_batch*_pred.jpg` and `val_batch*_labels.jpg`

---

### 4) Snapshot a git-friendly report (recommended)

Ultralytics writes a lot into `runs/` (gitignored). Snapshot key artifacts into `reports/`:

```bash
ybc-yolo report --run-dir runs/segment/runs/<your_run_name>
```

Or snapshot the latest run automatically:

```bash
ybc-yolo report
```

This creates `reports/<run>_<timestamp>/` containing `results.csv`, `summary.json`, `metadata.json`, and key figures.

---

### Notes for robotics

For top-down manipulation, use the predicted mask to compute:
- centroid (grasp XY in pixels → then convert to world using calibration)
- orientation angle (PCA on mask pixels or oriented bbox)

