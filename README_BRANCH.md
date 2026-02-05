## Branch: `with-masks`

Detection training using **tight bounding boxes derived from PBM masks**.

### What this branch does

- **Extract**: YCB Berkeley `*.tgz` archives (raw data lives under `data/raw/`)
- **Prepare (detection)**: build a YOLO detection dataset under `data/processed/ybc/`
  - images: `data/processed/ybc/images/{train,val,test}/`
  - labels: `data/processed/ybc/labels/{train,val,test}/`
  - labels are YOLO bboxes computed from the PBM masks (`--label-mode mask-bbox`)
- **Train**: Ultralytics YOLO detection
- **Report snapshot**: copy commit-friendly artifacts into `reports/<run>_<timestamp>/`

### Prerequisites

- Python >= 3.10
- A virtualenv at `.venv/` (recommended)
- Raw dataset present at:
  - `data/raw/ycb/berkeley/<object>/...`

### Run everything (external terminal)

```bash
cd /home/tnlab_sk/Desktop/YBC_YOLO

# activate venv
source .venv/bin/activate

# install package
pip install -e .

# 1) extract raw archives (safe to re-run; skips already-extracted objects)
bash scripts/extract_berkeley.sh

# 2) prepare YOLO detection dataset using PBM masks → tight bboxes
ybc-yolo prepare --label-mode mask-bbox

# 3) train detection
ybc-yolo train --config configs/train/detect.yaml

# 4) snapshot latest run into reports/
ybc-yolo report
```

### Verify labels are mask-derived (not full-image)

```bash
head -n 1 data/processed/ybc/labels/test/*.txt | head -n 5
```

You should see widths/heights less than `1.0` most of the time.

### Outputs

- **Training runs**: `runs/detect/...`
  - `weights/best.pt`, `weights/last.pt`
  - `results.csv` (metrics per epoch)
- **Report snapshots**: `reports/` (commit-friendly)

### Notes

- `data/raw/`, `data/processed/`, `runs/`, and `reports/` are gitignored by default.
- Split lists are tracked in `data/splits/`.

