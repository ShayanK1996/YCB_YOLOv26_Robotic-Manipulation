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

50 epochs completed in 1.338 hours.
Optimizer stripped from /home/tnlab_sk/Desktop/YBC_YOLO/runs/detect/runs/ycb_berkeley_detect2/weights/last.pt, 6.2MB
Optimizer stripped from /home/tnlab_sk/Desktop/YBC_YOLO/runs/detect/runs/ycb_berkeley_detect2/weights/best.pt, 6.2MB

Validating /home/tnlab_sk/Desktop/YBC_YOLO/runs/detect/runs/ycb_berkeley_detect2/weights/best.pt...
Ultralytics 8.4.11 🚀 Python-3.13.11 torch-2.10.0+cu128 CUDA:0 (NVIDIA GeForce RTX 3070, 7842MiB)
Model summary (fused): 73 layers, 3,008,963 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 64/64 4.1it/s 15.4s
                   all       2040       2040      0.999          1      0.995      0.994
             chips_can        127        127      0.999          1      0.995      0.995
       master_chef_can        127        127      0.999          1      0.995      0.995
           cracker_box        131        131      0.999          1      0.995      0.995
             sugar_box        111        111      0.999          1      0.995      0.995
       tomato_soup_can        119        119      0.999          1      0.995      0.995
        mustard_bottle        112        112          1          1      0.995      0.995
           gelatin_box        134        134          1          1      0.995      0.993
       potted_meat_can        108        108      0.999          1      0.995      0.994
       bleach_cleanser        130        130          1          1      0.995      0.995
         windex_bottle        125        125      0.999          1      0.995      0.995
                  bowl        112        112      0.999          1      0.995      0.995
                   mug        125        125      0.999          1      0.995      0.995
                sponge        117        117      0.999          1      0.995      0.993
            wood_block        112        112          1          1      0.995      0.995
          medium_clamp        115        115      0.999          1      0.995      0.989
           tennis_ball        116        116          1          1      0.995      0.995
           rubiks_cube        119        119      0.999          1      0.995      0.995
Speed: 0.3ms preprocess, 0.9ms inference, 0.0ms loss, 1.9ms postprocess per image
Results saved to /home/tnlab_sk/Desktop/YBC_YOLO/runs/detect/runs/ycb_berkeley_detect2

