"""Convert YCB Berkeley RGB highres to YOLO format (object-crop = full-image bbox)."""
from pathlib import Path
import shutil
from typing import List, Tuple

from ybc_yolo.utils.io import safe_mkdir, path_hash

# Object dirs in sorted order → class index 0..16
YCB_BERKELEY_OBJECTS = [
    "001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "009_gelatin_box",
    "010_potted_meat_can",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "036_wood_block",
    "050_medium_clamp",
    "056_tennis_ball",
    "077_rubiks_cube",
]


def discover_images(raw_root: Path) -> List[Tuple[Path, int]]:
    """Find all .jpg under raw_root and return (path, class_id) list."""
    raw_root = Path(raw_root)
    obj_to_id = {name: i for i, name in enumerate(YCB_BERKELEY_OBJECTS)}
    out: List[Tuple[Path, int]] = []
    for obj_name in YCB_BERKELEY_OBJECTS:
        obj_dir = raw_root / obj_name
        if not obj_dir.is_dir():
            continue
        class_id = obj_to_id[obj_name]
        # Structure: obj_dir / "<obj>_berkeley_rgb_highres" / "<obj>" / *.jpg
        for sub in obj_dir.iterdir():
            if not sub.is_dir():
                continue
            inner = sub / obj_name if (sub / obj_name).is_dir() else sub
            for jpg in inner.rglob("*.jpg"):
                out.append((jpg, class_id))
    return out


def berkeley_to_yolo(
    raw_root: Path,
    out_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Convert Berkeley YCB to YOLO dirs (images + labels) with deterministic train/val/test split."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    pairs = discover_images(raw_root)
    if not pairs:
        raise FileNotFoundError(f"No images found under {raw_root}")

    # Deterministic split by path hash
    def split_key(item: Tuple[Path, int]) -> int:
        return path_hash(item[0], length=16)

    pairs.sort(key=split_key)
    n = len(pairs)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    train_list = pairs[:t]
    val_list = pairs[t : t + v]
    test_list = pairs[t + v :]

    for split_name, part in [("train", train_list), ("val", val_list), ("test", test_list)]:
        im_dir = safe_mkdir(out_root / "images" / split_name)
        lb_dir = safe_mkdir(out_root / "labels" / split_name)
        for img_path, class_id in part:
            stem = img_path.stem
            # Ensure unique name across objects (same stem in different folders)
            rel = img_path.relative_to(raw_root)
            unique_stem = str(rel.parent).replace("/", "_") + "_" + stem
            dest_img = im_dir / f"{unique_stem}{img_path.suffix}"
            dest_lbl = lb_dir / f"{unique_stem}.txt"
            shutil.copy2(img_path, dest_img)
            # YOLO: class_id x_center y_center width height (normalized). Full image = 0.5 0.5 1 1
            dest_lbl.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")

    # Write split lists (paths relative to raw_root) for tracking in data/splits
    # Assume out_root is like .../data/processed/ybc → splits = .../data/splits
    data_dir = out_root.parent  # processed
    if data_dir.name == "processed":
        splits_dir = data_dir.parent / "splits"
    else:
        splits_dir = out_root / "splits"
    safe_mkdir(splits_dir)
    for split_name, part in [("train", train_list), ("val", val_list), ("test", test_list)]:
        lines = [str(p.relative_to(raw_root)) for p, _ in part]
        (splits_dir / f"ycb_berkeley_{split_name}.txt").write_text("\n".join(lines) + "\n")
