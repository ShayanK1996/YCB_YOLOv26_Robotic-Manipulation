"""Convert YCB Berkeley RGB highres to YOLO format.

Supports two label modes:
- full-image bbox (legacy sanity-check)
- tight bbox derived from provided PBM masks (recommended when masks exist)
"""
from pathlib import Path
import shutil
from typing import Any, List, Tuple, Optional

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


def _mask_path_for_image(img_path: Path) -> Optional[Path]:
    """Map an image path to its PBM mask path if present."""
    # Common structure: <obj>/<obj>/N1_0.jpg with masks in <obj>/<obj>/masks/N1_0_mask.pbm
    cand = img_path.parent / "masks" / f"{img_path.stem}_mask.pbm"
    if cand.exists():
        return cand
    # Alternate: masks could be stored higher up
    cand2 = img_path.parent.parent / "masks" / f"{img_path.stem}_mask.pbm"
    if cand2.exists():
        return cand2
    return None


def _pbm_bbox(mask_path: Path) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return (xmin, ymin, xmax, ymax, width, height) for PBM mask foreground (bit=1).

    Supports PBM P1 (ASCII) and P4 (binary). If no foreground is found, returns None.
    """
    data = mask_path.read_bytes()
    if len(data) < 2:
        return None
    magic = data[:2].decode(errors="ignore")
    if magic not in {"P1", "P4"}:
        return None

    # Tokenize header (skip comments)
    i = 2
    tokens: List[bytes] = []
    while len(tokens) < 2 and i < len(data):
        # skip whitespace
        while i < len(data) and data[i] in b" \t\r\n":
            i += 1
        if i >= len(data):
            break
        if data[i] == ord("#"):
            # comment to end of line
            while i < len(data) and data[i] != ord("\n"):
                i += 1
            continue
        # read token
        j = i
        while j < len(data) and data[j] not in b" \t\r\n":
            j += 1
        tokens.append(data[i:j])
        i = j

    if len(tokens) < 2:
        return None
    w = int(tokens[0])
    h = int(tokens[1])

    xmin, ymin, xmax, ymax = w, h, -1, -1

    if magic == "P1":
        # remaining tokens are ASCII 0/1 values
        # collect bytes from i onward and split
        body = data[i:].split()
        n = min(len(body), w * h)
        for idx in range(n):
            v = body[idx][:1]
            if v == b"1":
                y = idx // w
                x = idx % w
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
    else:
        # P4 binary: each row is padded to full bytes, MSB first
        row_bytes = (w + 7) // 8
        # Skip any whitespace before bitmap payload
        while i < len(data) and data[i] in b" \t\r\n":
            i += 1
        payload = data[i : i + row_bytes * h]

        # Fast path: numpy unpackbits
        try:
            import numpy as np  # type: ignore

            arr = np.frombuffer(payload, dtype=np.uint8)
            if arr.size < row_bytes * h:
                return None
            bits = np.unpackbits(arr).reshape(h, row_bytes * 8)[:, :w]
            ys, xs = np.nonzero(bits)
            if xs.size == 0:
                return None
            xmin = int(xs.min())
            xmax = int(xs.max())
            ymin = int(ys.min())
            ymax = int(ys.max())
        except Exception:
            # Fallback: slow Python scan
            off = 0
            for y in range(h):
                row = payload[off : off + row_bytes]
                off += row_bytes
                if len(row) < row_bytes:
                    break
                for xb in range(row_bytes):
                    b = row[xb]
                    if b == 0:
                        continue
                    for bit in range(8):
                        x = xb * 8 + bit
                        if x >= w:
                            break
                        if (b >> (7 - bit)) & 1:
                            if x < xmin:
                                xmin = x
                            if x > xmax:
                                xmax = x
                            if y < ymin:
                                ymin = y
                            if y > ymax:
                                ymax = y

    if xmax < xmin or ymax < ymin:
        return None
    return xmin, ymin, xmax, ymax, w, h


def _bbox_to_yolo(xmin: int, ymin: int, xmax: int, ymax: int, w: int, h: int) -> Tuple[float, float, float, float]:
    """Convert pixel bbox (inclusive) to YOLO normalized (xc, yc, bw, bh)."""
    bw = (xmax - xmin + 1) / w
    bh = (ymax - ymin + 1) / h
    xc = (xmin + xmax + 1) / 2 / w
    yc = (ymin + ymax + 1) / 2 / h
    return xc, yc, bw, bh


def _pbm_to_mask(mask_path: Path) -> Tuple[Any, int, int]:
    """Load PBM (P1/P4) into a boolean numpy array (h, w) where True=foreground."""
    import numpy as np  # type: ignore

    data = mask_path.read_bytes()
    magic = data[:2].decode(errors="ignore")
    if magic not in {"P1", "P4"}:
        raise ValueError(f"Unsupported PBM magic: {magic}")

    # Tokenize header
    i = 2
    tokens: List[bytes] = []
    while len(tokens) < 2 and i < len(data):
        while i < len(data) and data[i] in b" \t\r\n":
            i += 1
        if i >= len(data):
            break
        if data[i] == ord("#"):
            while i < len(data) and data[i] != ord("\n"):
                i += 1
            continue
        j = i
        while j < len(data) and data[j] not in b" \t\r\n":
            j += 1
        tokens.append(data[i:j])
        i = j

    if len(tokens) < 2:
        raise ValueError("Invalid PBM header")
    w = int(tokens[0])
    h = int(tokens[1])

    if magic == "P1":
        body = data[i:].split()
        n = min(len(body), w * h)
        flat = np.fromiter((1 if body[k][:1] == b"1" else 0 for k in range(n)), dtype=np.uint8, count=n)
        if flat.size < w * h:
            flat = np.pad(flat, (0, w * h - flat.size), constant_values=0)
        mask = flat.reshape(h, w).astype(bool)
        return mask, w, h

    # P4: skip whitespace and unpack bits
    while i < len(data) and data[i] in b" \t\r\n":
        i += 1
    row_bytes = (w + 7) // 8
    payload = data[i : i + row_bytes * h]
    arr = np.frombuffer(payload, dtype=np.uint8)
    bits = np.unpackbits(arr).reshape(h, row_bytes * 8)[:, :w]
    return bits.astype(bool), w, h


def _mask_to_yolo_polygon(mask: Any, w: int, h: int) -> Optional[List[float]]:
    """Convert a binary mask to a single external polygon in YOLO-seg format.

    Returns a flat list [x1, y1, x2, y2, ...] normalized to [0,1], or None if empty.
    """
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # choose largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 3:
        return None

    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.001 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2)
    if pts.shape[0] < 3:
        return None

    xs = (pts[:, 0].astype(np.float32) / float(w)).clip(0.0, 1.0)
    ys = (pts[:, 1].astype(np.float32) / float(h)).clip(0.0, 1.0)
    poly: List[float] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        poly.extend([x, y])
    return poly


def berkeley_to_yolo_seg(
    raw_root: Path,
    out_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> None:
    """Convert Berkeley YCB to YOLO segmentation dataset (polygons from PBM masks)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    pairs = discover_images(raw_root)
    if not pairs:
        raise FileNotFoundError(f"No images found under {raw_root}")

    pairs.sort(key=lambda it: path_hash(it[0], length=16))
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
            rel = img_path.relative_to(raw_root)
            unique_stem = str(rel.parent).replace("/", "_") + "_" + img_path.stem
            dest_img = im_dir / f"{unique_stem}{img_path.suffix}"
            dest_lbl = lb_dir / f"{unique_stem}.txt"
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            mask_path = _mask_path_for_image(img_path)
            if not mask_path or not mask_path.exists():
                continue
            mask, w, h = _pbm_to_mask(mask_path)
            poly = _mask_to_yolo_polygon(mask, w=w, h=h)
            if not poly:
                continue
            coords = " ".join(f"{v:.6f}" for v in poly)
            dest_lbl.write_text(f"{class_id} {coords}\n")


def berkeley_to_yolo(
    raw_root: Path,
    out_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    label_mode: str = "mask-bbox",
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
            # Avoid re-copying large images if they already exist
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            if label_mode == "full":
                xc, yc, bw, bh = 0.5, 0.5, 1.0, 1.0
            else:
                mask_path = _mask_path_for_image(img_path)
                bbox = _pbm_bbox(mask_path) if mask_path else None
                if bbox is None:
                    # fallback: full-image
                    xc, yc, bw, bh = 0.5, 0.5, 1.0, 1.0
                else:
                    xmin, ymin, xmax, ymax, w, h = bbox
                    xc, yc, bw, bh = _bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)

            dest_lbl.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

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
