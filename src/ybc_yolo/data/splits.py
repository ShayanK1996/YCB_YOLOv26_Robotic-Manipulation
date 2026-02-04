"""Deterministic dataset splits."""
from pathlib import Path
from typing import List

from ybc_yolo.utils.io import path_hash


def make_splits(
    paths: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> tuple[List[Path], List[Path], List[Path]]:
    """Split paths deterministically by hash into train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ordered = sorted(paths, key=lambda p: path_hash(p, length=16))
    n = len(ordered)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return ordered[:t], ordered[t : t + v], ordered[t + v :]
