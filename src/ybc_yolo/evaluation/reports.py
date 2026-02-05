"""Save metrics to JSON/CSV and snapshot Ultralytics run artifacts.

Goal: keep lightweight, commit-friendly run records (metrics + plots + metadata)
without committing the full `runs/` directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunMetadata:
    created_at_utc: str
    run_dir: str
    report_dir: str
    git_branch: Optional[str]
    git_commit: Optional[str]
    python: str
    platform: str
    torch: Optional[str]
    cuda: Optional[str]
    ultralytics: Optional[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(cmd: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _versions() -> Dict[str, Optional[str]]:
    torch_v = None
    cuda_v = None
    ul_v = None
    try:
        import torch  # type: ignore

        torch_v = getattr(torch, "__version__", None)
        cuda_v = getattr(torch.version, "cuda", None) if hasattr(torch, "version") else None
    except Exception:
        pass

    try:
        import ultralytics  # type: ignore

        ul_v = getattr(ultralytics, "__version__", None)
    except Exception:
        pass

    return {"torch": torch_v, "cuda": cuda_v, "ultralytics": ul_v}


def find_latest_run_dir(runs_root: Path) -> Path:
    """Find the most recently modified directory under runs_root that contains results.csv."""
    runs_root = Path(runs_root)
    candidates: List[Path] = []
    for p in runs_root.rglob("results.csv"):
        candidates.append(p.parent)
    if not candidates:
        raise FileNotFoundError(f"No results.csv found under {runs_root}")
    return max(candidates, key=lambda d: (d / "results.csv").stat().st_mtime)


def read_results_csv(results_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Coerce numeric values when possible
            rr: Dict[str, Any] = {}
            for k, v in r.items():
                if v is None:
                    rr[k] = None
                    continue
                try:
                    rr[k] = float(v)
                except ValueError:
                    rr[k] = v
            rows.append(rr)
    return rows


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"epochs_logged": 0}

    # Use mAP50-95 when available as the primary score
    key = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in rows[0] else None
    best = None
    if key:
        best = max(rows, key=lambda r: float(r.get(key, float("-inf"))))

    return {
        "epochs_logged": len(rows),
        "last_epoch": rows[-1].get("epoch"),
        "best_epoch": best.get("epoch") if best else None,
        "best_score_key": key,
        "best_score": best.get(key) if best and key else None,
        "last": rows[-1],
    }


def snapshot_ultralytics_run(
    run_dir: Path,
    report_root: Path = Path("reports"),
    project_root: Optional[Path] = None,
) -> Path:
    """Copy key run artifacts + metadata into a tracked `reports/` subfolder.

    Returns the created report directory.
    """
    run_dir = Path(run_dir)
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    run_name = run_dir.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    report_dir = Path(report_root) / f"{run_name}_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)

    # Copy small, useful artifacts if present
    for fname in [
        "args.yaml",
        "results.csv",
        "results.png",
        "labels.jpg",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
    ]:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, report_dir / fname)

    # Also snapshot the dataset + train config if they exist
    for cfg in [
        project_root / "configs" / "datasets" / "ybc_detect.yaml",
        project_root / "configs" / "train" / "detect.yaml",
    ]:
        if cfg.exists():
            shutil.copy2(cfg, report_dir / cfg.name)

    # Metadata
    versions = _versions()
    meta = RunMetadata(
        created_at_utc=_utc_now_iso(),
        run_dir=str(run_dir.resolve()),
        report_dir=str(report_dir.resolve()),
        git_branch=_run(["git", "branch", "--show-current"], cwd=project_root),
        git_commit=_run(["git", "rev-parse", "HEAD"], cwd=project_root),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        torch=versions["torch"],
        cuda=versions["cuda"],
        ultralytics=versions["ultralytics"],
    )
    (report_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2) + "\n")

    # Metrics summary
    rows = read_results_csv(results_csv)
    summary = summarize_results(rows)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Convenience pointer for humans
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                f"## Run snapshot: `{run_name}`",
                "",
                f"- **Run dir**: `{run_dir}`",
                f"- **Created (UTC)**: `{meta.created_at_utc}`",
                "",
                "### Included",
                "- `results.csv` + `summary.json`",
                "- Common Ultralytics figures (if generated)",
                "- `args.yaml`",
                "- Copied configs (`ybc_detect.yaml`, `detect.yaml`) when present",
                "",
            ]
        )
        + "\n"
    )

    return report_dir
