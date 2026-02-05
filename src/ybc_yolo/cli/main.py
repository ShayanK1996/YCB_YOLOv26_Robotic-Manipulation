"""`ybc-yolo` command entry point."""
from pathlib import Path
import typer

from ybc_yolo.config.loader import load_yaml
from ybc_yolo.data.converters.ybc_native import berkeley_to_yolo, berkeley_to_yolo_seg
from ybc_yolo.evaluation.reports import find_latest_run_dir, snapshot_ultralytics_run
from ybc_yolo.yolo.trainer import run_train, run_val, run_predict, run_export

app = typer.Typer(no_args_is_help=True)


@app.command()
def prepare(
    raw_dir: Path = typer.Option(
        "data/raw/ycb/berkeley",
        help="Raw YCB Berkeley root (object folders with .tgz or extracted images)",
    ),
    out_dir: Path = typer.Option(
        "data/processed/ybc",
        help="Output YOLO dataset root (images/train|val|test, labels/...)",
    ),
    train_ratio: float = typer.Option(0.7, help="Train fraction"),
    val_ratio: float = typer.Option(0.2, help="Val fraction"),
    test_ratio: float = typer.Option(0.1, help="Test fraction"),
    label_mode: str = typer.Option(
        "mask-bbox",
        help="Label mode: 'mask-bbox' (from PBM masks) or 'full' (full-image bbox)",
    ),
):
    """Extract (run scripts/extract_berkeley.sh first if needed), convert to YOLO, write splits."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw dir not found: {raw_dir}")
    berkeley_to_yolo(
        raw_root=raw_dir,
        out_root=out_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        label_mode=label_mode,
    )
    typer.echo(f"Done. YOLO dataset at {out_dir}, split lists in data/splits/.")


@app.command()
def prepare_seg(
    raw_dir: Path = typer.Option(
        "data/raw/ycb/berkeley",
        help="Raw YCB Berkeley root (object folders with extracted images + masks)",
    ),
    out_dir: Path = typer.Option(
        "data/processed/ybc_seg",
        help="Output YOLO-seg dataset root (images/train|val|test, labels/...)",
    ),
    train_ratio: float = typer.Option(0.7, help="Train fraction"),
    val_ratio: float = typer.Option(0.2, help="Val fraction"),
    test_ratio: float = typer.Option(0.1, help="Test fraction"),
):
    """Convert PBM masks into YOLO-seg polygon labels and write dataset."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw dir not found: {raw_dir}")
    berkeley_to_yolo_seg(
        raw_root=raw_dir,
        out_root=out_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    typer.echo(f"Done. YOLO-seg dataset at {out_dir}.")


@app.command()
def train(config: Path = typer.Option(..., exists=True, help="Training YAML config path")):
    cfg = load_yaml(config)
    run_train(cfg)

@app.command()
def eval(weights: Path, data: Path, task: str = "detect", device: str = "0"):
    run_val(weights=str(weights), data=str(data), task=task, device=device)

@app.command()
def predict(weights: Path, source: Path, task: str = "detect", device: str = "0", save: bool = True):
    run_predict(weights=str(weights), source=str(source), task=task, device=device, save=save)

@app.command()
def export(weights: Path, task: str = "detect", fmt: str = "onnx"):
    run_export(weights=str(weights), task=task, fmt=fmt)


@app.command()
def report(
    run_dir: Path = typer.Option(
        None,
        help="Ultralytics run directory (defaults to latest under runs/)",
    ),
    runs_root: Path = typer.Option("runs", help="Where Ultralytics writes runs/"),
    report_root: Path = typer.Option("reports", help="Tracked report output folder"),
):
    """Snapshot a run into `reports/` for long-term tracking."""
    if run_dir is None:
        run_dir = find_latest_run_dir(Path(runs_root))
    out = snapshot_ultralytics_run(Path(run_dir), report_root=Path(report_root), project_root=Path.cwd())
    typer.echo(f"Wrote report snapshot to {out}")

if __name__ == "__main__":
    app()
