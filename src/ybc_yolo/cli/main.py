"""`ybc-yolo` command entry point."""
from pathlib import Path
import typer

from ybc_yolo.config.loader import load_yaml
from ybc_yolo.data.converters.ybc_native import berkeley_to_yolo
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
    )
    typer.echo(f"Done. YOLO dataset at {out_dir}, split lists in data/splits/.")


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

if __name__ == "__main__":
    app()
