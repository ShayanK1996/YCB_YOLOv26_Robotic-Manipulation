"""`ybc-yolo` command entry point."""
from pathlib import Path
import typer

from ybc_yolo.config.loader import load_yaml
from ybc_yolo.yolo.trainer import run_train, run_val, run_predict, run_export

app = typer.Typer(no_args_is_help=True)

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
