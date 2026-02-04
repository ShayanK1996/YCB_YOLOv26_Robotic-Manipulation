"""Wrappers around Ultralytics train/val/predict/export."""
from ultralytics import YOLO
from ybc_yolo.yolo.utils import set_seed

def run_train(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 42)))

    task = cfg["task"]  # "detect" or "segment"
    model_path = cfg["model"]
    data_yaml = cfg["data_yaml"]

    model = YOLO(model_path)

    # Ultralytics automatically infers task from model, but we keep cfg explicit for clarity.
    model.train(
        data=data_yaml,
        imgsz=int(cfg.get("imgsz", 640)),
        epochs=int(cfg.get("epochs", 100)),
        batch=int(cfg.get("batch", 16)),
        device=cfg.get("device", "0"),
        project=cfg.get("project", "runs"),
        name=cfg.get("name", f"ybc_{task}"),
    )

def run_val(weights: str, data: str, task: str, device: str = "0"):
    model = YOLO(weights)
    return model.val(data=data, device=device)

def run_predict(weights: str, source: str, task: str, device: str = "0", save: bool = True):
    model = YOLO(weights)
    return model.predict(source=source, device=device, save=save)

def run_export(weights: str, task: str, fmt: str = "onnx"):
    model = YOLO(weights)
    return model.export(format=fmt)
