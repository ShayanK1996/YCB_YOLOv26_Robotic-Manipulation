"""Structured logging utilities (minimal).

Ultralytics already logs rich training artifacts into `runs/`.
We keep this module lightweight to avoid extra dependencies.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict


def get_logger(name: str = "ybc_yolo") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_json(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))
