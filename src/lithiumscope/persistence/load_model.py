from __future__ import annotations

from pathlib import Path

import joblib

from lithiumscope.core.exceptions import ModelNotReadyError
from lithiumscope.core.paths import MODELS_DIR


def latest_model_path(model_group: str, prefix: str | None = None) -> Path:
    directory = MODELS_DIR / model_group / "trained"
    if not directory.exists():
        raise ModelNotReadyError(f"No trained model directory exists for {model_group}.")
    pattern = f"{prefix}_*.joblib" if prefix else "*.joblib"
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise ModelNotReadyError(f"No trained models found for {model_group}.")
    return candidates[0]


def load_latest_model(model_group: str, prefix: str | None = None):
    path = latest_model_path(model_group, prefix=prefix)
    return joblib.load(path), path
