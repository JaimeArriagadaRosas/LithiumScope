from __future__ import annotations

from pathlib import Path

import joblib

from lithiumscope.core.exceptions import ModelNotReadyError
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.persistence.active_models import active_model_path


def latest_model_path(model_group: str, prefix: str | None = None) -> Path:
    active = active_model_path(model_group)
    if active is not None and (
        prefix is None
        or prefix in active.name
        or prefix in active.parent.name
    ):
        return active

    directory = MODELS_DIR / model_group / "trained"
    if not directory.exists():
        raise ModelNotReadyError(f"No trained model directory exists for {model_group}.")

    candidates = list(directory.glob("*.joblib")) + list(directory.glob("*/model.joblib"))
    if prefix:
        candidates = [
            path
            for path in candidates
            if prefix in path.name or prefix in path.parent.name
        ]
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise ModelNotReadyError(f"No trained models found for {model_group}.")
    return candidates[0]


def load_latest_model(model_group: str, prefix: str | None = None):
    path = latest_model_path(model_group, prefix=prefix)
    return joblib.load(path), path
