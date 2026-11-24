from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    """Preserve checkout paths; use a writable home for installed wheels."""
    override = os.environ.get("LITHIUMSCOPE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src" / "lithiumscope").is_dir():
        return checkout
    return (Path.home() / ".lithiumscope").resolve()


PROJECT_ROOT = _project_root()

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

RUNTIME_DIRECTORIES = (
    DATA_DIR / "raw" / "model_1",
    DATA_DIR / "raw" / "model_2",
    DATA_DIR / "interim" / "model_1",
    DATA_DIR / "interim" / "model_2",
    DATA_DIR / "processed" / "model_1",
    DATA_DIR / "processed" / "model_2",
    MODELS_DIR / "model_1" / "trained",
    MODELS_DIR / "model_1" / "metadata",
    MODELS_DIR / "model_2" / "trained",
    MODELS_DIR / "model_2" / "metadata",
    RESULTS_DIR / "model_1" / "metrics",
    RESULTS_DIR / "model_1" / "predictions",
    RESULTS_DIR / "model_1" / "figures",
    RESULTS_DIR / "model_2" / "metrics",
    RESULTS_DIR / "model_2" / "predictions",
    RESULTS_DIR / "model_2" / "figures",
    LOGS_DIR,
)


def ensure_runtime_directories() -> None:
    for directory in RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
