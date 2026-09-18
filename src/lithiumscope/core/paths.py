from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    LOGS_DIR / "training",
    LOGS_DIR / "prediction",
    LOGS_DIR / "errors",
)


def ensure_runtime_directories() -> None:
    for directory in RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
