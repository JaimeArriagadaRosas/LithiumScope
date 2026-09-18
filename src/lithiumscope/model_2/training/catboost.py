from __future__ import annotations

from lithiumscope.core.device import DeviceInfo


def create_model(device: DeviceInfo, random_seed: int = 42, **params):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost es opcional. Instale con: pip install -e \".[ml]\"") from exc
    defaults = {
        "iterations": 500,
        "depth": 7,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU" if device.accelerator == "cuda" else "CPU",
    }
    defaults.update(params)
    return CatBoostClassifier(**defaults)
