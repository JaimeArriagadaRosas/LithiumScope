from __future__ import annotations

from lithiumscope.resources.detection.device import DeviceInfo
from lithiumscope.resources.manager import cpu_worker_budget, worker_limit


def create_model(device: DeviceInfo, random_seed: int = 42, **params):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost es opcional. Instale con: pip install -e \".[ml]\"") from exc

    defaults = {
        "thread_count": cpu_worker_budget(),
        "iterations": 500,
        "depth": 7,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU" if device.catboost_gpu_available else "CPU",
    }
    defaults.update(params)
    defaults["thread_count"]=worker_limit(defaults["thread_count"])
    return CatBoostRegressor(**defaults)


def optuna_space(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 250, 900),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
    }
