from __future__ import annotations

from lithiumscope.core.device import DeviceInfo, xgboost_device_parameters


def create_model(device: DeviceInfo, random_seed: int = 42, **params):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("XGBoost is optional. Install with: pip install -e '.[ml]'") from exc
    defaults = {
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.9, "colsample_bytree": 0.9, "random_state": random_seed,
        "objective": "reg:squarederror", **xgboost_device_parameters(device),
    }
    defaults.update(params)
    return XGBRegressor(**defaults)


def optuna_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
