from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _base_model(random_seed: int = 42, **params):
    defaults = {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "probability": False,
        "class_weight": "balanced",
        "random_state": random_seed,
    }
    defaults.update(params)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(**defaults)),
        ]
    )


def create_model(
    random_seed: int = 42,
    *,
    calibrated: bool = True,
    calibration_cv=None,
    **params,
):
    estimator = _base_model(
        random_seed=random_seed,
        **params,
    )
    if not calibrated:
        return estimator

    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=calibration_cv if calibration_cv is not None else 3,
        ensemble=False,
    )


def optuna_space(trial) -> dict:
    return {
        "C": trial.suggest_float(
            "C",
            0.1,
            100.0,
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            1e-4,
            0.1,
            log=True,
        ),
    }
