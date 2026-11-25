from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from lithiumscope.model_2.training.preprocessing.contracts import scaled_estimator
from sklearn.svm import SVC


def _base_model(random_seed: int = 42, **params):
    defaults = {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "random_state": random_seed,
    }
    from lithiumscope.resources.manager import svm_cache_limit
    defaults.update(params)
    defaults['cache_size']=svm_cache_limit(defaults.get('cache_size',200.))
    return scaled_estimator(SVC(**defaults))


def create_model(random_seed: int = 42, *, calibrated: bool = True, calibration_cv=None, **params):
    estimator = _base_model(random_seed=random_seed, **params)
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
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 0.1, log=True),
    }
