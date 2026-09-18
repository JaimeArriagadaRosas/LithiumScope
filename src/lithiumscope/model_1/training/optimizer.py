from __future__ import annotations

from collections.abc import Callable
import numpy as np
from sklearn.model_selection import cross_val_score
from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.optimizer")


def optimize_with_optuna(estimator_factory: Callable[[dict], object], parameter_space: Callable, x, y, cv, n_trials: int = 20) -> dict:
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna is not installed; continuing without hyperparameter optimization.")
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = parameter_space(trial)
        estimator = estimator_factory(params)
        scores = cross_val_score(estimator, x, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(-np.mean(scores))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna best RMSE=%.6f params=%s", study.best_value, study.best_params)
    return dict(study.best_params)
