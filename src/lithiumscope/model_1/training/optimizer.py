from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.optimizer")


def _sqlite_storage(path: Path | None) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.resolve().as_posix()


def optimize_with_optuna(
    estimator_factory: Callable[[dict], object],
    parameter_space: Callable,
    x,
    y,
    cv,
    n_trials: int = 20,
    progress_callback: Callable[[int, int], None] | None = None,
    storage_path: Path | None = None,
    study_name: str | None = None,
) -> dict:
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError(
            'Optuna es requerido para la competencia completa. Instale con: '
            'pip install -e ".[ml,imagery,dev]"'
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = parameter_space(trial)
        estimator = estimator_factory(params)
        scores = cross_val_score(
            estimator,
            x,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        return float(-np.mean(scores))

    storage = _sqlite_storage(storage_path)
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
    )
    completed_before = sum(
        trial.state == TrialState.COMPLETE
        for trial in study.trials
    )
    remaining = max(0, n_trials - completed_before)

    if progress_callback is not None:
        progress_callback(completed_before, n_trials)

    callbacks = []
    if progress_callback is not None:
        def on_trial_complete(study, trial):
            completed = sum(
                item.state == TrialState.COMPLETE
                for item in study.trials
            )
            progress_callback(min(completed, n_trials), n_trials)

        callbacks.append(on_trial_complete)

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=callbacks,
            show_progress_bar=False,
        )

    if not study.best_trials:
        raise RuntimeError(
            f"Optuna study {study_name or '<anonymous>'} has no completed trial."
        )

    logger.info(
        "Optuna best RMSE=%.6f params=%s completed=%d/%d storage=%s",
        study.best_value,
        study.best_params,
        sum(trial.state == TrialState.COMPLETE for trial in study.trials),
        n_trials,
        storage_path,
    )
    return dict(study.best_params)
