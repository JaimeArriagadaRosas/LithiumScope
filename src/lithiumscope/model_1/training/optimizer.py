from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.optimizer")


def _sqlite_storage(path: Path | None) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.resolve().as_posix()


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


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
    *,
    random_seed: int = 42,
    pruning_enabled: bool = True,
    startup_trials: int = 4,
    warmup_folds: int = 2,
) -> dict:
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError(
            "Optuna es requerido para la competencia completa. "
            'Instale con: pip install -e ".[ml,imagery,dev]"'
        ) from exc

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=max(
                1,
                startup_trials,
            ),
            n_warmup_steps=max(
                0,
                warmup_folds - 1,
            ),
        )
        if pruning_enabled
        else optuna.pruners.NopPruner()
    )

    def objective(trial):
        params = parameter_space(trial)
        rmses: list[float] = []
        split_iterator = (
            cv.split(x, y)
            if hasattr(cv, "split")
            else iter(cv)
        )
        for step, (train_idx, valid_idx) in enumerate(
            split_iterator
        ):
            estimator = estimator_factory(params)
            estimator.fit(
                _take_rows(x, train_idx),
                _take_rows(y, train_idx),
            )
            predictions = estimator.predict(
                _take_rows(x, valid_idx)
            )
            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        _take_rows(y, valid_idx),
                        predictions,
                    )
                )
            )
            rmses.append(rmse)
            running = float(np.mean(rmses))
            trial.report(running, step=step)
            if pruning_enabled and trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(rmses))

    storage = _sqlite_storage(storage_path)
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(
            storage and study_name
        ),
        sampler=TPESampler(seed=random_seed),
        pruner=pruner,
    )
    terminal = {
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.FAIL,
    }
    finished_before = sum(
        trial.state in terminal
        for trial in study.trials
    )
    remaining = max(
        0,
        n_trials - finished_before,
    )

    if progress_callback is not None:
        progress_callback(
            min(finished_before, n_trials),
            n_trials,
        )

    callbacks = []
    if progress_callback is not None:
        def on_trial_complete(study, trial):
            finished = sum(
                item.state in terminal
                for item in study.trials
            )
            progress_callback(
                min(finished, n_trials),
                n_trials,
            )

        callbacks.append(on_trial_complete)

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=callbacks,
            show_progress_bar=False,
        )

    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError(
            f"Optuna study {study_name or '<anonymous>'} "
            "has no completed trial."
        )

    logger.info(
        "Optuna best RMSE=%.6f params=%s complete=%d "
        "pruned=%d target_trials=%d storage=%s",
        study.best_value,
        study.best_params,
        len(completed),
        sum(
            trial.state == TrialState.PRUNED
            for trial in study.trials
        ),
        n_trials,
        storage_path,
    )
    return dict(study.best_params)
