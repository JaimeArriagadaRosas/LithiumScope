from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from lithiumscope.core.logging.session import get_logger

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
    groups=None,
) -> dict:
    try:
        import optuna
        from optuna.samplers import TPESampler
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

    from lithiumscope.core.scientific.optuna_contract import (
        study_contract, run_bounded_study, record_fold, validate_splits,
    )
    splits = validate_splits(cv.split(x, y, groups) if hasattr(cv, "split") else cv, x, y, groups)
    contract = study_contract(storage_path, x=x, y=y, groups=groups, splits=splits,
        seed=random_seed, budget=n_trials, pruning=[pruning_enabled,startup_trials,warmup_folds],
        direction="minimize", parameter_space=parameter_space, estimator_factory=estimator_factory,
        model_group="model_1")

    def objective(trial):
        params = parameter_space(trial)
        rmses: list[float] = []
        for step, (train_idx, valid_idx) in enumerate(splits):
            estimator = estimator_factory(params)
            from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
            fit_regression_estimator(estimator, _take_rows(x, train_idx), _take_rows(y, train_idx),
                                     _take_rows(groups,train_idx) if groups is not None else None)
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
            record_fold(trial, rmse, step)
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
    report = run_bounded_study(study, objective, contract, storage_path=storage_path,
                               progress_callback=progress_callback)
    logger.info("Optuna audit: %s", report)
    return dict(report["best_params"])
