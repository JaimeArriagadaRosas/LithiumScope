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
    device=None,
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
    from lithiumscope.resources.allocation.optimization_parallel import choose_execution,map_calls
    execution=choose_execution(device,pruning_enabled)
    logger.info("Optimization execution: %s",execution)
    contract = study_contract(storage_path, x=x, y=y, groups=groups, splits=splits,
        seed=random_seed, budget=n_trials, pruning=[pruning_enabled,startup_trials,warmup_folds],
        direction="minimize", parameter_space=parameter_space, estimator_factory=estimator_factory,
        model_group="model_1", execution=execution)

    from lithiumscope.core.logging.context import scientific_log_context

    def score_fold(params,step):
        train_idx,valid_idx=splits[step]
        estimator=estimator_factory(params)
        from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
        fit_regression_estimator(estimator,_take_rows(x,train_idx),_take_rows(y,train_idx),
                                 _take_rows(groups,train_idx) if groups is not None else None)
        predictions=estimator.predict(_take_rows(x,valid_idx))
        return float(np.sqrt(mean_squared_error(_take_rows(y,valid_idx),predictions)))

    def score_trial(params):
        return [score_fold(params,step) for step in range(len(splits))]

    def objective(trial):
        with scientific_log_context(trial=trial.number,stage="optuna_inner_cv"):
            params=parameter_space(trial)
            rmses=[]
            parallel=(map_calls(score_fold,[((params,step),trial.number,step+1) for step in range(len(splits))],execution)
                      if execution.mode=="inner_fold" else None)
            for step in range(len(splits)):
                rmse=parallel[step] if parallel is not None else score_fold(params,step)
                record_fold(trial,rmse,step)
                rmses.append(rmse)
                trial.report(float(np.mean(rmses)),step=step)
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
    batch_runner=None
    if execution.mode=="trial_batch":
        from functools import partial
        from lithiumscope.core.scientific.optuna_batch import run_batched_trials
        batch_runner=partial(run_batched_trials,parameter_space=parameter_space,score_trial=score_trial,
                             execution=execution,expected_folds=len(splits))
    report = run_bounded_study(study, objective, contract, storage_path=storage_path,
                               progress_callback=progress_callback,batch_runner=batch_runner)
    logger.info("Optuna audit: %s", report)
    return dict(report["best_params"])
