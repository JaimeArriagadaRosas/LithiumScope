from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lithiumscope.core.logging.session import get_logger

logger = get_logger("model_2.optimizer")


def _take_rows(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _score_vector(estimator, x) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x)
        return np.asarray(probabilities[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(
            estimator.decision_function(x),
            dtype=float,
        ).reshape(-1)
    return np.asarray(estimator.predict(x), dtype=float).reshape(-1)


def _sqlite_storage(path: Path | None) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.resolve().as_posix()


def optimize_classifier_with_optuna(
    estimator_factory: Callable[[dict], object],
    parameter_space: Callable,
    x,
    y,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_trials: int,
    random_seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
    storage_path: Path | None = None,
    study_name: str | None = None,
    pruning_enabled: bool = True,
    startup_trials: int = 3,
    warmup_folds: int = 2,
    lithium_values=None,
    label_quantile: float = 0.75,
    fixed_threshold_ppm: float | None = None,
    groups=None,
) -> dict:
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise RuntimeError(
            'Optuna es requerido para Modelo 2. Instale: '
            'pip install -e ".[ml,imagery,dev]"'
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=max(1, startup_trials),
            n_warmup_steps=max(0, warmup_folds - 1),
        )
        if pruning_enabled
        else optuna.pruners.NopPruner()
    )

    from lithiumscope.core.scientific.optuna_contract import (
        study_contract, run_bounded_study, record_fold, validate_splits,
    )
    splits = validate_splits(splits, x, y, groups)
    contract = study_contract(storage_path, x=x, y=y, groups=groups, splits=splits,
        seed=random_seed, budget=n_trials, pruning=[pruning_enabled,startup_trials,warmup_folds],
        direction="maximize", parameter_space=parameter_space, estimator_factory=estimator_factory,
        model_group="model_2", extra={"lithium_values": None if lithium_values is None else np.asarray(lithium_values).tolist(),
                                     "label_quantile":label_quantile,"fixed_threshold_ppm":fixed_threshold_ppm})
    prepared_splits = []
    if lithium_values is not None:
        inner_lithium = np.asarray(lithium_values, dtype=float)
        if inner_lithium.shape != (len(y),) or not np.isfinite(inner_lithium).all() or (inner_lithium < 0).any():
            raise ValueError("Invalid inner Li labels")
    for train_idx, valid_idx in splits:
        if lithium_values is not None:
            label_limit = (float(fixed_threshold_ppm) if fixed_threshold_ppm is not None
                           else float(np.quantile(inner_lithium[train_idx], label_quantile)))
            training_y = (inner_lithium[train_idx] >= label_limit).astype(int)
            validation_y = (inner_lithium[valid_idx] >= label_limit).astype(int)
        else:
            training_y = np.asarray(_take_rows(y, train_idx))
            validation_y = np.asarray(_take_rows(y, valid_idx))
        if set(np.unique(training_y)) != {0, 1} or set(np.unique(validation_y)) != {0, 1}:
            raise ValueError("Un fold interno no contiene dos clases; no se omitira el fold.")
        prepared_splits.append((train_idx, valid_idx, training_y, validation_y))

    from lithiumscope.core.logging.context import scientific_log_context

    def objective(trial):
        with scientific_log_context(trial=trial.number,stage="optuna_inner_cv"):
            params = parameter_space(trial)
            scores: list[float] = []

            for step, (train_idx, valid_idx, training_y, validation_y) in enumerate(prepared_splits):
                estimator = estimator_factory(params)
                from lithiumscope.resources.manager import estimator_thread_scope
                with estimator_thread_scope():
                    estimator.fit(_take_rows(x, train_idx), training_y)
                score = roc_auc_score(
                    validation_y,
                    _score_vector(
                        estimator,
                        _take_rows(x, valid_idx),
                    ),
                )
                record_fold(trial, score, step)
                scores.append(float(score))
                running = float(np.mean(scores))
                trial.report(running, step=step)

                if pruning_enabled and trial.should_prune():
                    raise optuna.TrialPruned()

            if len(scores) != len(prepared_splits):
                raise RuntimeError(
                    "No inner validation split contained both target classes."
                )
            return float(np.mean(scores))

    storage = _sqlite_storage(storage_path)
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
        sampler=TPESampler(seed=random_seed),
        pruner=pruner,
    )

    report = run_bounded_study(study, objective, contract, storage_path=storage_path,
                               progress_callback=progress_callback)
    logger.info("Optuna audit: %s", report)
    return dict(report["best_params"])
