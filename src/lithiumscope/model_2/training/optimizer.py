from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lithiumscope.core.logger import get_logger

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
) -> dict:
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.trial import TrialState
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

    def objective(trial):
        params = parameter_space(trial)
        scores: list[float] = []

        for step, (train_idx, valid_idx) in enumerate(splits):
            estimator = estimator_factory(params)
            estimator.fit(
                _take_rows(x, train_idx),
                _take_rows(y, train_idx),
            )
            validation_y = np.asarray(
                _take_rows(y, valid_idx),
                dtype=int,
            )
            if np.unique(validation_y).size < 2:
                continue

            score = roc_auc_score(
                validation_y,
                _score_vector(
                    estimator,
                    _take_rows(x, valid_idx),
                ),
            )
            scores.append(float(score))
            running = float(np.mean(scores))
            trial.report(running, step=step)

            if pruning_enabled and trial.should_prune():
                raise optuna.TrialPruned()

        if not scores:
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

    terminal_states = {
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.FAIL,
    }
    finished_before = sum(
        trial.state in terminal_states
        for trial in study.trials
    )
    remaining = max(0, n_trials - finished_before)

    if progress_callback is not None:
        progress_callback(
            min(finished_before, n_trials),
            n_trials,
        )

    callbacks = []
    if progress_callback is not None:
        def on_trial_complete(study, trial):
            finished = sum(
                item.state in terminal_states
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
        "Model 2 Optuna best ROC-AUC=%.6f params=%s "
        "complete=%d pruned=%d target_trials=%d storage=%s",
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
