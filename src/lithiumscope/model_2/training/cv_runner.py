from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from lithiumscope.core.checkpoints import FoldCheckpointStore
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.scientific_checks import (
    assert_group_isolation,
    assert_oof_complete,
)
from lithiumscope.model_2.evaluation.metrics import classification_metrics
from lithiumscope.model_2.training.factory import (
    create_model,
    get_algorithm,
    get_label,
)
from lithiumscope.model_2.training.optimizer import (
    optimize_classifier_with_optuna,
)
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("model_2.cv_runner")


@dataclass
class ClassificationCVResult:
    algorithm: str
    label: str
    probabilities: np.ndarray
    fold_table: pd.DataFrame
    overall_metrics: dict[str, float]
    elapsed_seconds: float
    final_params: dict


def _splitter(
    y: pd.Series,
    groups: pd.Series | None,
    folds: int,
    seed: int,
    *,
    announce: bool = False,
):
    if groups is not None and groups.nunique() >= folds:
        if announce:
            print(
                "    Validación espacial agrupada: "
                f"{groups.nunique()} grupos"
            )
        logger.info(
            "Model 2 validation: StratifiedGroupKFold with %d spatial groups",
            groups.nunique(),
        )
        return (
            StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            ),
            groups,
        )

    if announce:
        print(
            "    Aviso: sin grupos espaciales suficientes; "
            "se usará StratifiedKFold."
        )
    logger.warning(
        "Model 2 spatial groups unavailable/insufficient; "
        "falling back to StratifiedKFold"
    )
    return (
        StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        ),
        None,
    )


def _materialize_splits(
    x,
    y,
    groups,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter, split_groups = _splitter(
        y,
        groups,
        folds,
        seed,
        announce=False,
    )
    iterator = (
        splitter.split(x, y, split_groups)
        if split_groups is not None
        else splitter.split(x, y)
    )
    return [
        (
            np.asarray(train_idx, dtype=int),
            np.asarray(valid_idx, dtype=int),
        )
        for train_idx, valid_idx in iterator
    ]


def _fit_estimator(
    algorithm: str,
    device: DeviceInfo,
    seed: int,
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    inner_folds: int,
):
    calibration_cv = None
    if algorithm == "svm_rbf":
        calibration_cv = _materialize_splits(
            x_train,
            y_train,
            groups_train,
            inner_folds,
            seed + 1000,
        )

    estimator = create_model(
        algorithm,
        device,
        seed,
        params,
        calibrated=True,
        calibration_cv=calibration_cv,
    )
    estimator.fit(x_train, y_train)
    return estimator


def run_classification_cv(
    x: pd.DataFrame,
    y: pd.Series,
    algorithm: str,
    device: DeviceInfo,
    seed: int,
    folds: int,
    groups: pd.Series | None = None,
    checkpoint_root: Path | None = None,
    optimization: dict | None = None,
    inner_folds: int = 3,
) -> ClassificationCVResult:
    splitter, split_groups = _splitter(
        y,
        groups,
        folds,
        seed,
        announce=False,
    )
    probabilities = np.full(len(y), np.nan, dtype=float)
    records: list[dict] = []
    started = time.perf_counter()
    label = get_label(algorithm)
    spec = get_algorithm(algorithm)
    checkpoint = (
        FoldCheckpointStore(checkpoint_root, algorithm)
        if checkpoint_root is not None
        else None
    )

    optimization = optimization or {}
    optimize = bool(optimization.get("enabled", False))
    trials = int(
        optimization.get("per_algorithm_trials", {}).get(
            algorithm,
            optimization.get("n_trials", 8),
        )
    )
    pruning = optimization.get("pruning", {})
    print(f"\n  ▶ {label}")

    split_iter = (
        splitter.split(x, y, split_groups)
        if split_groups is not None
        else splitter.split(x, y)
    )
    for fold, (train_idx, test_idx) in enumerate(
        split_iter,
        start=1,
    ):
        if split_groups is not None:
            assert_group_isolation(
                train_idx,
                test_idx,
                split_groups,
            )

        cached = (
            checkpoint.load_fold(fold, test_idx)
            if checkpoint
            else None
        )
        if cached is not None:
            fold_probabilities = np.asarray(
                cached["predictions"],
                dtype=float,
            )
            probabilities[test_idx] = fold_probabilities
            metrics = dict(cached["metrics"])
            records.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "resumed": True,
                }
            )
            print(
                f"      Fold {fold}/{folds} ↻ reutilizado | "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                f"AP={metrics['average_precision']:.4f}"
            )
            continue

        spinner = Spinner(
            f"{label} · fold {fold}/{folds} · preparando"
        ).start()
        try:
            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_test = y.iloc[test_idx]
            groups_train = (
                groups.iloc[train_idx]
                if groups is not None
                else None
            )

            params: dict = {}
            if optimize:
                inner_splits = _materialize_splits(
                    x_train,
                    y_train,
                    groups_train,
                    inner_folds,
                    seed + fold,
                )

                def factory(candidate: dict):
                    return create_model(
                        algorithm,
                        device,
                        seed,
                        candidate,
                        calibrated=False,
                    )

                def progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · fold {fold}/{folds} · "
                        f"Optuna {done}/{total}"
                    )

                params = optimize_classifier_with_optuna(
                    factory,
                    spec.module.optuna_space,
                    x_train,
                    y_train,
                    inner_splits,
                    n_trials=trials,
                    random_seed=seed + fold,
                    progress_callback=progress,
                    storage_path=(
                        checkpoint.root
                        / f"optuna_fold_{fold:02d}.db"
                        if checkpoint
                        else None
                    ),
                    study_name=f"{algorithm}_fold_{fold:02d}",
                    pruning_enabled=bool(
                        pruning.get("enabled", True)
                    ),
                    startup_trials=int(
                        pruning.get("startup_trials", 3)
                    ),
                    warmup_folds=int(
                        pruning.get("warmup_folds", 2)
                    ),
                )

            spinner.update(
                f"{label} · fold {fold}/{folds} · ajustando"
            )
            estimator = _fit_estimator(
                algorithm,
                device,
                seed,
                params,
                x_train,
                y_train,
                groups_train,
                inner_folds,
            )
            fold_probabilities = estimator.predict_proba(
                x_test
            )[:, 1]
            probabilities[test_idx] = fold_probabilities
            metrics = classification_metrics(
                y_test.to_numpy(),
                fold_probabilities,
            )
            records.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "resumed": False,
                }
            )
            if checkpoint:
                checkpoint.save_fold(
                    fold,
                    test_idx,
                    fold_probabilities,
                    metrics,
                    params=params,
                )
            spinner.succeed(
                f"{label} · fold {fold}/{folds} | "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                f"AP={metrics['average_precision']:.4f} | "
                f"F1={metrics['f1']:.4f}"
            )
        except BaseException:
            spinner.fail(
                f"{label} · fold {fold}/{folds} interrumpido"
            )
            raise

    assert_oof_complete(probabilities, len(y))
    overall = classification_metrics(
        y.to_numpy(),
        probabilities,
    )
    fold_table = pd.DataFrame(records)

    algorithm_cache = (
        checkpoint.load_algorithm()
        if checkpoint
        else None
    )
    if algorithm_cache is not None:
        final_params = dict(
            algorithm_cache.get("final_params", {})
        )
        previous_elapsed = float(
            algorithm_cache.get("elapsed_seconds", 0.0)
        )
    else:
        final_params: dict = {}
        if optimize:
            spinner = Spinner(
                f"{label} · optimización final"
            ).start()
            try:
                final_splits = _materialize_splits(
                    x,
                    y,
                    groups,
                    inner_folds,
                    seed + 5000,
                )

                def final_factory(candidate: dict):
                    return create_model(
                        algorithm,
                        device,
                        seed,
                        candidate,
                        calibrated=False,
                    )

                def final_progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · optimización final "
                        f"{done}/{total}"
                    )

                final_params = optimize_classifier_with_optuna(
                    final_factory,
                    spec.module.optuna_space,
                    x,
                    y,
                    final_splits,
                    n_trials=trials,
                    random_seed=seed + 5000,
                    progress_callback=final_progress,
                    storage_path=(
                        checkpoint.root / "optuna_final.db"
                        if checkpoint
                        else None
                    ),
                    study_name=f"{algorithm}_final",
                    pruning_enabled=bool(
                        pruning.get("enabled", True)
                    ),
                    startup_trials=int(
                        pruning.get("startup_trials", 3)
                    ),
                    warmup_folds=int(
                        pruning.get("warmup_folds", 2)
                    ),
                )
                spinner.succeed(
                    f"{label} · hiperparámetros finales listos"
                )
            except BaseException:
                spinner.fail(
                    f"{label} · optimización final interrumpida"
                )
                raise
        previous_elapsed = 0.0

    elapsed = (
        previous_elapsed
        + (time.perf_counter() - started)
    )
    if checkpoint and algorithm_cache is None:
        checkpoint.save_algorithm(
            {
                "algorithm": algorithm,
                "label": label,
                "final_params": final_params,
                "overall_metrics": overall,
                "elapsed_seconds": elapsed,
            }
        )

    return ClassificationCVResult(
        algorithm=algorithm,
        label=label,
        probabilities=probabilities,
        fold_table=fold_table,
        overall_metrics=overall,
        elapsed_seconds=elapsed,
        final_params=final_params,
    )
