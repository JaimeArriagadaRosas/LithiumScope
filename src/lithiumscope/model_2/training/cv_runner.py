from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold

from lithiumscope.core.checkpoints import FoldCheckpointStore
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.scientific_checks import (
    assert_group_isolation,
    assert_oof_complete,
)
from lithiumscope.model_2.evaluation.metrics import classification_metrics
from lithiumscope.model_2.training.scientific_validation import ScientificFold
from lithiumscope.model_2.training.factory import (
    create_model,
    get_algorithm,
    get_label,
)
from lithiumscope.model_2.training.optimizer import (
    optimize_classifier_with_optuna,
)
from lithiumscope.model_2.training.search_policy import parameter_space_for
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
    scientific_folds: list[ScientificFold] | None = None,
    lithium_values=None,
    label_quantile: float = 0.75,
    fixed_threshold_ppm: float | None = None,
    quality_training_mask: np.ndarray | None = None,
) -> ClassificationCVResult:
    if quality_training_mask is not None:
        quality_training_mask = np.asarray(quality_training_mask, dtype=bool)
        if quality_training_mask.shape != (len(x),) or not quality_training_mask.any():
            raise ValueError("Train-only quality mask must align with all observations")
    if scientific_folds is not None:
        if lithium_values is None or groups is None:
            raise ValueError("Scientific CV requires Li values and spatial groups.")
        split_groups = groups
        splitter = None
    else:
        splitter, split_groups = _splitter(
            y, groups, folds, seed, announce=False,
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
    parameter_space = parameter_space_for(algorithm, spec, optimization)
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
        [(f.train_idx, f.test_idx) for f in scientific_folds]
        if scientific_folds is not None else (
            splitter.split(x, y, split_groups)
            if split_groups is not None else splitter.split(x, y)
        )
    )
    _evaluate_outer_classification_folds(algorithm, checkpoint, device, fixed_threshold_ppm, folds, groups, inner_folds, label, label_quantile, lithium_values, optimize, probabilities, pruning, quality_training_mask, records, scientific_folds, seed, spec, split_groups, split_iter, trials, x, y, parameter_space)

    assert_oof_complete(probabilities, len(y))
    final_indices = (
        np.flatnonzero(quality_training_mask)
        if quality_training_mask is not None
        else np.arange(len(y), dtype=int)
    )
    final_x = x.iloc[final_indices]
    final_y = y.iloc[final_indices]
    final_groups = (
        groups.iloc[final_indices] if groups is not None else None
    )
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
                if scientific_folds is not None:
                    if final_groups is None or final_groups.nunique() < inner_folds:
                        raise ValueError("Insufficient groups for final parameter search.")
                    final_splits = list(GroupKFold(n_splits=inner_folds).split(
                        final_x, groups=final_groups,
                    ))
                else:
                    final_splits = _materialize_splits(
                        final_x, final_y, final_groups, inner_folds, seed + 5000,
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
                    parameter_space,
                    final_x,
                    final_y,
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
                    warmup_folds=int(pruning.get("warmup_folds", 2)),
                    lithium_values=(
                        np.asarray(lithium_values, dtype=float)[final_indices]
                        if scientific_folds is not None else None
                    ),
                    label_quantile=label_quantile,
                    fixed_threshold_ppm=fixed_threshold_ppm,
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

from lithiumscope.model_2.training.cv_runner_fold_utilities import _splitter, _materialize_splits, _fit_estimator

from lithiumscope.model_2.training.cv_runner_evaluate_outer_classification_folds import _evaluate_outer_classification_folds
