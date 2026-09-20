from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lithiumscope.core.checkpoints import FoldCheckpointStore
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.scientific_checks import (
    assert_group_isolation,
    assert_oof_complete,
)
from lithiumscope.model_1.evaluation.metrics import regression_metrics
from lithiumscope.model_1.steps.step_09_validation import (
    materialize_regression_splits,
)
from lithiumscope.model_1.training.factory import build_pipeline, get_algorithm
from lithiumscope.model_1.training.optimizer import optimize_with_optuna
from lithiumscope.model_1.training.target_transform import (
    transform_label,
    validate_target_transform,
)
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("model_1.cv_runner")


@dataclass
class CVRunResult:
    algorithm: str
    label: str
    predictions: np.ndarray
    fold_table: pd.DataFrame
    overall_metrics: dict[str, float]
    elapsed_seconds: float
    final_params: dict
    target_transform: str = "identity"
    variant_id: str | None = None


def run_nested_cv(
    prepared,
    algorithm: str,
    device: DeviceInfo,
    config: dict,
    checkpoint_root: Path | None = None,
    *,
    target_transform: str = "identity",
    variant_id: str | None = None,
) -> CVRunResult:
    validation = config["validation"]
    optimization = config["optimization"]
    seed = int(validation["random_seed"])
    groups = (
        prepared.groups
        if bool(
            validation.get(
                "prefer_spatial_groups",
                False,
            )
        )
        else None
    )
    outer_splits, outer_strategy = (
        materialize_regression_splits(
            prepared.x,
            prepared.y,
            groups,
            int(validation["outer_folds"]),
            seed,
        )
    )
    spec = get_algorithm(algorithm)
    target_transform = validate_target_transform(
        target_transform,
        prepared.y,
    )
    variant_id = (
        variant_id
        or f"{algorithm}__{target_transform}"
    )
    label = (
        spec.label
        if target_transform == "identity"
        else (
            f"{spec.label} · "
            f"{transform_label(target_transform)}"
        )
    )
    predictions = np.full(len(prepared.y), np.nan, dtype=float)
    folds: list[dict] = []
    started = time.perf_counter()
    trials = int(
        optimization.get("per_algorithm_trials", {}).get(
            algorithm,
            optimization.get("n_trials", 20),
        )
    )
    optimize = bool(optimization.get("enabled", True))
    pruning = optimization.get("pruning", {})
    checkpoint = (
        FoldCheckpointStore(checkpoint_root, variant_id)
        if checkpoint_root is not None
        else None
    )

    print(f"\n  ▶ {label}")
    logger.info(
        "Starting Model 1 algorithm=%s variant=%s target_transform=%s "
        "samples=%d checkpoint=%s",
        algorithm,
        variant_id,
        target_transform,
        len(prepared.y),
        checkpoint_root,
    )

    print(
        "    Validación externa: "
        f"{outer_strategy}"
    )

    for fold, (train_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        if (
            groups is not None
            and outer_strategy == "group_kfold_spatial"
        ):
            assert_group_isolation(
                train_idx,
                test_idx,
                groups,
            )
        cached = checkpoint.load_fold(fold, test_idx) if checkpoint else None
        if cached is not None:
            fold_predictions = np.asarray(
                cached["predictions"],
                dtype=float,
            )
            predictions[test_idx] = fold_predictions
            metrics = dict(cached["metrics"])
            folds.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "validation_strategy": outer_strategy,
                    "resumed": True,
                }
            )
            print(
                f"      Fold externo {fold}/{len(outer_splits)} "
                f"↻ reutilizado | RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}"
            )
            continue

        spinner = Spinner(
            f"{label} · fold {fold}/{len(outer_splits)} · preparando"
        ).start()
        try:
            x_train = prepared.x.iloc[train_idx]
            y_train = prepared.y.iloc[train_idx]
            x_test = prepared.x.iloc[test_idx]
            y_test = prepared.y.iloc[test_idx]
            groups_train = (
                groups.iloc[train_idx]
                if groups is not None
                else None
            )

            params: dict = {}
            if optimize:
                def factory(candidate: dict):
                    return build_pipeline(
                        algorithm,
                        device,
                        seed,
                        prepared.schema,
                        candidate,
                        target_transform=target_transform,
                    )

                def progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · fold {fold}/{len(outer_splits)} · "
                        f"Optuna {done}/{total}"
                    )

                inner_splits, _ = materialize_regression_splits(
                    x_train,
                    y_train,
                    groups_train,
                    int(validation["inner_folds"]),
                    seed + fold,
                )
                params = optimize_with_optuna(
                    factory,
                    spec.module.optuna_space,
                    x_train,
                    y_train,
                    inner_splits,
                    trials,
                    progress_callback=progress,
                    storage_path=(
                        checkpoint.root / f"optuna_fold_{fold:02d}.db"
                        if checkpoint
                        else None
                    ),
                    study_name=f"{algorithm}_fold_{fold:02d}",
                    random_seed=seed + fold,
                    pruning_enabled=bool(
                        pruning.get("enabled", True)
                    ),
                    startup_trials=int(
                        pruning.get("startup_trials", 4)
                    ),
                    warmup_folds=int(
                        pruning.get("warmup_folds", 2)
                    ),
                )

            spinner.update(
                f"{label} · fold {fold}/{len(outer_splits)} · ajustando"
            )
            estimator = build_pipeline(
                algorithm,
                device,
                seed,
                prepared.schema,
                params,
                target_transform=target_transform,
            )
            estimator.fit(x_train, y_train)
            fold_predictions = estimator.predict(x_test)
            predictions[test_idx] = fold_predictions
            metrics = regression_metrics(
                y_test,
                fold_predictions,
            ).to_dict()
            folds.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "validation_strategy": outer_strategy,
                    "resumed": False,
                }
            )
            if checkpoint:
                checkpoint.save_fold(
                    fold,
                    test_idx,
                    fold_predictions,
                    metrics,
                    params=params,
                )

            spinner.succeed(
                f"{label} · fold {fold}/{len(outer_splits)} | "
                f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
                f"R²={metrics['r2']:.4f}"
            )
            logger.info(
                "algorithm=%s fold=%d metrics=%s",
                algorithm,
                fold,
                metrics,
            )
        except BaseException:
            spinner.fail(
                f"{label} · fold {fold}/{len(outer_splits)} interrumpido"
            )
            raise

    assert_oof_complete(predictions, len(prepared.y))
    overall = regression_metrics(
        prepared.y,
        predictions,
    ).to_dict()
    fold_table = pd.DataFrame(folds)

    algorithm_cache = checkpoint.load_algorithm() if checkpoint else None
    if algorithm_cache is not None:
        final_params = dict(algorithm_cache.get("final_params", {}))
        previous_elapsed = float(algorithm_cache.get("elapsed_seconds", 0.0))
    else:
        final_params: dict = {}
        if optimize:
            spinner = Spinner(
                f"{label} · optimización final de hiperparámetros"
            ).start()
            try:
                def final_factory(candidate: dict):
                    return build_pipeline(
                        algorithm,
                        device,
                        seed,
                        prepared.schema,
                        candidate,
                        target_transform=target_transform,
                    )

                def final_progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · optimización final {done}/{total}"
                    )

                final_splits, _ = materialize_regression_splits(
                    prepared.x,
                    prepared.y,
                    groups,
                    int(validation["inner_folds"]),
                    seed + 5000,
                )
                final_params = optimize_with_optuna(
                    final_factory,
                    spec.module.optuna_space,
                    prepared.x,
                    prepared.y,
                    final_splits,
                    trials,
                    progress_callback=final_progress,
                    storage_path=(
                        checkpoint.root / "optuna_final.db"
                        if checkpoint
                        else None
                    ),
                    study_name=f"{algorithm}_final",
                    random_seed=seed + 5000,
                    pruning_enabled=bool(
                        pruning.get("enabled", True)
                    ),
                    startup_trials=int(
                        pruning.get("startup_trials", 4)
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

    elapsed = previous_elapsed + (time.perf_counter() - started)
    if checkpoint and algorithm_cache is None:
        checkpoint.save_algorithm(
            {
                "algorithm": algorithm,
                "label": label,
                "variant_id": variant_id,
                "target_transform": target_transform,
                "final_params": final_params,
                "overall_metrics": overall,
                "elapsed_seconds": elapsed,
            }
        )

    return CVRunResult(
        algorithm=algorithm,
        label=label,
        predictions=predictions,
        fold_table=fold_table,
        overall_metrics=overall,
        elapsed_seconds=elapsed,
        final_params=final_params,
        target_transform=target_transform,
        variant_id=variant_id,
    )
