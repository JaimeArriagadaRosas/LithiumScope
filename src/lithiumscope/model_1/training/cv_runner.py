from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.scientific_checks import assert_oof_complete
from lithiumscope.model_1.evaluation.metrics import regression_metrics
from lithiumscope.model_1.steps.step_09_validation import nested_cv
from lithiumscope.model_1.training.factory import build_pipeline, get_algorithm
from lithiumscope.model_1.training.optimizer import optimize_with_optuna

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


def run_nested_cv(prepared, algorithm: str, device: DeviceInfo, config: dict) -> CVRunResult:
    validation = config["validation"]
    optimization = config["optimization"]
    seed = int(validation["random_seed"])
    outer, inner = nested_cv(
        int(validation["outer_folds"]),
        int(validation["inner_folds"]),
        seed,
    )
    spec = get_algorithm(algorithm)
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

    print(f"\n  ▶ {spec.label}")
    logger.info("Starting Model 1 algorithm=%s samples=%d", algorithm, len(prepared.y))

    for fold, (train_idx, test_idx) in enumerate(outer.split(prepared.x), start=1):
        print(f"      Fold externo {fold}/{outer.get_n_splits()} ...", end=" ", flush=True)
        x_train = prepared.x.iloc[train_idx]
        y_train = prepared.y.iloc[train_idx]
        x_test = prepared.x.iloc[test_idx]
        y_test = prepared.y.iloc[test_idx]

        params: dict = {}
        if optimize:
            def factory(candidate: dict):
                return build_pipeline(algorithm, device, seed, prepared.schema, candidate)

            params = optimize_with_optuna(
                factory,
                spec.module.optuna_space,
                x_train,
                y_train,
                inner,
                trials,
            )

        estimator = build_pipeline(algorithm, device, seed, prepared.schema, params)
        estimator.fit(x_train, y_train)
        fold_predictions = estimator.predict(x_test)
        predictions[test_idx] = fold_predictions
        metrics = regression_metrics(y_test, fold_predictions).to_dict()
        folds.append(
            {
                "fold": fold,
                **metrics,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
        print(
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}"
        )
        logger.info("algorithm=%s fold=%d metrics=%s", algorithm, fold, metrics)

    assert_oof_complete(predictions, len(prepared.y))
    overall = regression_metrics(prepared.y, predictions).to_dict()
    fold_table = pd.DataFrame(folds)

    final_params: dict = {}
    if optimize:
        def final_factory(candidate: dict):
            return build_pipeline(algorithm, device, seed, prepared.schema, candidate)

        final_params = optimize_with_optuna(
            final_factory,
            spec.module.optuna_space,
            prepared.x,
            prepared.y,
            inner,
            trials,
        )

    elapsed = time.perf_counter() - started
    return CVRunResult(
        algorithm,
        spec.label,
        predictions,
        fold_table,
        overall,
        elapsed,
        final_params,
    )
