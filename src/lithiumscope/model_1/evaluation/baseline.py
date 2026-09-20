from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from lithiumscope.core.scientific_checks import (
    assert_group_isolation,
    assert_oof_complete,
)
from lithiumscope.model_1.evaluation.metrics import regression_metrics
from lithiumscope.model_1.steps.step_09_validation import (
    materialize_regression_splits,
)


def evaluate_mean_baseline(prepared, config: dict) -> tuple[dict, pd.DataFrame, np.ndarray]:
    validation = config["validation"]
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
    outer_splits, strategy = materialize_regression_splits(
        prepared.x,
        prepared.y,
        groups,
        int(validation["outer_folds"]),
        int(validation["random_seed"]),
    )
    predictions = np.full(len(prepared.y), np.nan, dtype=float)
    rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        if (
            groups is not None
            and strategy == "group_kfold_spatial"
        ):
            assert_group_isolation(
                train_idx,
                test_idx,
                groups,
            )
        estimator = DummyRegressor(strategy="mean")
        estimator.fit(prepared.x.iloc[train_idx], prepared.y.iloc[train_idx])
        fold_predictions = estimator.predict(prepared.x.iloc[test_idx])
        predictions[test_idx] = fold_predictions
        metrics = regression_metrics(prepared.y.iloc[test_idx], fold_predictions).to_dict()
        rows.append(
            {
                "fold": fold,
                **metrics,
                "validation_strategy": strategy,
            }
        )

    assert_oof_complete(predictions, len(prepared.y))
    fold_table = pd.DataFrame(rows)
    ranking_row = {
        "algorithm": "baseline_mean",
        "label": "Baseline — media de entrenamiento",
        "rmse_mean": float(fold_table["rmse"].mean()),
        "rmse_std": float(fold_table["rmse"].std(ddof=0)),
        "mae_mean": float(fold_table["mae"].mean()),
        "mae_std": float(fold_table["mae"].std(ddof=0)),
        "r2_mean": float(fold_table["r2"].mean()),
        "r2_std": float(fold_table["r2"].std(ddof=0)),
        "median_ae_mean": float(fold_table["median_ae"].mean()),
        "explained_variance_mean": float(fold_table["explained_variance"].mean()),
        "elapsed_seconds": 0.0,
        "status": "ok",
        "is_baseline": True,
    }
    return ranking_row, fold_table, predictions
