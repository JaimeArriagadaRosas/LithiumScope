from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from lithiumscope.core.scientific_checks import assert_group_isolation, assert_oof_complete
from lithiumscope.model_2.evaluation.metrics import classification_metrics


def evaluate_prior_baseline(
    x: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
    groups: pd.Series | None,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    if groups is not None and groups.nunique() >= folds:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(x, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(x, y)

    probabilities = np.full(len(y), np.nan, dtype=float)
    rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        if groups is not None and groups.nunique() >= folds:
            assert_group_isolation(train_idx, test_idx, groups)
        estimator = DummyClassifier(strategy="prior")
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        fold_probabilities = estimator.predict_proba(x.iloc[test_idx])[:, 1]
        probabilities[test_idx] = fold_probabilities
        metrics = classification_metrics(y.iloc[test_idx].to_numpy(), fold_probabilities)
        rows.append({"fold": fold, **metrics})

    assert_oof_complete(probabilities, len(y))
    table = pd.DataFrame(rows)
    ranking_row = {
        "algorithm": "baseline_prior",
        "label": "Baseline — prevalencia de clase",
        "roc_auc_mean": float(table["roc_auc"].mean()),
        "roc_auc_std": float(table["roc_auc"].std(ddof=0)),
        "average_precision_mean": float(table["average_precision"].mean()),
        "balanced_accuracy_mean": float(table["balanced_accuracy"].mean()),
        "f1_mean": float(table["f1"].mean()),
        "precision_mean": float(table["precision"].mean()),
        "recall_mean": float(table["recall"].mean()),
        "brier_score_mean": float(table["brier_score"].mean()),
        "elapsed_seconds": 0.0,
        "status": "ok",
        "is_baseline": True,
    }
    return ranking_row, table, probabilities
