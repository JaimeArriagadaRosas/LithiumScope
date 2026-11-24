"""Spatially separated outer folds with labels derived from training data only.

No target information is used to allocate the folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.core.scientific_checks import assert_group_isolation


@dataclass(frozen=True)
class ScientificFold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    label_threshold_ppm: float
    y_train: np.ndarray
    y_test: np.ndarray
    n_groups_train: int
    n_groups_test: int


def make_scientific_folds(
    lithium: pd.Series,
    groups: pd.Series | None,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    quantile: float = 0.75,
    fixed_threshold_ppm: float | None = None,
) -> list[ScientificFold]:
    values = np.asarray(lithium, dtype=float)
    if len(values) < n_splits or not np.all(np.isfinite(values)):
        raise ValueError("Modelo 2: litio no finito o muestras insuficientes.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("El cuantil debe estar entre 0 y 1.")
    if groups is None:
        raise ValueError("Modelo 2 científico requiere grupos espaciales.")
    groups = pd.Series(groups).reset_index(drop=True)
    if groups.isna().any() or groups.nunique() < n_splits:
        raise ValueError("Modelo 2: grupos espaciales faltantes o insuficientes.")
    if fixed_threshold_ppm is not None and not np.isfinite(fixed_threshold_ppm):
        raise ValueError("El umbral fijo de Li debe ser finito.")

    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in splitter.split(values.reshape(-1, 1), groups=groups):
        assert_group_isolation(train_idx, test_idx, groups)
        threshold = (
            float(fixed_threshold_ppm)
            if fixed_threshold_ppm is not None
            else float(np.quantile(values[train_idx], quantile))
        )
        y_train = (values[train_idx] >= threshold).astype(int)
        y_test = (values[test_idx] >= threshold).astype(int)
        if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
            raise ValueError(
                "Modelo 2: fold espacial con una sola clase. "
                "Revise los grupos o cambie el número de folds; "
                "no se realizará fallback aleatorio."
            )
        folds.append(ScientificFold(
            train_idx=np.asarray(train_idx, dtype=int),
            test_idx=np.asarray(test_idx, dtype=int),
            label_threshold_ppm=threshold,
            y_train=y_train,
            y_test=y_test,
            n_groups_train=int(groups.iloc[train_idx].nunique()),
            n_groups_test=int(groups.iloc[test_idx].nunique()),
        ))
    return folds


def scientific_labels(folds: list[ScientificFold], count: int) -> np.ndarray:
    labels = np.full(count, -1, dtype=int)
    for fold in folds:
        if np.any(labels[fold.test_idx] != -1):
            raise ValueError("Una observación aparece en más de un fold.")
        labels[fold.test_idx] = fold.y_test
    if np.any(labels < 0):
        raise ValueError("Faltan etiquetas de evaluación exterior.")
    return labels



def evaluate_scientific_baseline(
    folds: list[ScientificFold], count: int,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    from lithiumscope.model_2.evaluation.metrics import classification_metrics

    scores = np.full(count, np.nan, dtype=float)
    records = []
    for i, fold in enumerate(folds, start=1):
        prior = float(np.mean(fold.y_train))
        scores[fold.test_idx] = prior
        metrics = classification_metrics(fold.y_test, np.full(len(fold.test_idx), prior))
        records.append({
            "fold": i,
            **metrics,
            "n_train": len(fold.train_idx),
            "n_test": len(fold.test_idx),
            "label_threshold_ppm": fold.label_threshold_ppm,
            "group_overlap": 0,
        })
    table = pd.DataFrame(records)
    row = {
        "algorithm": "baseline_prior",
        "label": "Baseline - prevalencia de clase por fold",
        "status": "ok",
        "is_baseline": True,
        "elapsed_seconds": 0.0,
    }
    for metric in ["roc_auc", "average_precision", "balanced_accuracy", "f1",
                   "precision", "recall", "brier_score"]:
        row[f"{metric}_mean"] = float(table[metric].mean())
    row["roc_auc_std"] = float(table["roc_auc"].std(ddof=0))
    return row, table, scores



def make_anchored_scientific_folds(
    lithium: pd.Series,
    groups: pd.Series,
    legacy_folds: list[dict],
    *,
    baseline_count: int,
    quantile: float = .75,
    fixed_threshold_ppm: float | None = None,
) -> list[ScientificFold]:
    """Preserve exactly the historical 706-row outer tests and their labels.

    Additional geographical groups are distributed across the same five folds.
    Fold label thresholds are derived from the original historical training
    subset, never from the held-out lithium values or added GEOROC labels.
    """
    values = np.asarray(lithium, dtype=float)
    group_values = pd.Series(groups).reset_index(drop=True)
    if len(values) != len(group_values) or not np.isfinite(values).all():
        raise ValueError("Invalid anchored Model 2 Li or group vector")
    if not 0 < baseline_count < len(values):
        raise ValueError("Anchored M2 requires an extended source manifest")
    if group_values.isna().any() or not 0.0 < quantile < 1.0:
        raise ValueError("Invalid anchor grouping or quantile")
    n_folds = len(legacy_folds)
    if n_folds < 2:
        raise ValueError("At least two legacy fold manifests required")
    baseline_set = set(group_values.iloc[:baseline_count])
    new_groups = sorted(set(group_values.iloc[baseline_count:]))
    if baseline_set.intersection(new_groups):
        raise ValueError("GEOROC expansion overlaps a historical spatial group")
    assignment = {group: index % n_folds for index, group in enumerate(new_groups)}
    extra = np.arange(baseline_count,len(values))
    seen_old = set()
    seen_new = set()
    folds = []
    for index, record in enumerate(legacy_folds):
        historical_train = np.asarray(record["train_indices"], dtype=int)
        historical_test = np.asarray(record["test_indices"], dtype=int)
        if (len(historical_train) == 0 or len(historical_test) == 0
                or historical_train.min() < 0 or historical_test.min() < 0
                or historical_train.max() >= baseline_count
                or historical_test.max() >= baseline_count):
            raise ValueError("Legacy folds reference incorrect historical rows")
        if (set(historical_train).intersection(historical_test)
                or len(set(historical_train).union(historical_test)) != baseline_count):
            raise ValueError("Legacy fold train/test indices are incomplete")
        if seen_old.intersection(historical_test):
            raise ValueError("Legacy test rows appear in multiple folds")
        seen_old.update(historical_test.tolist())
        extra_test = np.asarray(
            [i for i in extra if assignment[group_values.iloc[i]] == index],
            dtype=int,
        )
        seen_new.update(extra_test.tolist())
        extra_train = np.asarray(
            [i for i in extra if assignment[group_values.iloc[i]] != index],
            dtype=int,
        )
        train_idx = np.concatenate([historical_train, extra_train])
        test_idx = np.concatenate([historical_test, extra_test])
        assert_group_isolation(train_idx,test_idx,group_values)
        threshold = (
            float(fixed_threshold_ppm)
            if fixed_threshold_ppm is not None
            else float(np.quantile(values[historical_train], quantile))
        )
        y_train = (values[train_idx] >= threshold).astype(int)
        y_test = (values[test_idx] >= threshold).astype(int)
        if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
            raise ValueError("Anchored spatial fold lacks a class")
        folds.append(ScientificFold(
            train_idx=train_idx,
            test_idx=test_idx,
            label_threshold_ppm=threshold,
            y_train=y_train,
            y_test=y_test,
            n_groups_train=int(group_values.iloc[train_idx].nunique()),
            n_groups_test=int(group_values.iloc[test_idx].nunique()),
        ))
    if seen_old != set(range(baseline_count)):
        raise ValueError("Not all historical test rows were covered")
    if seen_new != set(extra.tolist()):
        raise ValueError("Not all expanded samples were tested")
    return folds
