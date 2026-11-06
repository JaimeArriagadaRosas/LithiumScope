"""Scientifically scoped, auditable out-of-fold diagnostics.

These reports describe internal CV only; they are NOT external validation.
Bootstrap units are spatial groups when those are available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, roc_auc_score,
)


def _groups(groups, count: int) -> np.ndarray:
    if groups is None:
        return np.asarray([f"row_{i}" for i in range(count)], dtype=str)
    values = pd.Series(groups).reset_index(drop=True).astype(str).to_numpy()
    if len(values) != count:
        raise ValueError("Wrong number of spatial groups.")
    return values


def _summary(values: np.ndarray, *, kind: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    if kind == "regression":
        return {
            "mae": float(np.mean(np.abs(finite[:, 0] - finite[:, 1]))),
            "rmse": float(np.sqrt(np.mean(np.square(finite[:, 0] - finite[:, 1])))),
            "bias_ppm": float(np.mean(finite[:, 1] - finite[:, 0])),
            "median_ae": float(np.median(np.abs(finite[:, 0] - finite[:, 1]))),
            "p90_ae": float(np.quantile(np.abs(finite[:, 0] - finite[:, 1]), 0.90)),
        }
    y = finite[:, 0].astype(int)
    score = finite[:, 1]
    predicted = score >= 0.5
    out = {
        "balanced_accuracy_0_5": float(balanced_accuracy_score(y, predicted)),
        "prevalence": float(y.mean()),
        "brier_score": float(np.mean((score - y) ** 2)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["average_precision"] = float(average_precision_score(y, score))
    return out


def _bootstrap(
    pairs: np.ndarray, group_names: np.ndarray, *, kind: str,
    repetitions: int = 300, seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    unique, inverse = np.unique(group_names, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    generator = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {}
    for _ in range(repetitions):
        selected = generator.integers(0, len(unique), len(unique))
        indices = np.concatenate([members[i] for i in selected])
        # A resample with only one class does not define ROC-AUC/AP.
        if kind == "classification" and len(np.unique(pairs[indices, 0])) < 2:
            continue
        for metric, value in _summary(pairs[indices], kind=kind).items():
            estimates.setdefault(metric, []).append(value)
    return {
        metric: {
            "n": len(sample),
            "lower_95": float(np.quantile(sample, 0.025)),
            "median": float(np.median(sample)),
            "upper_95": float(np.quantile(sample, 0.975)),
        }
        for metric, sample in estimates.items() if sample
    }


def save_oof_diagnostics(
    path: Path, y_true, prediction, *,
    groups=None, kind: str, seed: int = 42,
    source_label: str = "internal_outer_cv",
) -> dict:
    """Write raw OOF errors, per-site statistics, group bootstrap and scope."""
    if kind not in {"regression", "classification"}:
        raise ValueError("Unsupported diagnostic kind.")
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("OOF predictions and targets must be aligned.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("OOF predictions must be finite.")
    if kind == "classification" and not np.isin(observed, [0, 1]).all():
        raise ValueError("Classification labels must be binary.")
    keys = _groups(groups, len(observed))
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "row_index": np.arange(len(observed)), "spatial_group": keys,
        "y_true": observed, "prediction": predicted,
        "residual": predicted - observed,
        "absolute_error": np.abs(predicted - observed),
    })
    frame.to_csv(path / "oof_diagnostics.csv", index=False)
    pairs = np.column_stack([observed, predicted])
    records = []
    for group, indices in frame.groupby("spatial_group").indices.items():
        detail = _summary(pairs[indices], kind=kind)
        records.append({"spatial_group": group, "n_samples": len(indices), **detail})
    pd.DataFrame(records).to_csv(path / "per_spatial_group.csv", index=False)
    summary = {
        "scope": source_label,
        "external_validation": "NOT_EVALUATED",
        "model_selection_bias": (
            "Winner selected using the same outer folds; winning metrics "
            "are selection-affected and not independent holdout estimates."
        ),
        "n_samples": len(observed),
        "n_groups": len(np.unique(keys)),
        "metrics": _summary(pairs, kind=kind),
        "cluster_bootstrap": _bootstrap(pairs, keys, kind=kind, seed=seed),
        "bootstrap_units": "spatial_group" if groups is not None else "row",
        "random_seed": seed,
    }
    (path / "scientific_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
