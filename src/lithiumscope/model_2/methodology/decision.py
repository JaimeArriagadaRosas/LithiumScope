"""Strict train-only inner-OOF threshold selection for future experiments.

No estimator fitting is performed here. Upstream must generate inner-fold
predictions with group-disjoint fits. The outer holdout never influences
threshold selection. Calibration fitting requires a separate nested protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from lithiumscope.model_2.evaluation.thresholds import (
    ThresholdSelection, select_operating_threshold,
)
from lithiumscope.model_2.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class FrozenDecision:
    selection: ThresholdSelection
    inner_count: int
    group_count: int
    inner_folds: int
    inner_rows_sha256: str
    provenance: str = "inner_group_oof_train_only"


def freeze_inner_oof_threshold(
    inner_predictions: pd.DataFrame,
    *,
    permitted_outer_train_indices,
    reserved_outer_test_indices,
    objective: str = "balanced_accuracy",
) -> FrozenDecision:
    """Accept exactly one independently held-out prediction per train row.

    Columns: row_index (original global index), group, inner_fold, y_true, score.
    Each whole group must belong to only one inner validation fold.
    This validates the OOF *manifest*, not the model-fit provenance itself.
    """
    columns = {"row_index", "group", "inner_fold", "y_true", "score"}
    if not columns.issubset(inner_predictions):
        raise ValueError(f"Inner OOF missing: {sorted(columns-set(inner_predictions))}")
    frame = inner_predictions.copy().reset_index(drop=True)
    if frame.empty or frame.isna().any().any():
        raise ValueError("Inner OOF must be nonempty without missing values")
    try:
        row = frame.row_index.to_numpy(dtype=int)
        fold = frame.inner_fold.to_numpy(dtype=int)
        y = frame.y_true.to_numpy(dtype=int)
        scores = frame.score.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid inner OOF numeric columns") from exc
    if len(set(row)) != len(row):
        raise ValueError("Inner OOF rows appear more than once")
    allowed = set(int(x) for x in permitted_outer_train_indices)
    reserved = set(int(x) for x in reserved_outer_test_indices)
    if allowed.intersection(reserved):
        raise ValueError("Outer train and test index overlap")
    if set(row) != allowed or set(row).intersection(reserved):
        raise ValueError("Inner predictions must cover ONLY the full outer train")
    if len(np.unique(fold)) < 2 or len(np.unique(y)) != 2:
        raise ValueError("Inner validation requires multiple folds and both classes")
    if not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("Scores must be finite within [0,1]")
    if not set(y).issubset({0, 1}):
        raise ValueError("Binary labels required")
    # All observations of a spatial group must share their validation fold.
    if frame.groupby("group", observed=True).inner_fold.nunique().max() != 1:
        raise ValueError("A spatial group is fragmented across inner validation folds")
    for _, sub in frame.groupby("inner_fold", observed=True):
        if sub.y_true.nunique() != 2:
            raise ValueError("Inner fold lacks both labels")
    selected = select_operating_threshold(y, scores, objective=objective)
    digest = hashlib.sha256(
        np.asarray(sorted(row), dtype="<i8").tobytes()
    ).hexdigest()
    return FrozenDecision(
        selection=selected,
        inner_count=len(frame),
        group_count=int(frame.group.nunique()),
        inner_folds=int(frame.inner_fold.nunique()),
        inner_rows_sha256=digest,
    )


def evaluate_frozen_threshold(y_outer, scores_outer, frozen: FrozenDecision) -> dict:
    """Evaluate an independently held-out outer test with no re-selection."""
    y = np.asarray(y_outer, dtype=int)
    scores = np.asarray(scores_outer, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("Outer labels and scores must have equal 1D shape")
    if len(y) < 2 or len(np.unique(y)) < 2 or not np.isfinite(scores).all():
        raise ValueError("Outer test requires two classes and finite scores")
    if not set(y).issubset({0, 1}):
        raise ValueError("Outer labels must be binary")
    return classification_metrics(y, scores, threshold=frozen.selection.threshold)
