from __future__ import annotations

import numpy as np
import pandas as pd


class ScientificValidationError(ValueError):
    pass


def assert_no_target_leakage(target: str, feature_columns: list[str]) -> None:
    normalized_target = target.strip().lower()
    normalized_features = {column.strip().lower() for column in feature_columns}
    if normalized_target in normalized_features:
        raise ScientificValidationError(
            f"Target leakage detected: {target!r} is present in predictor columns."
        )


def assert_oof_complete(predictions, expected_length: int) -> None:
    values = np.asarray(predictions, dtype=float)
    if values.shape != (expected_length,):
        raise ScientificValidationError(
            f"OOF prediction length mismatch: {values.shape} != ({expected_length},)."
        )
    if not np.isfinite(values).all():
        missing = int((~np.isfinite(values)).sum())
        raise ScientificValidationError(
            f"OOF predictions are incomplete/non-finite: {missing} invalid values."
        )


def assert_group_isolation(
    train_indices,
    test_indices,
    groups: pd.Series,
) -> None:
    train_groups = set(groups.iloc[train_indices].astype(str))
    test_groups = set(groups.iloc[test_indices].astype(str))
    overlap = train_groups.intersection(test_groups)
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ScientificValidationError(
            f"Spatial leakage detected: train/test share groups: {examples}"
        )

def assert_partition_coverage(splits, count: int) -> None:
    """Every observation is tested exactly once; each split is a full partition."""
    coverage = np.zeros(count, dtype=int)
    if not splits or count < 1:
        raise ScientificValidationError("No outer evaluation partition")
    for train, test in splits:
        train, test = np.asarray(train), np.asarray(test)
        for indices in (train, test):
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or not len(indices) or (indices < 0).any() or (indices >= count).any()
                    or len(np.unique(indices)) != len(indices)):
                raise ScientificValidationError("Invalid or duplicated outer partition indices")
        if np.intersect1d(train, test).size or len(train) + len(test) != count:
            raise ScientificValidationError("Outer TRAIN/TEST partition is incomplete or overlaps")
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise ScientificValidationError("Outer TEST coverage must be exactly once per observation")
