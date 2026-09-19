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
    if len(values) != expected_length:
        raise ScientificValidationError(
            f"OOF prediction length mismatch: {len(values)} != {expected_length}."
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
