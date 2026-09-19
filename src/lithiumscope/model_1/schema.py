from __future__ import annotations

import pandas as pd

from lithiumscope.core.scientific_checks import assert_no_target_leakage


def validate_training_schema(target: str, numeric: list[str], categorical: list[str]) -> None:
    columns = numeric + categorical
    if len(columns) != len(set(columns)):
        raise ValueError("Feature schema contains duplicated predictor columns.")
    assert_no_target_leakage(target, columns)


def applicability_profile(frame: pd.DataFrame, numeric_columns: list[str]) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        profile[column] = {
            "q01": float(values.quantile(0.01)),
            "q99": float(values.quantile(0.99)),
        }
    return profile


def applicability_fraction(frame: pd.DataFrame, profile: dict[str, dict[str, float]]) -> pd.Series:
    if not profile:
        return pd.Series(0.0, index=frame.index)
    flags = []
    for column, bounds in profile.items():
        if column not in frame.columns:
            flags.append(pd.Series(True, index=frame.index))
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        flags.append(
            values.isna()
            | (values < float(bounds["q01"]))
            | (values > float(bounds["q99"]))
        )
    return pd.concat(flags, axis=1).mean(axis=1)
