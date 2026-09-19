from __future__ import annotations

import pandas as pd


def feature_range_profile(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        profile[str(column)] = {
            "q01": float(values.quantile(0.01)),
            "q99": float(values.quantile(0.99)),
        }
    return profile


def out_of_range_fraction(frame: pd.DataFrame, profile: dict[str, dict[str, float]]) -> float:
    if not profile:
        return 0.0
    flags = 0
    total = 0
    for column, bounds in profile.items():
        total += 1
        if column not in frame.columns:
            flags += 1
            continue
        value = pd.to_numeric(frame[column], errors="coerce").iloc[0]
        if pd.isna(value) or value < bounds["q01"] or value > bounds["q99"]:
            flags += 1
    return flags / total if total else 0.0
