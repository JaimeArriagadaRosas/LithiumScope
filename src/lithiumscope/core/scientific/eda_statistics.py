"""Descriptive statistics without learning transformations or selecting features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def source_series(frame: pd.DataFrame) -> pd.Series:
    if "source_dataset" in frame.columns:
        return frame["source_dataset"].astype("string").fillna("unknown")
    if "source_cohort" in frame.columns:
        return frame["source_cohort"].astype("string").fillna("unknown").map(lambda v: f"cohort:{v}")
    return pd.Series("unknown", index=frame.index, dtype="string")


def feature_profiles(frame: pd.DataFrame, columns: list[str], *, stage: str) -> pd.DataFrame:
    records = []
    for source, group in frame.groupby(source_series(frame), sort=True):
        for column in columns:
            values = group[column] if column in group else pd.Series(np.nan, index=group.index)
            numeric = pd.to_numeric(values, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            records.append({
                "stage": stage, "source": source, "feature": column, "rows": len(group),
                "missing": int(values.isna().sum()), "present": int(values.notna().sum()),
                "numeric_finite": len(finite), "numeric_nonfinite": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
                "non_numeric_present": int((values.notna() & numeric.isna()).sum()),
                "zero": int((finite == 0).sum()), "negative": int((finite < 0).sum()),
                "min": finite.min() if len(finite) else None, "q01": finite.quantile(.01) if len(finite) else None,
                "q25": finite.quantile(.25) if len(finite) else None, "median": finite.median() if len(finite) else None,
                "q75": finite.quantile(.75) if len(finite) else None, "q99": finite.quantile(.99) if len(finite) else None,
                "max": finite.max() if len(finite) else None, "mean": finite.mean() if len(finite) else None,
                "std": finite.std(ddof=0) if len(finite) else None,
                "unique_present": int(values.nunique(dropna=True)),
            })
    return pd.DataFrame(records)


def category_frequencies(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records = []
    for source, group in frame.groupby(source_series(frame), sort=True):
        for name in columns:
            if name not in group:
                continue
            for value, count in group[name].astype("string").fillna("<missing>").value_counts().items():
                records.append({"source": source, "feature": name, "value": value, "count": int(count)})
    return pd.DataFrame(records, columns=["source", "feature", "value", "count"])


def numeric_correlations(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.corr(method="spearman", min_periods=3)


def cohort_summary(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    records = []
    for source, group in frame.groupby(source_series(frame), sort=True):
        numeric = pd.to_numeric(group[target], errors="coerce")
        valid = numeric[np.isfinite(numeric)]
        records.append({"source": source, "rows": len(group), "finite_target": len(valid),
                        "target_min": valid.min(), "target_median": valid.median(),
                        "target_q75": valid.quantile(.75), "target_max": valid.max()})
    return pd.DataFrame(records)
