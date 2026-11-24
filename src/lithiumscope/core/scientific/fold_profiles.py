"""Descriptive inspection of planned outer TRAIN subsets; no fitted transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fold_profiles(x: pd.DataFrame, y: pd.Series, groups: pd.Series,
                  splits: list[tuple[np.ndarray, np.ndarray]],
                  eligibility: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = np.ones(len(x), dtype=bool) if eligibility is None else np.asarray(eligibility, dtype=bool)
    if mask.shape != (len(x),) or len(y) != len(x) or len(groups) != len(x):
        raise ValueError("Fold profile inputs do not align")
    stats, features = [], []
    for fold, (train, test) in enumerate(splits, 1):
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if set(train).intersection(test) or set(groups.iloc[train]).intersection(groups.iloc[test]):
            raise ValueError("Fold profile contains train/test overlap")
        fit = train[mask[train]]
        subset = x.iloc[fit]
        target = y.iloc[fit]
        stats.append({"fold": fold, "train_rows": len(train), "fit_eligible_rows": len(fit),
                      "test_rows_reserved": len(test), "train_groups": groups.iloc[train].nunique(),
                      "test_groups": groups.iloc[test].nunique(),
                      "train_li_min": target.min(), "train_li_median": target.median(),
                      "train_li_max": target.max()})
        for name in x:
            values = subset[name]
            features.append({"fold": fold, "feature": name, "train_eligible_rows": len(fit),
                             "train_missing": int(values.isna().sum()),
                             "train_unique_present": int(values.nunique(dropna=True)),
                             "train_all_missing": bool(values.isna().all())})
    return pd.DataFrame(stats), pd.DataFrame(features)
