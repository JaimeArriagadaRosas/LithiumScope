"""Fold-local feature construction, stable ranking and geological imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


RATIOS = (
    ("Rb_icpms", "Sr_icpms"), ("Rb_icpms", "Ba_icpms"),
    ("Th_icpms", "U_icpms"), ("Nb_icpms", "Ta_icpms"),
    ("Zr_icpms", "Hf_icpms"), ("K2O", "Na2O"),
)


def add_ratios(frame):
    result = frame.copy()
    for a, b in RATIOS:
        if a in frame and b in frame:
            left, right = frame[a].astype(float), frame[b].astype(float)
            valid = (left > 0) & (right > 0)
            result[f"log_ratio_{a}_{b}"] = np.log(left.where(valid)) - np.log(right.where(valid))
    return result


def stable_columns(x, y, groups, numeric, limit=18):
    """Rank numeric features by lower-quartile absolute rank correlation across TRAIN group blocks."""
    if groups.nunique() < 3:
        raise ValueError("Stable selection needs three TRAIN groups")
    records = []
    for _, ix in GroupKFold(3).split(x, y, groups):
        target = y.iloc[ix].rank()
        row = {}
        for name in numeric:
            values = x[name].iloc[ix].rank()
            valid = values.notna() & target.notna()
            if valid.sum() >= 8 and values[valid].nunique() > 1 and target[valid].nunique() > 1:
                row[name] = abs(float(values[valid].corr(target[valid])))
            else:
                row[name] = 0.
        records.append(row)
    scores = pd.DataFrame(records).fillna(0.).quantile(.25)
    ranked = sorted(numeric, key=lambda name: (-scores[name], name))
    return ranked[:limit]


class FoldFeatures:
    def __init__(self, arm, numeric, categorical):
        self.arm = arm
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def fit(self, x, y, groups):
        if self.arm.features == "ratios":
            self.numeric = list(add_ratios(x).columns.difference(self.categorical, sort=False))
        elif self.arm.features == "stable":
            self.numeric = stable_columns(x, y, groups, self.numeric)
        self.medians = None
        if self.arm.impute:
            if "Rock_type" not in x:
                raise ValueError("Geology-conditioned imputation requires Rock_type")
            rock = x["Rock_type"].astype("string").fillna("<missing>")
            self.medians = x[self.numeric].groupby(rock).median()
            counts = x[self.numeric].groupby(rock).count()
            self.medians = self.medians.where(counts >= 5)
        return self

    def transform(self, x):
        result = add_ratios(x) if self.arm.features == "ratios" else x.copy()
        if self.medians is not None:
            rock = result["Rock_type"].astype("string").fillna("<missing>")
            for col in self.numeric:
                result[col] = result[col].fillna(rock.map(self.medians[col]))
        return result[self.numeric + self.categorical]
