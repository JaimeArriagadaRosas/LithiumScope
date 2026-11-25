"""Group-split conformal research protocol. No fitted production model is modified."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_partition(groups, *, seed=42):
    groups = pd.Series(groups).reset_index(drop=True)
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any() or groups.nunique() < 5:
        raise ValueError("At least five valid groups required")
    positions = np.arange(len(groups))
    development, evaluation = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(positions, groups=groups))
    fit, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1).split(
        development, groups=groups.iloc[development]))
    return development[fit], development[calibration], evaluation


def group_calibration_radius(y, prediction, groups, *, alpha=.1):
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    g = pd.Series(groups).reset_index(drop=True)
    if (y.ndim != 1 or y.shape != p.shape or len(g) != len(y) or not len(y)
            or not np.isfinite(np.c_[y,p]).all() or (y < 0).any() or g.isna().any()
            or not 0 < alpha < 1):
        raise ValueError("Invalid calibration evidence")
    maxima = pd.DataFrame({"group": g, "error": np.abs(y-p)}).groupby("group").error.max().to_numpy()
    n = len(maxima)
    rank = math.ceil((n+1)*(1-alpha))
    if rank > n:
        raise ValueError("Too few calibration groups for a finite conformal radius")
    return float(np.sort(maxima)[rank-1]), n


def evaluate_group_intervals(y, prediction, groups, radius):
    y, p = np.asarray(y,float), np.asarray(prediction,float)
    g = pd.Series(groups).reset_index(drop=True)
    if (y.ndim != 1 or y.shape != p.shape or not len(y) or len(g) != len(y)
            or not np.isfinite(np.c_[y,p]).all() or (y < 0).any()
            or g.isna().any() or not np.isfinite(radius) or radius < 0):
        raise ValueError("Invalid held-out evaluation evidence")
    # Truncating only the lower bound at zero preserves coverage for physical y>=0.
    low, high = np.maximum(0., p-radius), p+radius
    valid = high >= low
    covered = valid & (y >= low) & (y <= high)
    table = pd.DataFrame({"group":g, "truth":y, "prediction":p, "lower":low, "upper":high,
                          "interval_valid":valid, "covered":covered})
    by_group = table.groupby("group").covered.all()
    return table, {"row_coverage":float(covered.mean()), "simultaneous_group_coverage":float(by_group.mean()),
                   "evaluation_groups":len(by_group), "rows":len(y), "invalid_intervals":int((~valid).sum()),
                   "mean_valid_width":float((high-low)[valid].mean()) if valid.any() else None}


def run_group_interval_study(x, y, groups, *, fit, predict, seed=42, alpha=.1, execute=False):
    if not x.index.equals(y.index) or not x.index.equals(groups.index) or not x.index.is_unique:
        raise ValueError("Grouped calibration rows must align")
    train, cal, test = group_partition(groups, seed=seed)
    sets = [set(groups.iloc[i]) for i in (train,cal,test)]
    if any(sets[a] & sets[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError("Group overlap in uncertainty protocol")
    if not 0 < alpha < 1:
        raise ValueError("Invalid alpha")
    report = {"state":"preflight", "alpha":alpha, "seed":seed,
              "fit_positions":train.tolist(), "calibration_positions":cal.tolist(),
              "evaluation_positions":test.tolist(), "group_counts":[len(g) for g in sets],
              "external_validation":False, "guarantee":"conditional_on_group_exchangeability",
              "model_saved":False}
    if math.ceil((len(sets[1])+1)*(1-alpha)) > len(sets[1]):
        report.update(state="unavailable", reason="insufficient_calibration_groups")
        return report, None
    if not execute:
        return report, None
    estimator = fit(x.iloc[train], y.iloc[train], groups.iloc[train])
    radius, n = group_calibration_radius(y.iloc[cal], predict(estimator,x.iloc[cal]), groups.iloc[cal], alpha=alpha)
    table, metrics = evaluate_group_intervals(y.iloc[test], predict(estimator,x.iloc[test]), groups.iloc[test], radius)
    table.insert(0,"row_position",test)
    report.update(state="complete", radius_ppm=radius, calibration_groups=n, **metrics)
    return report, table
