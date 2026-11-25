"""Train-only resampling with explicit synthetic parent provenance."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(x, y, groups):
    if (not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(groups.index)
            or not len(x) or groups.isna().any()):
        raise ValueError("Resampling inputs must be aligned TRAIN rows")


def regression_resample(x, y, groups, sources, method, seed):
    _aligned(x, y, groups)
    if not x.index.equals(sources.index):
        raise ValueError("Source identity mismatch")
    rng = np.random.default_rng(seed)
    keep = np.arange(len(x))
    if method == "deduplicate":
        keys = x.assign(__target=y, __group=groups, __source=sources)
        keep = np.flatnonzero(~keys.duplicated())
    elif method in {"rare_bootstrap", "group_bootstrap"}:
        if method == "rare_bootstrap":
            pool = np.flatnonzero(y.to_numpy() >= y.quantile(.90))
            weights = None
        else:
            pool = keep
            weights = (1 / groups.map(groups.value_counts()).to_numpy(float))
            weights /= weights.sum()
        extra = rng.choice(pool, size=max(1, len(x)//4), replace=True, p=weights)
        keep = np.concatenate([keep, extra])
    elif method != "none":
        raise ValueError("Unknown regression resampling")
    # Unique fit row indexes, original group retained, no synthetic measurements.
    result = x.iloc[keep].reset_index(drop=True)
    trace = [{"row": str(x.index[i]), "action": "bootstrap_copy"} for i in keep[len(x):]]
    if method == "deduplicate":
        trace = [{"row": str(x.index[i]), "action": "removed_train_duplicate"} for i in np.setdiff1d(np.arange(len(x)), keep)]
    return result, y.iloc[keep].reset_index(drop=True), groups.iloc[keep].reset_index(drop=True), trace


def classification_resample(x, y, groups, method, seed, ratio=.75):
    """SMOTE in standardized feature space, within the same original TRAIN group.
    
    No invented locations or claims of physically measured satellite spectra.
    """
    _aligned(x, y, groups)
    values = x.to_numpy(float)
    labels = y.to_numpy(int)
    if not np.isfinite(values).all() or set(np.unique(labels)) != {0, 1} or not 0 < ratio <= 1:
        raise ValueError("Invalid binary TRAIN data")
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=2)
    minority = int(np.argmin(counts))
    minor = np.flatnonzero(labels == minority)
    major = np.flatnonzero(labels != minority)
    needed = max(0, int(np.ceil(ratio * len(major))) - len(minor))
    if method == "undersample":
        chosen = rng.choice(major, size=min(len(major), int(np.floor(len(minor)/ratio))), replace=False)
        keep = np.sort(np.concatenate([minor, chosen]))
        return values[keep], labels[keep], []
    if method == "none" or not needed:
        return values, labels, []
    if method == "oversample":
        chosen = rng.choice(minor, size=needed, replace=True)
        return np.vstack([values, values[chosen]]), np.concatenate([labels, labels[chosen]]), [
            {"parent_a": str(x.index[i]), "parent_b": str(x.index[i]), "lambda": 0., "group": str(groups.iloc[i])}
            for i in chosen]
    if method != "smote_group":
        raise ValueError("Unknown classification resampling")
    neighbors = {}
    group_values = groups.to_numpy()
    for i in minor:
        pool = minor[(group_values[minor] == group_values[i]) & (minor != i)]
        if len(pool):
            d = np.sum((values[pool] - values[i])**2, axis=1)
            pool = pool[d > 0]
            if len(pool):
                d = np.sum((values[pool] - values[i])**2, axis=1)
                neighbors[i] = pool[np.argsort(d)[:5]]
    if not neighbors:
        raise ValueError("SMOTE blocked: no distinct same-group minority neighbors")
    synthetic, provenance = [], []
    for i in rng.choice(list(neighbors), size=needed, replace=True):
        j = int(rng.choice(neighbors[i]))
        alpha = float(rng.uniform(0, 1))
        synthetic.append(values[i] + alpha*(values[j]-values[i]))
        provenance.append({"parent_a": str(x.index[i]), "parent_b": str(x.index[j]),
                           "lambda": alpha, "group": str(groups.iloc[i])})
    return np.vstack([values, synthetic]), np.concatenate([labels, np.full(needed, minority)]), provenance

def regression_smoter(x, y, groups, sources, seed, numeric, categorical):
    """Conservative SMOTER-style interpolation of rare TRAIN observations.

    Parents share source, spatial group, exact coordinates, categories and
    measured-feature missingness. Derived ratios are recomputed. No noise,
    artificial extrapolation, invented geological categories or locations.
    """
    from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
    from lithiumscope.model_1.training.preprocessing.tabular import DERIVED, OXIDES
    _aligned(x, y, groups)
    if not x.index.equals(sources.index):
        raise ValueError("Source identity mismatch")
    measured = [name for name in numeric if name not in {*DERIVED, "Latitude", "Longitude"}]
    if not measured or not {"Latitude", "Longitude"} <= set(x.columns):
        raise ValueError("SMOTER requires measured geo chemistry and coordinates")
    values = x[measured].to_numpy(float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("Invalid measured geochemical input")
    coordinates = x[["Latitude","Longitude"]].to_numpy(float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Invalid source coordinates")
    rare = np.flatnonzero(y.to_numpy() >= y.quantile(.90))
    categories = x[categorical].astype("string").fillna("<missing>").to_numpy()
    missing = np.isnan(values)
    scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    neighbors = {}
    for i in rare:
        pool = [j for j in rare if j != i and groups.iloc[j] == groups.iloc[i]
                and sources.iloc[j] == sources.iloc[i] and np.array_equal(coordinates[j],coordinates[i])
                and np.array_equal(categories[j],categories[i]) and np.array_equal(missing[j],missing[i])]
        if not pool:
            continue
        pool = np.asarray(pool)
        distance = np.nansum(((values[pool]-values[i])/scale)**2,axis=1)
        valid = distance > 0
        if valid.any():
            neighbors[i] = pool[valid][np.argsort(distance[valid])[:5]]
    if not neighbors:
        raise ValueError("SMOTER blocked: no compatible distinct rare parents")
    rng = np.random.default_rng(seed)
    rows, targets, fit_groups, trace = [], [], [], []
    for i in rng.choice(list(neighbors), size=max(1,len(x)//4),replace=True):
        j = int(rng.choice(neighbors[i]))
        alpha = float(rng.uniform(0,1))
        row = x.iloc[i].copy()
        row.loc[measured] = values[i] + alpha*(values[j]-values[i])
        # Convex mixing preserves nonnegative measurements and bounded oxide
        # fractions if both original parents are valid. Never fabricate QC.
        oxides = [name for name in OXIDES if name in measured]
        if oxides and (row[oxides].dropna().astype(float) > 100).any():
            continue
        rows.append(row)
        targets.append(float(y.iloc[i] + alpha*(y.iloc[j]-y.iloc[i])))
        fit_groups.append(groups.iloc[i])
        trace.append({"parent_a":str(x.index[i]),"parent_b":str(x.index[j]),
                      "lambda":alpha,"group":str(groups.iloc[i]),"source":str(sources.iloc[i]),
                      "action":"synthetic_interpolation"})
    if not rows:
        raise ValueError("SMOTER blocked: all interpolations failed physical screening")
    generated = add_geochemical_features(pd.DataFrame(rows),version=2).reindex(columns=x.columns)
    return (pd.concat([x,generated],ignore_index=True), pd.concat([y,pd.Series(targets)],ignore_index=True),
            pd.concat([groups,pd.Series(fit_groups)],ignore_index=True),trace)
