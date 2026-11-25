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
    return result, y.iloc[keep].reset_index(drop=True), groups.iloc[keep].reset_index(drop=True), [
        {"row": str(x.index[i]), "action": "bootstrap_copy"} for i in keep[len(x):]]


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
