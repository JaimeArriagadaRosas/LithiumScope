"""Single spatial partition contract for EDA preparation and screening."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def partitions(group, data, cfg):
    x, y, groups, _, eligible, _, _ = data
    if group not in {"model_1", "model_2"}:
        raise ValueError("Unknown model group")
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Spatial groups missing")
    validation = cfg["validation"]
    outer_count = int(validation.get("outer_folds", validation.get("folds", 5)))
    inner_count = int(validation.get("inner_folds", 3))
    if group == "model_2" and cfg.get("training", {}).get("expansion", {}).get("enabled"):
        raise ValueError("Expanded anchored cohorts are not supported by this screening protocol")
    outer = []
    seen = np.zeros(len(x), dtype=int)
    for tr, te in GroupKFold(n_splits=outer_count).split(x, groups=groups):
        seen[te] += 1
        inner = [(tr[a], tr[b]) for a,b in GroupKFold(n_splits=inner_count).split(x.iloc[tr], groups=groups.iloc[tr])]
        for a,b in [(tr,te), *inner]:
            if np.intersect1d(a,b).size or set(groups.iloc[a]) & set(groups.iloc[b]):
                raise ValueError("Spatial overlap in preparation")
            if not eligible[a].any():
                raise ValueError("No QA eligible TRAIN")
        outer.append((tr, te, inner))
    if not np.all(seen == 1):
        raise ValueError("Incomplete outer partition coverage")
    return outer


def training_target(group, data, cfg, train):
    """Compute endpoint only from this partition's original TRAIN."""
    _, y, _, _, eligible, _, _ = data
    fit = train[eligible[train]]
    if group == "model_1":
        return fit, y.iloc[fit], None
    fixed = cfg["model"].get("fixed_threshold_ppm")
    threshold = float(fixed if fixed is not None else y.iloc[train].quantile(cfg["model"]["high_lithium_quantile"]))
    labels = (y.iloc[fit] >= threshold).astype(int)
    if labels.nunique() != 2:
        raise ValueError("Single-class QA eligible TRAIN")
    return fit, labels, threshold
