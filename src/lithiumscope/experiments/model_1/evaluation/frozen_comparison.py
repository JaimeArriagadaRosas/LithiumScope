"""Pair equal dataset versions by verified row identity, not rounded site keys."""
from __future__ import annotations

import numpy as np


def pair_frozen_m1(old, new):
    identity = ["row_index", "row_position", "source_dataset", "Longitude",
                "Latitude", "y_true", "spatial_group", "outer_fold"]
    for frame in (old, new):
        if not set(identity + ["prediction"]) <= set(frame):
            raise ValueError("Missing frozen M1 identity fields")
        if frame[identity].isna().any().any() or frame.row_index.duplicated().any():
            raise ValueError("Null or duplicate frozen M1 row identity")
        if sorted(frame.row_position.tolist()) != list(range(len(frame))):
            raise ValueError("Invalid frozen M1 row positions")
        if frame.groupby("spatial_group").outer_fold.nunique().max() > 1:
            raise ValueError("M1 spatial group spans outer folds")
    a, b = (frame.sort_values("row_index").reset_index(drop=True) for frame in (old,new))
    if len(a) != len(b):
        raise ValueError("Same dataset hash but different M1 cohort size")
    for name in identity:
        if name in {"Longitude", "Latitude", "y_true"}:
            equal = np.allclose(a[name], b[name], rtol=0, atol=1e-8, equal_nan=False)
        else:
            equal = a[name].equals(b[name])
        if not equal:
            raise ValueError("Same dataset hash but different M1 identity: "+name)
    paired = a[identity].copy()
    paired["prediction_old"] = a.prediction
    paired["prediction_new"] = b.prediction
    paired["spatial_group_old"] = a.spatial_group
    paired["spatial_group_new"] = b.spatial_group
    paired["outer_fold_old"] = a.outer_fold
    paired["outer_fold_new"] = b.outer_fold
    return paired, {"old_total":len(a),"new_total":len(b),"matched":len(a),
                    "excluded_old":0,"ambiguous_old":0,"ambiguous_new":0,
                    "same_outer_fold":len(a),"different_outer_fold":0,
                    "protocol":"verified_same_dataset_row_identity_and_spatial_folds",
                    "comparison_limit":"Reused development OOF; not independent holdout."}
