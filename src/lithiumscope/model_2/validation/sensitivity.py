"""Predeclared outer spatial sensitivity protocols for M2, not model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree

from lithiumscope.core.scientific.spatial_audit import EARTH_RADIUS_M, geographic_coordinates

# Historical grid reference, finer/coarser alternatives, and conservative
# patch separation (640 m square diagonal < 1 km). No TEST-driven choice.
POLICIES = (
    ("grid_050", .5, 0., False),
    ("grid_025", .25, 0., False),
    ("grid_100", 1., 0., False),
    ("grid_050_buffer_1km", .5, 1000., False),
    ("grid_050_buffer_5km", .5, 5000., False),
    ("grid_050_scene_buffer_1km", .5, 1000., True),
)


def spatial_components(manifest, grid, *, combine_scene=False):
    coords = geographic_coordinates(manifest)
    if not np.isfinite(grid) or grid <= 0:
        raise ValueError("Grid must be positive")
    cells = np.floor(np.degrees(coords) / grid).astype(int)
    parent = list(range(len(cells)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def connect(keys):
        seen = {}
        for i, key in enumerate(keys):
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i

    connect(map(tuple, cells))
    if combine_scene:
        col = next((c for c in ("sentinel_scene_id", "scene_id") if c in manifest), None)
        if col is None or manifest[col].isna().any() or manifest[col].astype(str).str.strip().eq("").any():
            raise ValueError("Complete scene identities required for scene-separated protocol")
        connect(manifest[col].astype(str).str.strip())
    return pd.Series([f"component_{root(i)}" for i in range(len(cells))], index=manifest.index)


def plan_protocol(manifest, lithium, *, grid, buffer_m, combine_scene, folds, endpoint_ppm, quality_mask=None):
    """Keep every TEST row; embargo only TRAIN. Fixed external endpoint for comparability."""
    coords = geographic_coordinates(manifest)
    li = np.asarray(lithium, dtype=float)
    keep = np.ones(len(li), dtype=bool) if quality_mask is None else np.asarray(quality_mask)
    if (li.shape != (len(manifest),) or not np.isfinite(li).all() or (li < 0).any()
            or keep.shape != li.shape or keep.dtype.kind != "b" or not keep.any()
            or not np.isfinite(endpoint_ppm) or endpoint_ppm < 0
            or not np.isfinite(buffer_m) or buffer_m < 0 or folds < 2):
        raise ValueError("Invalid spatial sensitivity inputs")
    groups = spatial_components(manifest, grid, combine_scene=combine_scene)
    if groups.nunique() < folds:
        raise ValueError("Too few independent components for requested folds")
    labels = (li >= endpoint_ppm).astype(int)
    records = []
    for number, (outer_train, test) in enumerate(
            GroupKFold(n_splits=folds).split(li, groups=groups), 1):
        eligible = outer_train[keep[outer_train]]
        if len(eligible):
            d, _ = BallTree(coords[test], metric="haversine").query(coords[eligible], k=1)
            train = eligible[d[:, 0] * EARTH_RADIUS_M >= buffer_m]
        else:
            train = eligible
        nearest = None
        if len(train):
            d, _ = BallTree(coords[train], metric="haversine").query(coords[test], k=1)
            nearest = float(d.min() * EARTH_RADIUS_M)
        ready = (groups.iloc[train].nunique() >= 2
                 and np.unique(labels[train]).size == np.unique(labels[test]).size == 2)
        records.append({
            "fold": number, "train": train.tolist(), "test": test.tolist(),
            "n_outer_train": len(outer_train), "n_quality_train": len(eligible),
            "n_fit": len(train), "n_test": len(test), "n_embargoed": len(eligible) - len(train),
            "n_fit_groups": int(groups.iloc[train].nunique()),
            "minimum_fit_test_m": nearest, "state": "ready" if ready else "unavailable",
        })
    return groups, records
