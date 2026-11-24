"""Quantify physical separation without claiming that grid separation proves independence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6371008.8


def geographic_coordinates(frame: pd.DataFrame) -> np.ndarray:
    names = [next((k for k in candidates if k in frame), None)
             for candidates in (("Latitude", "latitude", "Latitude (Y)"),
                                ("Longitude", "longitude", "Longitude (X)", "Logintude (X)"))]
    if None in names:
        raise ValueError("Missing coordinate columns for spatial audit")
    coords = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(coords).all() or (np.abs(coords) > [90, 180]).any():
        raise ValueError("Invalid coordinates; spatial separation is unverified")
    return np.radians(coords)


def nearest_distance(coords: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(train) or not len(test):
        raise ValueError("Empty train/test partition")
    distances, indices = BallTree(coords[train], metric="haversine").query(coords[test], k=1)
    return distances[:, 0] * EARTH_RADIUS_M, train[indices[:, 0]]


def audit_spatial_folds(frame: pd.DataFrame, groups: pd.Series, splits,
                        *, eligible=None, patch_side_m: float | None = None,
                        minimum_distance_m: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minimum_distance_m < 0 or (patch_side_m is not None and patch_side_m <= 0):
        raise ValueError("Invalid spatial separation policy")
    n = len(frame)
    mask = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    groups = pd.Series(groups).reset_index(drop=True)
    if mask.shape != (n,) or len(groups) != n or groups.isna().any():
        raise ValueError("Spatial audit inputs do not align")
    coords = geographic_coordinates(frame)
    ids = next((frame[c].astype(str).to_numpy() for c in ("sample_id", "source_sample", "Sample") if c in frame),
               frame.index.astype(str).to_numpy())
    seen = np.zeros(n, dtype=int)
    summary, rows = [], []
    for i, (train, test) in enumerate(splits, 1):
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if not len(train) or not len(test) or len(set(train)) != len(train) or len(set(test)) != len(test):
            raise ValueError("Invalid repeated or empty partition")
        if set(train).intersection(test) or set(train).union(test) != set(range(n)):
            raise ValueError("Partition row isolation/coverage failed")
        if set(groups.iloc[train]).intersection(groups.iloc[test]):
            raise ValueError("Spatial group overlap")
        seen[test] += 1
        fit = train[mask[train]]
        all_dist, _ = nearest_distance(coords, train, test)
        distances, nearest = nearest_distance(coords, fit, test)
        record = {"fold": i, "train_rows": len(train), "fit_rows": len(fit), "test_rows": len(test),
                  "train_groups": groups.iloc[train].nunique(), "test_groups": groups.iloc[test].nunique(),
                  "minimum_all_train_m": float(all_dist.min()), "minimum_fit_m": float(distances.min()),
                  "median_fit_m": float(np.median(distances)), "within_100m": int((distances <= 100).sum()),
                  "within_1km": int((distances <= 1000).sum()), "within_10km": int((distances <= 10000).sum()),
                  "minimum_required_m": minimum_distance_m,
                  "distance_policy_pass": bool(np.all(distances >= minimum_distance_m))}
        for column in ("sentinel_scene_id", "source_file", "source_sample", "image_sha256"):
            if column not in frame:
                continue
            known_train = set(frame.iloc[fit][column].dropna().astype(str)) - {""}
            test_values = frame.iloc[test][column].astype("string")
            same = test_values.isin(known_train).fillna(False)
            record[f"{column}_shared_test_rows"] = int(same.sum())
        if patch_side_m is not None:
            # Circumscribed circles are a conservative screen, not proof of polygon intersection.
            record["possible_patch_overlap_test_rows"] = int((distances <= np.sqrt(2) * patch_side_m).sum())
        summary.append(record)
        for j, d, near, full_d in zip(test, distances, nearest, all_dist, strict=True):
            rows.append({"fold": i, "test_position": int(j), "test_id": ids[j],
                         "nearest_fit_position": int(near), "nearest_fit_id": ids[near],
                         "nearest_fit_distance_m": float(d), "nearest_all_train_distance_m": float(full_d)})
    if not np.all(seen == 1):
        raise ValueError("Every row must appear in one outer TEST partition")
    return pd.DataFrame(summary), pd.DataFrame(rows)
