"""Geodesic isolation audits and opt-in buffered training indices.

No labels are used to form or alter the test partitions. A scene shared
between folds is a dependence warning, not proof of direct label leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class FoldIsolation:
    fold: int
    n_train: int
    n_test: int
    min_distance_m: float
    n_test_near_train: int
    n_test_shared_scene: int
    n_test_shared_publication: int
    n_train_removed_buffer_preview: int
    n_train_removed_scene_preview: int


def _indices(record: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        train = np.asarray(record["train_indices"], dtype=int)
        test = np.asarray(record["test_indices"], dtype=int)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid fold index manifest") from exc
    if train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test):
        raise ValueError("Fold train/test must be nonempty 1D index arrays")
    if (
        (train < 0).any() or (test < 0).any() or
        (train >= n).any() or (test >= n).any() or
        len(set(train)) != len(train) or len(set(test)) != len(test)
    ):
        raise ValueError("Fold index out of range or repeated")
    if set(train).intersection(test):
        raise ValueError("Fold train/test indices overlap")
    if set(train).union(test) != set(range(n)):
        raise ValueError("Fold must partition the entire training dataset")
    return train, test


def _positions(contract: pd.DataFrame) -> np.ndarray:
    return np.radians(
        contract[["latitude", "longitude"]].to_numpy(dtype=float)
    )


def _shared_flags(
    contract: pd.DataFrame, train: np.ndarray, test: np.ndarray, column: str
) -> np.ndarray:
    a = contract[column].iloc[train].astype(str)
    b = contract[column].iloc[test].astype(str)
    nonblank = set(a) - {""}
    return b.isin(nonblank).to_numpy(dtype=bool)


def _distances(positions, train, test):
    tree = BallTree(positions[train], metric="haversine")
    distances, indices = tree.query(positions[test], k=1)
    return distances[:, 0] * EARTH_RADIUS_M, train[indices[:, 0]]


def assess_folds(
    contract: pd.DataFrame, folds: Sequence[dict], *,
    warning_distance_m: float = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assess frozen splits; raise on direct group/site overlap.

    Distances and shared scene/publication are OBSERVATIONS. No automatic
    claim of target leakage is made. Test partitions are never rewritten.
    """
    if not np.isfinite(warning_distance_m) or warning_distance_m < 0:
        raise ValueError("Distance warning must be nonnegative and finite")
    n = len(contract)
    locations = _positions(contract)
    tested = np.zeros(n, dtype=int)
    per_test = []
    summary = []
    for k, record in enumerate(folds, start=1):
        train, test = _indices(record, n)
        tested[test] += 1
        groups = contract.spatial_group.astype(str)
        if set(groups.iloc[train]).intersection(groups.iloc[test]):
            raise ValueError(f"Fold {k} mixes training and test spatial groups")
        sites = contract.site_screening_id.astype(str)
        if set(sites.iloc[train]).intersection(sites.iloc[test]):
            raise ValueError(f"Fold {k} shares screening sites between train and test")
        distance, nearest_train = _distances(locations, train, test)
        scene_flags = _shared_flags(contract, train, test, "scene_id")
        citation_flags = _shared_flags(contract, train, test, "publication_id")
        min_distance = float(distance.min())
        for j, idx in enumerate(test):
            per_test.append({
                "fold": k, "test_index": int(idx),
                "sample_id": str(contract.sample_id.iloc[idx]),
                "nearest_train_sample_id": str(contract.sample_id.iloc[nearest_train[j]]),
                "nearest_train_m": float(distance[j]),
                "too_close_for_warning_buffer": bool(distance[j] < warning_distance_m),
                "same_scene_in_train": bool(scene_flags[j]),
                "same_publication_in_train": bool(citation_flags[j]),
                "label_eligibility": str(contract.label_eligibility.iloc[idx]),
            })
        summary.append({
            "fold": k, "n_train": len(train), "n_test": len(test),
            "min_distance_m": min_distance,
            "n_test_too_close": int(sum(distance < warning_distance_m)),
            "n_test_scene_shared": int(scene_flags.sum()),
            "n_test_publication_shared": int(citation_flags.sum()),
            "warning_buffer_m": float(warning_distance_m),
        })
    if not np.all(tested == 1):
        raise ValueError("Each sample must appear in exactly one outer test fold")
    return pd.DataFrame(summary), pd.DataFrame(per_test)


def buffered_train_preview(
    contract: pd.DataFrame, folds: Sequence[dict], *, buffer_m: float,
    exclude_shared_scenes: bool = False,
) -> pd.DataFrame:
    """Return a train-only exclusion preview, NEVER change test indices.

    Does NOT estimate predictive performance and does not fit/calibrate.
    The next training phase must recompute fold labels from buffered TRAIN Li.
    """
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Buffer must be finite and nonnegative")
    locations = _positions(contract)
    rows = []
    for k, record in enumerate(folds, 1):
        tr, te = _indices(record, len(contract))
        if buffer_m > 0:
            tree = BallTree(locations[te], metric="haversine")
            in_buffer = tree.query_radius(
                locations[tr], r=buffer_m / EARTH_RADIUS_M, count_only=True
            ) > 0
        else:
            in_buffer = np.zeros(len(tr), dtype=bool)
        scene = (
            _shared_flags(contract, te, tr, "scene_id")
            if exclude_shared_scenes else np.zeros(len(tr), dtype=bool)
        )
        keep = ~(in_buffer | scene)
        rows.append({
            "fold": k, "n_train_original": len(tr),
            "n_test_fixed": len(te),
            "n_removed_buffer": int(in_buffer.sum()),
            "n_removed_scene_only": int((scene & ~in_buffer).sum()),
            "n_train_remaining": int(keep.sum()),
            "removed_fraction": float((~keep).mean()),
            "train_indices_remaining": tuple(int(x) for x in tr[keep]),
            "test_indices_unchanged": tuple(int(x) for x in te),
        })
    return pd.DataFrame(rows)
