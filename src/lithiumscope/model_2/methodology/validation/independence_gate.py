"""Fail-closed audit of independence between training and external samples.

Coordinate/source overlap is reported, never used to tune candidates or
silently remove or rewrite historical training/validation rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def _coordinates(data: pd.DataFrame) -> np.ndarray:
    lat = "latitude" if "latitude" in data else "Latitude"
    lon = "longitude" if "longitude" in data else "Longitude"
    if lat not in data or lon not in data:
        raise ValueError("Missing external/train latitude or longitude")
    coords = data[[lat, lon]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite geographical coordinates")
    if (np.abs(coords[:, 0]) > 90).any() or (np.abs(coords[:, 1]) > 180).any():
        raise ValueError("Coordinates outside geographical limits")
    return np.deg2rad(coords)


def pairwise_distances_km(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    a, b = _coordinates(train), _coordinates(test)
    dy = b[:, None, 0] - a[None, :, 0]
    dx = b[:, None, 1] - a[None, :, 1]
    h = np.sin(dy / 2) ** 2 + (
        np.cos(b[:, None, 0]) * np.cos(a[None, :, 0]) * np.sin(dx / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def audit_external_overlap(
    train: pd.DataFrame,
    external: pd.DataFrame,
    *,
    near_meters: float = 250.0,
    li_tolerance_ppm: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    if len(train) == 0 or len(external) == 0:
        raise ValueError("Training and external sample tables must not be empty")
    if "Li_icpms" not in train or "Li_icpms" not in external:
        raise ValueError("Geochemical Li values required for source overlap audit")
    distances = pairwise_distances_km(train, external)
    nearest = np.argmin(distances, axis=1)
    train_li = train.Li_icpms.to_numpy(dtype=float)
    external_li = external.Li_icpms.to_numpy(dtype=float)
    if not (np.isfinite(train_li).all() and np.isfinite(external_li).all()):
        raise ValueError("Nonfinite Li values")
    records = []
    for index, nearest_idx in enumerate(nearest):
        distance = float(distances[index, nearest_idx] * 1000)
        difference = float(abs(external_li[index] - train_li[nearest_idx]))
        near = distance <= near_meters
        matched = near and difference <= li_tolerance_ppm
        records.append({
            "external_case_id": str(external.iloc[index].get("case_id", index)),
            "train_sample_id": str(train.iloc[nearest_idx].get("sample_id", nearest_idx)),
            "distance_m": distance,
            "external_li_ppm": float(external_li[index]),
            "train_li_ppm": float(train_li[nearest_idx]),
            "equal_li_within_tolerance": difference <= li_tolerance_ppm,
            "near_train_location": near,
            "probable_duplicate": matched,
            "train_source": str(train.iloc[nearest_idx].get("source_dataset", "")),
        })
    audit = pd.DataFrame(records)
    n_positive = int((external_li >= 21.55).sum())
    duplicate_count = int(audit.probable_duplicate.sum())
    # Conservative release-design guardrails, not universal statistical constants.
    approved = (
        duplicate_count == 0 and len(external) >= 50 and n_positive >= 10
        and int(audit.near_train_location.sum()) == 0
    )
    result = {
        "external_cases": len(external),
        "external_positive_at_21p55ppm": n_positive,
        "within_250m": int(audit.near_train_location.sum()),
        "probable_duplicate_samples": duplicate_count,
        "within_50km": int((audit.distance_m < 50000).sum()),
        "min_distance_m": float(audit.distance_m.min()),
        "evaluation_independence_approved": approved,
        "external_used_for_model_selection": False,
        "release_gate_policy": "fail_closed_require_50_cases_10_positive_no_250m_matches",
        "note": "Proximity and equal concentration suggest, but cannot certify, sample identity",
    }
    return audit, result


def release_gate(
    independence: dict,
    *,
    independent_roc_auc: float | None = None,
    independent_average_precision: float | None = None,
    reference_roc_auc: float | None = None,
    reference_average_precision: float | None = None,
    confirmatory_test_locked: bool = False,
) -> dict:
    issues: list[str] = []
    if not independence["evaluation_independence_approved"]:
        issues.append("external_holdout_not_independent_or_underpowered")
    if not confirmatory_test_locked:
        issues.append("confirmatory_external_holdout_not_frozen_before_selection")
    if None in (independent_roc_auc, independent_average_precision,
                reference_roc_auc, reference_average_precision):
        issues.append("missing_paired_independent_reference_metrics")
    elif independent_roc_auc < reference_roc_auc or (
        independent_average_precision < reference_average_precision
    ):
        issues.append("candidate_not_better_than_reference_on_external_discrimination")
    return {
        "approved": not issues,
        "eligible_for_release": not issues,
        "reasons": issues,
        "selection_uses_external_holdout": False,
        "individual_model_release_allowed": False,
    }
