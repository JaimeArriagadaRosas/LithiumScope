"""Independent-use audit: compare proposed holdouts with *all* M2 sources.

Never infer independence solely from different sample IDs or publication names.
Coordinates close to the same location AND matching measured Li indicate a
possible duplicate. Nearby unrelated sites imply spatial dependence, not proof
of duplicate data. This module does not modify either source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _coordinates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    latitude = next(
        (key for key in ("Latitude", "latitude", "Latitude (Y)") if key in frame),
        None,
    )
    longitude = next(
        (key for key in ("Longitude", "longitude", "Longitude (X)") if key in frame),
        None,
    )
    if latitude is None or longitude is None:
        raise ValueError("Missing geographic coordinates for independence audit")
    lat = pd.to_numeric(frame[latitude], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(frame[longitude], errors="coerce").to_numpy(float)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("Nonfinite geographic coordinates; independence unverified")
    if (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
        raise ValueError("Invalid geographic coordinates")
    return lat, lon


def _lithium(frame: pd.DataFrame) -> np.ndarray:
    if "Li_icpms" not in frame:
        raise ValueError("Li_icpms required for possible-duplicate audit")
    value = pd.to_numeric(frame["Li_icpms"], errors="coerce").to_numpy(float)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite Li values; independence unverified")
    return value


def audit_external_overlap(
    external: pd.DataFrame,
    training: pd.DataFrame,
    *,
    near_duplicate_m: float = 100.0,
    nearby_km: float = 50.0,
    lithium_tolerance_ppm: float = 0.005,
) -> dict:
    """Evidence of geographic overlap, possible duplicates, and source leakage.

    Results are not definitive sample matches without documentary review.
    The 'external_independence_verified' flag is always False when there are
    near-duplicate candidates, and is not True without a separate provenance
    and spatial-holdout protocol even if none are found.
    """
    if external.empty or training.empty:
        raise ValueError("An independence audit requires both nonempty datasets")
    if min(near_duplicate_m, nearby_km, lithium_tolerance_ppm) < 0:
        raise ValueError("Invalid negative tolerance")
    lat_e, lon_e = _coordinates(external)
    lat_t, lon_t = _coordinates(training)
    li_e, li_t = _lithium(external), _lithium(training)
    rad_e = np.radians(np.column_stack((lat_e, lon_e)))
    rad_t = np.radians(np.column_stack((lat_t, lon_t)))
    delta_lat = rad_e[:, None, 0] - rad_t[None, :, 0]
    delta_lon = rad_e[:, None, 1] - rad_t[None, :, 1]
    hav = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(rad_e[:, None, 0])
        * np.cos(rad_t[None, :, 0])
        * np.sin(delta_lon / 2) ** 2
    )
    distance_m = 6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    nearest = np.argmin(distance_m, axis=1)
    candidate_matches = []
    near_cases = []
    nearby_cases = []
    rows = []
    ids_e = (
        external["case_id"].astype(str).tolist()
        if "case_id" in external
        else [str(i) for i in range(len(external))]
    )
    ids_t = (
        training["sample_id"].astype(str).tolist()
        if "sample_id" in training
        else [str(i) for i in range(len(training))]
    )
    sources = (
        training["source_dataset"].astype(str).tolist()
        if "source_dataset" in training
        else ["unreported"] * len(training)
    )
    for i, case_id in enumerate(ids_e):
        j = int(nearest[i])
        dist = float(distance_m[i, j])
        # Scan ALL training rows: the geographically nearest one does not
        # necessarily have the most similar lithium concentration.
        duplicate = np.flatnonzero(
            (distance_m[i] <= near_duplicate_m)
            & (np.abs(li_t - li_e[i]) <= lithium_tolerance_ppm)
        )
        if len(duplicate):
            candidate_matches.append(case_id)
        if dist <= near_duplicate_m:
            near_cases.append(case_id)
        if dist <= nearby_km * 1000:
            nearby_cases.append(case_id)
        rows.append({
            "case_id": case_id,
            "nearest_training_id": ids_t[j],
            "nearest_source": sources[j],
            "nearest_distance_m": round(dist, 3),
            "nearest_training_li_ppm": float(li_t[j]),
            "external_li_ppm": float(li_e[i]),
            "possible_duplicate": bool(len(duplicate)),
            "possible_duplicate_ids": [ids_t[k] for k in duplicate],
            "same_site_100m": dist <= near_duplicate_m,
            "nearby_50km": dist <= nearby_km * 1000,
        })
    return {
        "status": (
            "possible_training_overlap_not_independent"
            if candidate_matches else "no_near_duplicate_detected_provenance_unverified"
        ),
        "training_rows_checked": len(training),
        "external_rows_checked": len(external),
        "possible_duplicate_cases": candidate_matches,
        "possible_duplicate_count": len(candidate_matches),
        "same_site_cases": near_cases,
        "nearby_cases": nearby_cases,
        "external_independence_verified": False,
        "rows": rows,
        "limitations": (
            "Geographic and Li similarity is not documentary identity proof; "
            "absence of a match does not certify external independence."
        ),
    }


def audit_manifest_paths(external_path: Path, training_path: Path) -> dict:
    """Read-only CLI-friendly audit. Raises rather than certifying missing data."""
    external = pd.read_csv(external_path)
    training = pd.read_csv(training_path)
    return audit_external_overlap(external, training)
