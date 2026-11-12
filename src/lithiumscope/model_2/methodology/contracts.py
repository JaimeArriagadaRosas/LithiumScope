"""Auditable sample/site/patch/scene provenance and label eligibility.

Source metadata is *evidence*, not certification of field correspondence.
The cohort labelled historical is not assumed field-verified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = (
    "sample_id", "image_path", "Li_icpms",
    "latitude", "longitude", "spatial_group",
)
STATUS_VERIFIED = "FIELD_VERIFIED_SURFACE"
STATUS_PROVISIONAL = "PROVISIONAL_SURFACE"
STATUS_HISTORICAL = "HISTORICAL_UNVERIFIED"
STATUS_EXCLUDED = "EXCLUDE_SUBSURFACE_OR_INVALID"


def sample_contract(manifest: pd.DataFrame, *, lithium_cut_ppm: float = 21.6) -> pd.DataFrame:
    """Build a new frame; never mutate the input.

    A rounded coordinate is a *screening site*, NOT a proven field location.
    Mixed labels must be reviewed, not resolved by copying a single label.
    """
    missing = sorted(set(REQUIRED).difference(manifest.columns))
    if missing:
        raise ValueError(f"Missing provenance columns: {missing}")
    if not np.isfinite(lithium_cut_ppm) or lithium_cut_ppm < 0:
        raise ValueError("Lithium screening threshold must be finite and nonnegative")
    if manifest.empty:
        raise ValueError("Empty M2 manifest")
    frame = manifest.copy(deep=True).reset_index(drop=True)
    ids = frame.sample_id.astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Missing or duplicate sample IDs")
    paths = frame.image_path.astype("string").str.strip()
    groups = frame.spatial_group.astype("string").str.strip()
    if paths.isna().any() or paths.eq("").any():
        raise ValueError("Missing image paths")
    if groups.isna().any() or groups.eq("").any():
        raise ValueError("Missing spatial groups")
    lat = pd.to_numeric(frame.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(frame.longitude, errors="coerce").to_numpy(float)
    li = pd.to_numeric(frame.Li_icpms, errors="coerce").to_numpy(float)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()
            and np.isfinite(li).all()):
        raise ValueError("Li and coordinates must be finite")
    if (abs(lat) > 90).any() or (abs(lon) > 180).any() or (li < 0).any():
        raise ValueError("Invalid coordinate or negative Li concentration")
    frame["site_screening_id"] = [
        f"{y:.5f}|{x:.5f}" for y, x in zip(lat, lon)
    ]
    # Patch id is stable for a path, not a geological identity.
    frame["patch_path_id"] = paths.map(
        lambda value: hashlib.sha256(
            str(value).replace("\\", "/").casefold().encode("utf-8")
        ).hexdigest()[:20]
    )
    frame["scene_id"] = (
        frame["sentinel_scene_id"].fillna("").astype(str).str.strip()
        if "sentinel_scene_id" in frame else ""
    )
    frame["publication_id"] = (
        frame["source_citation"].fillna("").astype(str).str.strip()
        if "source_citation" in frame else ""
    )
    verification = (
        frame["coordinate_verification"].fillna("").astype(str).str.strip()
        if "coordinate_verification" in frame else pd.Series("", index=frame.index)
    )
    sources = (
        frame["source_dataset"].fillna("").astype(str).str.strip()
        if "source_dataset" in frame else pd.Series("", index=frame.index)
    )
    status = []
    for label, source in zip(verification, sources):
        upper = label.upper()
        if upper in {"FIELD_VERIFIED_SURFACE", "GPS_AND_SURFACE_VERIFIED"}:
            status.append(STATUS_VERIFIED)
        elif any(word in upper for word in ("EXCLUDE", "SUBSURFACE", "DRILL_CORE", "INVALID")):
            status.append(STATUS_EXCLUDED)
        elif upper in {"PRE_SCREENED_NOT_FIELD_VERIFIED", "PROVISIONAL_SITE_REVIEW"}:
            status.append(STATUS_PROVISIONAL)
        elif source.startswith("GEOROC_B_SURFACE_PROVISIONAL"):
            status.append(STATUS_PROVISIONAL)
        else:
            status.append(STATUS_HISTORICAL)
    frame["label_eligibility"] = status
    frame["high_li_descriptive"] = li >= float(lithium_cut_ppm)
    frame["same_coordinate_count"] = frame.site_screening_id.map(
        frame.site_screening_id.value_counts()
    ).astype(int)
    by_site = frame.groupby("site_screening_id", sort=False).high_li_descriptive.agg(
        ["min", "max"]
    )
    conflicts = set(by_site.index[by_site["min"] != by_site["max"]])
    frame["site_has_conflicting_labels"] = frame.site_screening_id.isin(conflicts)
    return frame


def site_summary(contract: pd.DataFrame) -> pd.DataFrame:
    return contract.groupby("site_screening_id", sort=False).agg(
        n_samples=("sample_id", "size"),
        n_scenes=("scene_id", "nunique"),
        n_publications=("publication_id", "nunique"),
        n_groups=("spatial_group", "nunique"),
        n_high=("high_li_descriptive", "sum"),
        mixed=("site_has_conflicting_labels", "max"),
        n_verified=("label_eligibility", lambda values: int((values == STATUS_VERIFIED).sum())),
        n_provisional=("label_eligibility", lambda values: int((values == STATUS_PROVISIONAL).sum())),
    ).reset_index()


def policy_counts(contract: pd.DataFrame) -> dict:
    counts = contract.label_eligibility.value_counts().to_dict()
    return {
        "sample_count": int(len(contract)),
        "site_count": int(contract.site_screening_id.nunique()),
        "conflicting_sites": int(
            contract.loc[contract.site_has_conflicting_labels, "site_screening_id"].nunique()
        ),
        "conflicting_samples": int(contract.site_has_conflicting_labels.sum()),
        "eligibility": {str(k): int(v) for k, v in counts.items()},
        "verified_surface_samples": int(counts.get(STATUS_VERIFIED, 0)),
        "excluded_samples": int(counts.get(STATUS_EXCLUDED, 0)),
        "historical_without_explicit_field_verification": int(
            counts.get(STATUS_HISTORICAL, 0)
        ),
        "missing_scene_id": int(contract.scene_id.astype(str).str.strip().eq("").sum()),
        "missing_publication_id": int(
            contract.publication_id.astype(str).str.strip().eq("").sum()
        ),
    }
