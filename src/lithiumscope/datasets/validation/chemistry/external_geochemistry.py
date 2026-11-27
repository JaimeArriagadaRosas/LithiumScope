"""Evidence-gated catalog for independent geochemistry, without fabricating observations."""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED = ("sample_id","material","latitude","longitude","Li_value",
            "Li_unit","analytical_method","source","sample_date","independent_of_training")
COMPATIBLE_MATERIAL = {"rock", "whole_rock"}


def validate_geochemistry(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [col for col in REQUIRED if col not in frame]
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))
    table = frame.copy()
    numeric = pd.to_numeric(table["Li_value"], errors="coerce")
    lat = pd.to_numeric(table["latitude"], errors="coerce")
    lon = pd.to_numeric(table["longitude"], errors="coerce")
    independent = table["independent_of_training"].astype(str).str.lower().isin(["true","yes","1"])
    compatible = (
        table["material"].astype(str).str.lower().isin(COMPATIBLE_MATERIAL)
        & table["Li_unit"].astype(str).str.lower().isin(["ppm", "mg/kg"])
        & numeric.ge(0) & np.isfinite(numeric)
        & lat.between(-90,90) & lon.between(-180,180)
        & independent
        & table["source"].notna() & table["analytical_method"].notna()
        & table["sample_id"].notna()
        & table["source"].astype(str).str.strip().ne("")
        & table["analytical_method"].astype(str).str.strip().ne("")
        & table["sample_id"].astype(str).str.strip().ne("")
    )
    table["evidence_status"] = compatible.map({True:"CANDIDATE_FOR_REVIEW",False:"NOT_ELIGIBLE"})
    return table.loc[compatible].copy(), table.loc[~compatible].copy()



def review_independence(candidates: pd.DataFrame, reference: pd.DataFrame, *,
                        spatial_buffer_km: float) -> pd.DataFrame:
    """Screen explicit reference overlap; absence of matches is not certification.

    Both tables use source/sample IDs, coordinates, and optional publication IDs.
    Publication tokens must already be resolved to the same identifier system.
    """
    if not np.isfinite(spatial_buffer_km) or spatial_buffer_km < 0:
        raise ValueError("Invalid spatial buffer")
    required = {"source", "sample_id", "latitude", "longitude"}
    for frame in (candidates, reference):
        if not required <= set(frame) or not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Independence review requires aligned identity and coordinate fields")
    if reference.empty:
        raise ValueError("Reference cohort is empty")
    def normalized(values):
        return values.astype("string").fillna("").str.strip().str.casefold()
    def coordinates(frame):
        lat, lon = (pd.to_numeric(frame[k], errors="coerce") for k in ("latitude", "longitude"))
        return lat, lon, lat.between(-90,90) & lon.between(-180,180)
    rs, ri = normalized(reference.source), normalized(reference.sample_id)
    reference_ids = set(zip(rs[rs.ne("") & ri.ne("")], ri[rs.ne("") & ri.ne("")]))
    cs, ci = normalized(candidates.source), normalized(candidates.sample_id)
    rl, rn, rv = coordinates(reference)
    cl, cn, cv = coordinates(candidates)
    ref_lat, ref_lon = np.radians(rl[rv].to_numpy(float)), np.radians(rn[rv].to_numpy(float))
    publications = set(normalized(reference["publication_id"])) - {""} if "publication_id" in reference else set()
    candidate_publications = normalized(candidates.get("publication_id", pd.Series("", index=candidates.index)))
    rows = []
    for pos, (source, sample, publication) in enumerate(zip(cs, ci, candidate_publications)):
        reasons = []
        if not source or not sample:
            reasons.append("missing_identity")
        if (source, sample) in reference_ids:
            reasons.append("known_sample_overlap")
        if publication and publication in publications:
            reasons.append("known_publication_overlap")
        if not publication or not publications:
            reasons.append("publication_review_unavailable")
        distance = None
        if cv.iloc[pos] and len(ref_lat):
            lat, lon = np.radians([cl.iloc[pos], cn.iloc[pos]])
            h = np.sin((ref_lat - lat) / 2)**2 + np.cos(lat)*np.cos(ref_lat)*np.sin((ref_lon - lon)/2)**2
            distance = float(np.min(6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))))
            if distance <= spatial_buffer_km:
                reasons.append("spatial_proximity_review")
        else:
            reasons.append("spatial_review_unavailable")
        if not rv.all():
            reasons.append("incomplete_reference_coordinates")
        if not rs.ne("").all() or not ri.ne("").all():
            reasons.append("incomplete_reference_identity")
        if "sample_date" in candidates:
            if pd.isna(pd.to_datetime(candidates.sample_date.iloc[pos], errors="coerce")):
                reasons.append("sample_date_unverified")
        else:
            reasons.append("sample_date_unverified")
        state = "known_overlap" if any(r.startswith("known_") for r in reasons) else ("review_required" if reasons else "no_overlap_detected")
        rows.append({"state": state, "reasons": reasons, "nearest_reference_km": distance,
                     "independence_certified": False})
    return pd.DataFrame(rows, index=candidates.index, columns=["state", "reasons", "nearest_reference_km", "independence_certified"])
