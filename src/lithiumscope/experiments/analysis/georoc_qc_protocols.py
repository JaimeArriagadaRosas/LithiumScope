"""Research-only geochemical QC protocols; production default remains protocol A.

Protocol C is *provisional* because missing oxide totals cannot certify analytical
closure. Its extra rows must not be silently treated as verified measurements.
"""
from __future__ import annotations

import pandas as pd

from lithiumscope.datasets.georoc_ingestion_schema import MAJOR_OXIDES


def protocol_masks(
    frame: pd.DataFrame,
    *,
    lower: float = 94.0,
    upper: float = 102.0,
    minimum_majors: int = 8,
) -> pd.DataFrame:
    if not 0 < lower < upper or not 1 <= minimum_majors <= len(MAJOR_OXIDES):
        raise ValueError("Invalid QC parameters.")
    oxide = frame.reindex(columns=MAJOR_OXIDES).apply(
        pd.to_numeric, errors="coerce"
    )
    published = pd.to_numeric(
        frame.get("SUM (no water)", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    total_complete = oxide.sum(axis=1, min_count=len(MAJOR_OXIDES))
    recovered_sum = published.fillna(total_complete)
    existing = published.between(lower, upper)
    calculated = recovered_sum.between(lower, upper)
    count = oxide.notna().sum(axis=1)

    # Protocol C uses the 8-plus oxide analytical coverage as a *hypothesis*,
    # not proof of 94-102 wt% closure. No total is invented for these records.
    core = {"SiO2", "Al2O3", "CaO", "Na2O", "K2O"}
    core_available = oxide[list(sorted(core))].notna().all(axis=1)
    physical = ~(oxide.lt(0) | oxide.gt(100)).any(axis=1)
    partial_sum = oxide.sum(axis=1, min_count=minimum_majors)
    trace_columns = [
        col for col in ("Rb_icpms", "Sr_icpms", "Zr_icpms", "V_icpms",
                        "Ba_icpms", "Nb_icpms", "Th_icpms")
        if col in frame.columns
    ]
    trace_coverage = (
        frame[trace_columns].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
        if trace_columns else pd.Series(0, index=frame.index)
    )
    material = frame.get(
        "Sample_type", pd.Series("", index=frame.index)
    ).astype("string").str.strip().str.upper()
    sources = frame.get(
        "source_dataset", pd.Series("", index=frame.index)
    ).astype("string").str.strip().str.upper()
    li = pd.to_numeric(frame.get("Li_icpms"), errors="coerce")
    lon = pd.to_numeric(
        frame.get("Longitude", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    lat = pd.to_numeric(
        frame.get("Latitude", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    valid = li.notna() & lon.between(-180, 180) & lat.between(-90, 90)
    potential = (
        published.isna() & total_complete.isna()
        & sources.eq("GEOROC") & material.eq("WHOLE ROCK")
        & valid & physical & core_available & (count >= minimum_majors)
        & (trace_coverage >= 2) & partial_sum.between(70, upper)
    )
    result = pd.DataFrame(index=frame.index)
    result["A_original"] = existing
    result["B_reconstructed"] = calculated
    result["C_experimental"] = calculated | potential
    result["C_unverified_extra"] = potential
    result["oxide_count"] = count.astype(int)
    result["measured_total"] = published
    result["fully_reconstructed_total"] = total_complete
    result["partial_oxide_sum"] = partial_sum
    result["qc_status"] = "REJECTED"
    result.loc[existing, "qc_status"] = "A_MEASURED_OR_PRECOMPUTED"
    result.loc[calculated & ~existing, "qc_status"] = "B_COMPLETE_RECONSTRUCTION"
    result.loc[potential, "qc_status"] = "C_UNVERIFIED_EXPERIMENT_ONLY"
    return result


def protocol_summary(frame: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    records = []
    source = frame.get(
        "source_dataset", pd.Series("unknown", index=frame.index)
    ).astype(str)
    for label, idx in source.groupby(source).groups.items():
        for protocol in ("A_original", "B_reconstructed", "C_experimental"):
            chosen = masks.loc[idx, protocol]
            records.append({
                "source": label,
                "protocol": protocol,
                "rows": int(chosen.sum()),
                "total": len(idx),
                "additional_vs_A": int(chosen.sum() - masks.loc[idx, "A_original"].sum()),
                "quality_certified": protocol != "C_experimental",
            })
    return pd.DataFrame(records)
