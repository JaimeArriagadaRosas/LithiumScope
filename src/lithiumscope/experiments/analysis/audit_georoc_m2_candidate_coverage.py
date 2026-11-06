"""Read-only M2 suitability check for new geochemical points from GEOROC B.

Coordinates must be chosen per row and verified for plausible geolocation
uncertainty before matching to a Sentinel-2 patch. No scene downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks
from lithiumscope.model_1.steps.step_09_validation import build_spatial_groups
from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples


DEFAULT_OUT = (
    RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage4_evidence"
)


def coalesced_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Use old Mamani09 and newer GEOROC coordinate fields on a row-by-row basis."""
    def column(name: str) -> pd.Series:
        if name not in frame:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.DataFrame({
        "longitude":column("Longitude (X)").combine_first(column("Longitude")),
        "latitude":column("Latitude (Y)").combine_first(column("Latitude")),
    },index=frame.index)


def coordinate_half_extent_m(frame: pd.DataFrame) -> pd.Series:
    """Half bounding-box diagonal in metres; geometric uncertainty proxy ONLY.

    Bounds may describe location ranges rather than measured positional error.
    A zero-width bounding box does not prove precise georeferencing.
    """
    columns=("Latitude_min","Latitude_max","Longitude_min","Longitude_max")
    if not all(c in frame.columns for c in columns):
        return pd.Series(np.nan,index=frame.index,dtype=float)
    vals=frame[list(columns)].apply(pd.to_numeric,errors="coerce")
    lat0,lat1=vals.Latitude_min,vals.Latitude_max
    lon0,lon1=vals.Longitude_min,vals.Longitude_max
    valid=lat0.notna()&lat1.notna()&lon0.notna()&lon1.notna()
    valid&=(lat1>=lat0)&(lon1>=lon0)
    avg_lat=(lat1+lat0)/2
    lat_m=(lat1-lat0)*111_320
    lon_m=(lon1-lon0)*111_320*np.cos(np.deg2rad(avg_lat))
    radius=.5*np.hypot(lat_m,lon_m)
    return radius.where(valid)


def run_candidate_audit(
    root: Path=PROJECT_ROOT, out: Path=DEFAULT_OUT,
) -> dict:
    b_file=root/"results/scientific_simulations/vnext_20261106/stage2_m1/feot_combined_experiment_only.csv"
    old_file=root/"data/processed/model_1/training_combined.csv"
    manifest_file=root/"data/processed/model_2/training_manifest.csv"
    frame=pd.read_csv(b_file,low_memory=False)
    flags=protocol_masks(frame)
    b=frame.loc[flags.B_reconstructed].copy()
    coords=coalesced_coordinates(b)
    lithium=pd.to_numeric(b.Li_icpms,errors="coerce")
    ready=(
        coords.longitude.between(-180,180)
        &coords.latitude.between(-90,90)&lithium.notna()
    )
    if not ready.all():
        raise ValueError("A B-certified record lacks Li or valid coordinates")
    # Do not turn missing original GEOROC sample names into synthetic IDs.
    georoc=b.source_dataset.astype(str).eq("GEOROC")
    original_id=b.get("Sample",pd.Series(np.nan,index=b.index)).astype("string")
    source_id=b.get("source_sample",pd.Series(np.nan,index=b.index)).astype("string")
    chosen_ids=original_id.fillna("").str.strip()
    chosen_ids=chosen_ids.mask(chosen_ids.eq(""),source_id.fillna("").str.strip())
    if chosen_ids.eq("").any():
        raise ValueError("Missing traceable geochemical sample identifiers")

    original=load_georeferenced_li_samples(old_file)
    manifest=pd.read_csv(manifest_file)
    if len(original)!=len(manifest):
        raise ValueError("Historical M2 manifest is not aligned with source loader")
    groups=build_spatial_groups(b)
    original_groups=set(manifest.spatial_group.astype(str))
    new_group_count=len(set(groups)-original_groups)
    current_method_valid=(
        pd.to_numeric(b.get("Longitude (X)"),errors="coerce").between(-180,180)
        &pd.to_numeric(b.get("Latitude (Y)"),errors="coerce").between(-90,90)
        &lithium.notna()
    )
    geo=b.loc[georoc]
    extent=coordinate_half_extent_m(geo)
    known=extent.notna()
    q=extent.dropna().quantile([0,.25,.5,.75,.9,.95,.99,1.])
    # M2 uses 640m square patches: 320m from center to side boundary.
    precise_proxy=known&extent.le(320)
    coarse_proxy=known&extent.gt(320)
    li_current_q75=float(manifest.Li_icpms.quantile(.75))
    li_new_q75=float(lithium.quantile(.75))
    target_b=(lithium>=li_current_q75)
    f_geo=pd.DataFrame({
        "sample_id":chosen_ids.loc[georoc].to_numpy(),
        "Longitude":coords.loc[georoc,"longitude"].to_numpy(),
        "Latitude":coords.loc[georoc,"latitude"].to_numpy(),
        "extent_half_diagonal_m":extent.to_numpy(),
        "within_320m_geometric_proxy":precise_proxy.to_numpy(),
        "Li_icpms":lithium.loc[georoc].to_numpy(),
        "spatial_group":groups.loc[georoc].to_numpy(),
    })
    out.mkdir(parents=True,exist_ok=True)
    # Write only the counts by group to keep raw analytical sample IDs local.
    f_geo.groupby("spatial_group",as_index=False).agg(
        n=("Li_icpms","size"),
        median_li=("Li_icpms","median"),
        positions_with_bounds=("extent_half_diagonal_m","count"),
        positions_with_radius_at_most_320m=("within_320m_geometric_proxy","sum"),
    ).sort_values("n",ascending=False).to_csv(
        out/"m2_georoc_candidate_groups.csv",index=False
    )

    findings={
        "current_m2_total":len(manifest),
        "current_m2_geo_groups":len(original_groups),
        "current_source_fingerprint_matches_m1_data":(
            __import__("hashlib").sha256(old_file.read_bytes()).hexdigest()
            ==(manifest_file.with_suffix(manifest_file.suffix+".source.sha256")
               .read_text(encoding="utf8").strip())
        ),
        "reference_loader_rows":len(original),
        "B_candidate_total_georeferenced":len(b),
        "B_candidate_georoc":int(georoc.sum()),
        "B_candidate_mamani":int((~georoc).sum()),
        "B_candidate_groups":int(groups.nunique()),
        "B_groups_missing_from_current_m2":int(new_group_count),
        "B_rows_current_column_selector_can_read":int(current_method_valid.sum()),
        "B_rows_current_column_selector_discards":int((~current_method_valid).sum()),
        "B_candidate_ids_present":int(chosen_ids.ne("").sum()),
        "current_m2_label_q75_for_descriptive_comparison":li_current_q75,
        "B_all_lithium_q75_descriptive_only":li_new_q75,
        "B_fraction_high_at_current_threshold":float(target_b.mean()),
        "current_m2_fraction_high_at_current_threshold":float(
            (manifest.Li_icpms>=li_current_q75).mean()
        ),
        "GEOROC_extent_with_min_max_bounds":int(known.sum()),
        "GEOROC_extent_without_both_min_max":int((~known).sum()),
        "GEOROC_half_extent_quantiles_metres":{str(k):float(v) for k,v in q.items()},
        "GEOROC_extent_radius_at_most_320m_proxy":int(precise_proxy.sum()),
        "GEOROC_extent_radius_greater_320m_proxy":int(coarse_proxy.sum()),
        "GEOROC_coordinate_precision_caution":"Extent is a source bounding-box proxy. A zero-width extent DOES NOT certify the published coordinate accuracy or sampling site.",
        "imagery_status":"No Sentinel-2 scenes fetched for candidate GEOROC points in this audit",
        "extra_M2_sample_count_is_only_potential":"Requires valid site provenance, source metadata, new Sentinel patches, domain-shift tests, common target definition.",
        "source_selector_issue":"M2 sample_source currently selects Longitude (X)/Latitude (Y) if columns exist, even when NULL for GEOROC rows; it ignores per-row Longitude/Latitude.",
        "sample_id_issue":"M2 sample_source prefers Sample and would generate synthetic sample_N IDs for GEOROC unless coalescing source_sample per row.",
    }
    (out/"m2_georoc_candidate_suitability.json").write_text(
        json.dumps(findings,ensure_ascii=False,indent=2),encoding="utf8"
    )
    print("GEOROC_M2_SUITABILITY",json.dumps(findings,ensure_ascii=False),flush=True)
    return findings


if __name__=="__main__":
    run_candidate_audit()
