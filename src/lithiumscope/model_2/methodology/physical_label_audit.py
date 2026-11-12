"""M2 point-to-Sentinel-patch correspondence dossier (opt-in, no model fitting).

Reconciles 706 historical labels to the original Mamani09 table by sanitized
name + *measured Li* + coordinates. Reconciles 77 GEOROC samples to the
source-quality screen and candidate queue. Audits image content, not just paths.

A satellite patch, literature citation, raw coordinate decimals or valid SCL
DO NOT certify that a chemical specimen is surface-exposed at that pixel.
Manual field/source verification is a separate requirement.

Use:
  python -m lithiumscope.model_2.methodology.physical_label_audit \
    --output results/scientific_simulations/m2_phase_d_20261112/physical_label_audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.contracts import sample_contract

EXPECTED_TRAINING_MANIFEST_SHA="764d9ab4d9f548e96f39b9a4bc843c7cb903fabdbbd0a982aa78027cc045e623"
MANIFEST_REL="data/processed/model_2/vnext_expanded/training_manifest.csv"
MAMANI_REL="data/raw/model_1/Mamani09_Table_DR2.csv"
GEOROC_SCREEN_REL="results/scientific_simulations/vnext_20261106/georoc_source_validation/georoc_B_screened_sites.csv"
GEOROC_BIB_REL="results/scientific_simulations/vnext_20261106/georoc_source_validation/GEOROC_B_bibliography.csv"
QUEUE_REL="data/interim/model_2/vnext_georoc/selected_candidate_queue.csv"
IMAGE_BANDS=("red","green","blue","nir","swir16","swir22")

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def safe_sample_id(value) -> str:
    """Use the existing M2 source identity normalization, do not strip accents."""
    if pd.isna(value) or not str(value).strip():
        return ""
    return "".join(x if x.isalnum() or x in "-_" else "_"
                   for x in str(value).strip())

def close_numeric(a,b, *, tol=1e-6)->bool:
    try:
        return bool(np.isfinite(float(a)) and np.isfinite(float(b))
                    and abs(float(a)-float(b))<=tol)
    except (ValueError,TypeError):
        return False

def normalized_text(value)->str:
    return "" if pd.isna(value) else str(value).strip()

def historical_raw_matches(row:pd.Series, candidates:pd.DataFrame)->pd.DataFrame:
    """Cross-check identity, Li AND exact coordinates. No guessed fuzzy join."""
    if candidates.empty:
        return candidates
    valid = candidates.apply(
        lambda x: close_numeric(row.Li_icpms,x.Li_icpms)
                  and close_numeric(row.latitude,x["Latitude (Y)"])
                  and close_numeric(row.longitude,x["Longitude (X)"]),axis=1)
    return candidates.loc[valid]

def historical_links(m:pd.DataFrame, raw:pd.DataFrame)->pd.DataFrame:
    required={"Sample","Li_icpms","Longitude (X)","Latitude (Y)",
              "Geochemistry Reference","Rock_type","Sample_type","Country",
              "Location","Geologycal_age"}
    if not required.issubset(raw):
        raise ValueError(f"Mamani09 columns missing: {required-set(raw)}")
    source=raw.copy()
    source["normalized_id"]=source.Sample.map(safe_sample_id)
    groups={key:g for key,g in source.groupby("normalized_id",sort=False)}
    rows=[]
    for n,item in m.reset_index(drop=True).iterrows():
        subset=groups.get(str(item.sample_id),source.iloc[:0])
        matched=historical_raw_matches(item,subset)
        status=("SOURCE_ROW_NOT_FOUND" if matched.empty
                else "MULTIPLE_RAW_ROWS" if len(matched)>1
                else "SOURCE_ROW_CROSSCHECKED")
        selected=matched.iloc[0] if len(matched)>0 else None
        get=lambda col:normalized_text(selected[col]) if selected is not None else ""
        rows.append({
            "manifest_index":int(n),"sample_id":str(item.sample_id),
            "site_screening_id":str(item.site_screening_id),
            "Li_icpms":float(item.Li_icpms),
            "raw_row_matches":int(len(matched)),
            "source_link_status":status,
            "raw_sample":get("Sample"),
            "geochemistry_reference":get("Geochemistry Reference"),
            "raw_sample_type":get("Sample_type"),
            "raw_rock_type":get("Rock_type"),
            "raw_geological_age":get("Geologycal_age"),
            "raw_location":get("Location"),
            "raw_country":get("Country"),
            "source_row_indices":";".join(str(x) for x in matched.index),
        })
    result=pd.DataFrame(rows)
    if len(result)!=len(m) or result.sample_id.nunique()!=len(m):
        raise AssertionError("Historical linkage lost sample identity")
    return result

from lithiumscope.model_2.methodology.physical_label_audit_georoc_links import georoc_links

def patch_quality(row:pd.Series)->dict:
    """Read saved bands and metadata; do not infer rock exposure from NDVI/SCL."""
    image=Path(str(row.image_path))
    meta=image.with_suffix(".json")
    if not image.is_file() or not meta.is_file():
        return {"image_available":False,"metadata_available":meta.exists(),
                "image_sha256":"","patch_integrity":"MISSING_IMAGE_OR_METADATA",
                "finite_pixel_fraction":None,"ndvi_median_proxy":None,
                "ndvi_gt_0p4_fraction_proxy":None,
                "central_vs_patch_red_median_delta":None,"scl_rock_exposure_certified":False}
    data=json.loads(meta.read_text(encoding="utf-8"))
    checks={
        "meta_sample_id_matches":str(data.get("sample_id"))==str(row.sample_id),
        "meta_scene_matches":str(data.get("scene_id"))==str(row.sentinel_scene_id),
        "meta_coords_match":(close_numeric(data.get("latitude"),row.latitude)
                            and close_numeric(data.get("longitude"),row.longitude)),
        "meta_bands_match":list(data.get("bands",[]))==list(IMAGE_BANDS),
    }
    valid=all(checks.values())
    try:
        arr=np.load(image,allow_pickle=False)
        shape_ok=arr.shape==(6,32,32)
        finite=np.isfinite(arr)
        finite_fraction=float(finite.mean()) if shape_ok else 0
        if shape_ok and finite_fraction>=.95:
            red=arr[0].astype(float)
            nir=arr[3].astype(float)
            den=red+nir
            ndvi=np.divide(nir-red,den,out=np.full_like(red,np.nan),
                           where=np.isfinite(den)&(np.abs(den)>1e-10))
            finite_ndvi=np.isfinite(ndvi)
            ndvi_med=float(np.nanmedian(ndvi)) if finite_ndvi.any() else None
            vegetation=float(np.mean(ndvi[finite_ndvi]>.4)) if finite_ndvi.any() else None
            center=red[13:19,13:19]
            red_delta=float(abs(np.median(center)-np.median(red)))
        else:
            ndvi_med=None;vegetation=None;red_delta=None
    except (ValueError,OSError) as exc:
        raise ValueError(f"Malformed physical patch {image}: {exc}") from exc
    return {
        "image_available":True,"metadata_available":True,
        "image_sha256":digest(image),
        **checks,
        "image_shape_ok":bool(shape_ok),
        "meta_sample_id":str(data.get("sample_id","")),
        "meta_scene_id":str(data.get("scene_id","")),
        "patch_integrity":(
            "BAD_SHAPE" if not shape_ok
            else "BAD_SIDECAR_METADATA" if not valid
            else "INSUFFICIENT_FINITE_PIXELS" if finite_fraction<.95
            else "OK"
        ),
        "finite_pixel_fraction":finite_fraction,
        "ndvi_median_proxy":ndvi_med,
        "ndvi_gt_0p4_fraction_proxy":vegetation,
        "central_vs_patch_red_median_delta":red_delta,
        "scene_cloud_fraction_from_manifest":float(row.sentinel_cloud_cover),
        "scl_cloud_fraction_from_manifest":(
            None if pd.isna(row.scl_cloud_fraction)
            else float(row.scl_cloud_fraction)
        ),
        "scl_rock_exposure_certified":False,
    }

def conflicting_site_dossier(contract:pd.DataFrame, samples:pd.DataFrame)->pd.DataFrame:
    combined=contract[["sample_id","site_screening_id",
         "sentinel_scene_id","site_has_conflicting_labels"]].merge(
        samples,on=["sample_id","site_screening_id"],validate="one_to_one")
    out=[]
    subset=combined.loc[combined.site_has_conflicting_labels]
    for site,group in subset.groupby("site_screening_id"):
        # Confirmation that repeated point coordinates and input pixels collide.
        all_equal_pixels=group.image_sha256.nunique()==1
        n_high=int(sum(group.Li_icpms>=21.6))
        n_low=int(sum(group.Li_icpms<21.6))
        geo=sorted({str(x).strip() for x in group.raw_rock_type if str(x).strip()})
        sampled=sorted({str(x).strip() for x in group.raw_sample_type if str(x).strip()})
        citations=sorted({str(x).strip() for x in group.geochemistry_reference if str(x).strip()})
        out.append({
            "site_screening_id":site,
            "n_samples":len(group),
            "min_Li_ppm":float(group.Li_icpms.min()),
            "max_Li_ppm":float(group.Li_icpms.max()),
            "Li_range_ppm":float(group.Li_icpms.max()-group.Li_icpms.min()),
            "high_count_at_21p6":n_high,"low_count_at_21p6":n_low,
            "distinct_patch_hashes":int(group.image_sha256.nunique()),
            "exactly_same_image_bytes":bool(all_equal_pixels),
            "distinct_scenes":int(group.sentinel_scene_id.nunique()),
            "raw_rock_type_count":len(geo),
            "raw_rock_types":";".join(geo),
            "raw_sample_types":";".join(sampled),
            "raw_geochemistry_references":";".join(citations),
            "sample_ids":";".join(group.sample_id),
            "review_priority":"P0_CONFLICTING_IDENTICAL_SPECTRAL_INPUT" if all_equal_pixels
                             else "P0_CONFLICTING_SPECTRAL_SITE",
            "next_action":"INSPECT_ORIGINAL_SAMPLE_MAP_AND_LITHOLOGY_NO_AUTOMATIC_RELABEL",
            "field_verified":False,
        })
    return pd.DataFrame(out).sort_values(
        ["Li_range_ppm","n_samples"],ascending=False
    ).reset_index(drop=True)


from lithiumscope.model_2.methodology.physical_label_audit_run import run

def scientific_readiness_gate(summary:dict)->dict:
    """Reject only scientific certification; never rewrite or purge samples."""
    reasons=[]
    if summary["historical_conflicting_sites"]:
        reasons.append("IDENTICAL_SPECTRAL_INPUT_WITH_DISCORDANT_LI")
    if summary["patch_integrity_failures"]:
        reasons.append("NONFINITE_PIXEL_COVERAGE_BELOW_95_PERCENT_OR_BAD_METADATA")
    if summary["historical_source_status"].get("MULTIPLE_RAW_ROWS",0):
        reasons.append("AMBIGUOUS_ORIGINAL_RAW_RECORD")
    if summary["georoc_77_coordinate_precision_3_or_fewer_decimals"]:
        reasons.append("NOMINAL_GEOROC_COORDINATE_PRECISION_REQUIRES_REVIEW")
    if summary["samples_with_auto_field_certification"]==0:
        reasons.append("NO_POINT_TO_PATCH_CORRESPONDENCE_FIELD_CERTIFIED")
    return {"scientific_certification_pass":not reasons,
            "rejection_reasons":reasons,
            "training_executed":False,"automatically_excluded":0,
            "verified_surface_labels":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--strict",action="store_true",
        help="Exit 2 when physical-label certification remains unresolved")
    args=parser.parse_args(argv)
    dest=Path(args.output)
    if not dest.is_absolute():dest=PROJECT_ROOT/dest
    result=run(PROJECT_ROOT,dest)
    gate=scientific_readiness_gate(result)
    (dest/"scientific_readiness_gate.json").write_text(
        json.dumps(gate,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print("PHYSICAL_LABEL_GATE",json.dumps(gate,ensure_ascii=False),flush=True)
    return 2 if args.strict and not gate["scientific_certification_pass"] else 0

if __name__=="__main__":
    raise SystemExit(main())
