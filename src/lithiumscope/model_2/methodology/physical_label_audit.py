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

def georoc_links(man:pd.DataFrame, screen:pd.DataFrame,
                 bibliography:pd.DataFrame,queue:pd.DataFrame)->pd.DataFrame:
    required={"sample_id","CITATIONS","source_sample","Latitude","Longitude",
              "Li_icpms","raw_rock_class","raw_comment","decimal_lat",
              "decimal_lon","references_complete","screening_tier",
              "geologic_label_certified","source_dois","raw_matches"}
    if not required.issubset(screen):
        raise ValueError(f"GEOROC screen columns missing: {required-set(screen)}")
    if screen.sample_id.duplicated().any() or queue.sample_id.duplicated().any():
        raise ValueError("Ambiguous GEOROC source IDs")
    if not {"ref_id","matched","citation_text","doi"}.issubset(bibliography):
        raise ValueError("Bibliography is missing required metadata")
    lookup=screen.set_index("sample_id")
    queue_ids=set(queue.sample_id.astype(str))
    references=bibliography.set_index(bibliography.ref_id.astype(str))
    rows=[]
    for i,item in man.reset_index(drop=True).iterrows():
        if str(item.sample_id) not in lookup.index:
            raise ValueError(f"GEOROC source-screen match missing: {item.sample_id}")
        src=lookup.loc[str(item.sample_id)]
        coords=close_numeric(item.latitude,src.Latitude) and close_numeric(
            item.longitude,src.Longitude)
        lithium=close_numeric(item.Li_icpms,src.Li_icpms, tol=1e-5)
        citation=str(item.source_citation)==str(src.CITATIONS)
        if not (coords and lithium and citation):
            raise ValueError(f"GEOROC source values disagree for {item.sample_id}")
        if str(item.sample_id) not in queue_ids:
            raise ValueError(f"GEOROC selected queue provenance missing: {item.sample_id}")
        ref_ids=__import__("re").findall(r"\[(\d+)\]",str(src.CITATIONS))
        matched_ref=bool(ref_ids) and all(r in references.index and
                   str(references.loc[r,"matched"]).strip().lower()=="true"
                   for r in ref_ids)
        if not matched_ref:
            raise ValueError(f"GEOROC bibliography match missing: {item.sample_id}")
        if str(src.screening_tier)!="PROVISIONAL_SITE_REVIEW":
            raise ValueError(f"GEOROC screening status not provisional: {item.sample_id}")
        if str(src.geologic_label_certified).strip().lower()=="true":
            raise ValueError("Unexpected certified flag: external field evidence required")
        rock=normalized_text(src.raw_rock_class)
        comment=normalized_text(src.raw_comment)
        decimal_min=int(min(int(src.decimal_lat),int(src.decimal_lon)))
        rows.append({
            "manifest_index":706+int(i),"sample_id":str(item.sample_id),
            "Li_icpms":float(item.Li_icpms),
            "latitude":float(item.latitude),"longitude":float(item.longitude),
            "screening_tier":str(src.screening_tier),
            "source_sample":normalized_text(src.source_sample),
            "citation_ids":str(src.CITATIONS),
            "source_dois":normalized_text(src.source_dois),
            "rock_class_source":rock,
            "raw_location":normalized_text(src.LOCATION),
            "raw_comment":comment,
            "minimum_raw_coordinate_decimals":decimal_min,
            "nominal_coordinate_precision_low":bool(decimal_min<=3),
            "rock_description_missing":bool("NOT GIVEN" in rock.upper() or not rock),
            "raw_comment_missing":not bool(comment),
            "reference_present_in_official_snapshot":bool(matched_ref),
            "source_identity_unique_in_raw":bool(int(src.raw_matches)==1),
            "field_gps_verified":False,
            "exposure_at_patch_verified":False,
            "sample_to_patch_certified":False,
            "source_traceability":"PROVISIONAL_SOURCE_AND_BIBLIOGRAPHY_CROSSCHECKED",
        })
    return pd.DataFrame(rows)

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


def run(root:Path,out:Path)->dict:
    inputs={
        "manifest":root/MANIFEST_REL,
        "mamani_source":root/MAMANI_REL,
        "georoc_source_screen":root/GEOROC_SCREEN_REL,
        "georoc_bibliography":root/GEOROC_BIB_REL,
        "georoc_candidate_queue":root/QUEUE_REL,
    }
    for name,path in inputs.items():
        if not path.is_file(): raise FileNotFoundError(f"{name}: {path}")
    hashes={name:digest(path) for name,path in inputs.items()}
    if hashes["manifest"]!=EXPECTED_TRAINING_MANIFEST_SHA:
        raise RuntimeError("Experiment manifest changed; refuse to inspect unknown training dataset")
    manifest=pd.read_csv(inputs["manifest"])
    contract=sample_contract(manifest)
    if len(contract)!=783 or len(contract.iloc[:706])!=706:
        raise ValueError("Expected immutable 706+77 experimental sample order")
    if not contract.source_dataset.iloc[:706].eq("MAMANI09_HISTORICAL").all():
        raise ValueError("Unexpected historical cohort")
    if not contract.source_dataset.iloc[706:].eq("GEOROC_B_SURFACE_PROVISIONAL").all():
        raise ValueError("Unexpected GEOROC cohort")
    raw=pd.read_csv(inputs["mamani_source"],encoding="cp1252",low_memory=False)
    old=historical_links(contract.iloc[:706].copy(),raw)
    screen=pd.read_csv(inputs["georoc_source_screen"])
    bibliography=pd.read_csv(inputs["georoc_bibliography"])
    queue=pd.read_csv(inputs["georoc_candidate_queue"])
    georoc=georoc_links(contract.iloc[706:].copy(),screen,bibliography,queue)
    quality=[]
    for i,item in contract.iterrows():
        if i%125==0: print("PATCH",i,"/783",flush=True)
        metrics=patch_quality(item)
        quality.append({"manifest_index":i,"sample_id":str(item.sample_id),**metrics})
    quality=pd.DataFrame(quality)
    # Flag mismatches explicitly; do not silently discard data or rewrite
    # original imagery/metadata. This stage is an audit, not an auto-fixer.
    mixed=old.merge(quality[["sample_id","image_sha256"]],
                    on="sample_id",validate="one_to_one")
    conflict=conflicting_site_dossier(contract.iloc[:706],mixed)
    if len(conflict)!=25:
        raise RuntimeError(f"Expected 25 historical mixed-Li sites, got {len(conflict)}")
    if int(conflict.n_samples.sum())!=77:
        raise RuntimeError("Original 77 conflicting historical samples changed")
    # Dataset has no verified labels, even if source record is cross-checked.
    # These categories describe review workload, not physical certainties.
    georoc=georoc.merge(
        quality[["sample_id","image_sha256","patch_integrity",
                 "ndvi_median_proxy","ndvi_gt_0p4_fraction_proxy",
                 "scl_cloud_fraction_from_manifest"]],
        on="sample_id",validate="one_to_one",
    )
    georoc["review_priority"]=np.where(
        georoc.patch_integrity.ne("OK"),
        "P0_PATCH_METADATA_INCONSISTENCY",
        np.where(georoc.nominal_coordinate_precision_low | georoc.rock_description_missing,
                 "P1_SOURCE_AND_LOCATION_EVIDENCE",
                 "P2_FIELD_EXPOSURE_AND_ALIGNMENT"),
    )
    georoc["scientific_admission"]="NOT_FIELD_VERIFIED_NOT_APPROVED"
    georoc["review_steps"]=(
        "ORIGINAL_PAPER_SUPPLEMENT_GPS; SAMPLE_MATERIAL_SURFACE_OR_DRILL; "+
        "GEOLOGIC_OUTCROP_MAP; PATCH_PIXEL_EXPOSURE_CHECK; INDEPENDENT_REVIEW"
    )
    old["historical_citation_recovered"]=old.geochemistry_reference.ne("")
    old["geologic_field_evidence"]="NOT_CERTIFIED_FROM_CSV"
    old["scientific_admission"]="LEGACY_ONLY_NO_NEW_CERTIFICATION"
    conflict_ids=set(conflict.site_screening_id)
    broken=set(quality.loc[quality.patch_integrity.ne("OK"),"sample_id"])
    old["manual_priority"]=np.where(old.site_screening_id.isin(conflict_ids),
                                    "P0_CONFLICTING_IDENTICAL_SPECTRAL_INPUT",
                                    np.where(old.sample_id.isin(broken),
                                             "P0_INCOMPLETE_PIXEL_COVERAGE",
                                    np.where(old.source_link_status.ne("SOURCE_ROW_CROSSCHECKED"),
                                             "P1_AMBIGUOUS_RAW_SOURCE_RECORD",
                                             "P3_HISTORICAL_GEOLOGIC_TRACEABILITY")))
    review=contract[["sample_id","Li_icpms","latitude","longitude","scene_id",
                     "site_screening_id","label_eligibility"]].merge(
        quality[["sample_id","image_sha256","patch_integrity",
                 "ndvi_median_proxy","ndvi_gt_0p4_fraction_proxy"]],
        on="sample_id",validate="one_to_one"
    )
    priority_map=dict(zip(old.sample_id,old.manual_priority))
    priority_map.update(zip(georoc.sample_id,georoc.review_priority))
    review["review_priority"]=review.sample_id.map(priority_map)
    review["field_certified"]=False
    review["review_action"]=np.select(
        [
            review.review_priority.str.startswith("P0_CONFLICTING"),
            review.review_priority.str.startswith(("P0_PATCH","P0_INCOMPLETE")),
            review.review_priority.str.startswith("P1"),
        ],
        [
            "RECONCILE_SAMPLE_LI_ROCK_TYPE_SAME_SATELLITE_INPUT",
            "INVESTIGATE_MISSING_PIXEL_COVERAGE_PRESERVE_ORIGINAL",
            "VERIFY_SOURCE_OR_GPS_WITH_ORIGINAL_PUBLICATION",
        ],
        default="VERIFY_FIELD_SURFACE_AND_IMAGE_ALIGNMENT",
    )
    manual_target=review.loc[review.review_priority.str.startswith(("P0","P1","P2"))].copy()
    manual_target["reviewer"]=""
    manual_target["source_pdf_or_field_map_uri"]=""
    manual_target["gps_uncertainty_m_documented"]=""
    manual_target["surface_outcrop_evidence_uri"]=""
    manual_target["sample_patch_correspondence_evidence_uri"]=""
    manual_target["review_decision"]=""
    manual_target["review_notes"]=""
    # Keep model input and raw sources strictly unchanged: only write output.
    out.mkdir(parents=True,exist_ok=True)
    old.to_csv(out/"historical_706_source_reconciliation.csv",index=False)
    georoc.to_csv(out/"georoc_77_scientific_review.csv",index=False)
    quality.to_csv(out/"patch_image_integrity_783.csv",index=False)
    mixed.to_csv(out/"historical_source_patch_join.csv",index=False)
    conflict.to_csv(out/"conflicting_25_sites_dossier.csv",index=False)
    review.to_csv(out/"full_sample_review_queue.csv",index=False)
    manual_target.to_csv(out/"manual_review_evidence_template.csv",index=False)
    source_counts={
        "historical_source_status":old.source_link_status.value_counts().to_dict(),
        "historical_raw_reference_recovered":int(old.historical_citation_recovered.sum()),
        "historical_rock_description_present":int(old.raw_rock_type.ne("").sum()),
        "historical_sample_type_present":int(old.raw_sample_type.ne("").sum()),
        "georoc_77_citation_snapshot_matched":int(georoc.reference_present_in_official_snapshot.sum()),
        "georoc_77_missing_rock_description":int(georoc.rock_description_missing.sum()),
        "georoc_77_missing_comment":int(georoc.raw_comment_missing.sum()),
        "georoc_77_coordinate_precision_3_or_fewer_decimals":int(georoc.nominal_coordinate_precision_low.sum()),
        "georoc_77_p1_reviews":int(georoc.review_priority.str.startswith("P1").sum()),
        "georoc_77_p2_reviews":int(georoc.review_priority.str.startswith("P2").sum()),
        "historical_conflicting_sites":len(conflict),
        "historical_conflicting_rows":int(conflict.n_samples.sum()),
        "historical_conflicts_with_identical_patch_bytes":int(conflict.exactly_same_image_bytes.sum()),
        "historical_conflicting_sites_with_multiple_raw_rock_types":int(sum(conflict.raw_rock_type_count>1)),
        "historical_conflicting_sites_with_multiple_raw_sample_types":int(
            sum(conflict.raw_sample_types.map(lambda s:len(set(x for x in s.split(";") if x)))>1)),
        "distinct_patches_by_sha256":int(quality.image_sha256.nunique()),
        "patch_metadata_and_arrays_ok":int(quality.patch_integrity.eq("OK").sum()),
        "patch_integrity_failures":int(quality.patch_integrity.ne("OK").sum()),
        "patch_issue_categories":quality.patch_integrity.value_counts().to_dict(),
        "historical_patches_below_95pct_finite":int(
            quality.iloc[:706].patch_integrity.eq("INSUFFICIENT_FINITE_PIXELS").sum()),
        "georoc_patches_below_95pct_finite":int(
            quality.iloc[706:].patch_integrity.eq("INSUFFICIENT_FINITE_PIXELS").sum()),
        "patch_failed_sample_ids":quality.loc[
            quality.patch_integrity.ne("OK"),"sample_id"].tolist(),
        "patch_failed_reason_counts":{column:int((~quality[column]).sum())
             for column in ("meta_sample_id_matches","meta_scene_matches",
                             "meta_coords_match","meta_bands_match","image_shape_ok")},
        "samples_with_auto_field_certification":0,
        "sources_hashes_sha256":hashes,
        "scientific_limitations":[
            "Mamani09 source-table match is NOT proof of sample representativeness.",
            "GEOROC official bibliography and nominal coordinate precision are not field GPS verification.",
            "SCL cloud fraction does NOT establish bare-rock surface in a pixel.",
            "NDVI derived from six reflectance bands is only a vegetation proxy, not a rock exposure classifier.",
            "An identical image with discordant Li cannot support a single deterministic sample label without a reviewed aggregation or revised objective.",
            "Nothing in this audit changes production labels, frozen folds, official models, or their calibration.",
        ]
    }
    (out/"scientific_quality_summary.json").write_text(
        json.dumps(source_counts,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print("QUALITY_COMPLETE",json.dumps({k:v for k,v in source_counts.items()
         if not isinstance(v,(list,dict))},ensure_ascii=False),flush=True)
    return source_counts

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
