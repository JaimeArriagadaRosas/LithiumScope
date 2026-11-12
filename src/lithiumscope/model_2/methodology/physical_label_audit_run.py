"""Separated run responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.methodology.physical_label_audit import Path

def run(root:Path,out:Path)->dict:
    from lithiumscope.model_2.methodology.physical_label_audit import EXPECTED_TRAINING_MANIFEST_SHA, GEOROC_BIB_REL, GEOROC_SCREEN_REL, MAMANI_REL, MANIFEST_REL, Path, QUEUE_REL, conflicting_site_dossier, digest, georoc_links, historical_links, json, np, patch_quality, pd, sample_contract
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
