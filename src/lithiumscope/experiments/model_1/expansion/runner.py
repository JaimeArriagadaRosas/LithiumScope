"""Advanced EDA and experimental data plan using only cached M1 sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.experiments.data_lab.candidate_screening import text
from lithiumscope.experiments.data_lab.candidate_audit import FOLD_SHA256
from lithiumscope.experiments.model_1.expansion.inputs import load_inputs
from lithiumscope.experiments.model_1.expansion.identity import resolve, describe_domain
from lithiumscope.experiments.model_1.expansion.cohorts import (
    GOOD_STATES, REVIEW_COLUMNS, review_candidates, reserve_holdout, fold_membership)
from lithiumscope.model_1.pipeline import prepare_training_data, prepare_prediction_data
from lithiumscope.model_1.validation.spatial import build_spatial_groups
from lithiumscope.results.atomic_exports import write_json, write_table, write_legacy_evidence_json


def run(*, review_path=None, output_root=None):
    source,base,folds,candidates,inputs,inventory=load_inputs()
    source_digest=file_sha256(source)
    prepared=prepare_training_data(source)
    x=prepared.x
    groups=prepared.groups
    if (len(x)!=3420 or len(x.columns)!=33 or groups is None
            or not np.array_equal(folds.row_position,np.arange(len(x)))
            or not np.array_equal(folds.row_index.to_numpy(str),x.index.to_numpy(str))
            or not np.array_equal(folds.spatial_group.to_numpy(str),groups.to_numpy(str))
            or not np.array_equal(folds.source_dataset.to_numpy(str),source_series(prepared.frame).to_numpy(str))
            or not np.allclose(folds[["Longitude","Latitude"]],x[["Longitude","Latitude"]],rtol=0,atol=1e-8)):
        raise ValueError("Frozen M1 prepared cohort or folds changed")
    audit=resolve(candidates,base,groups)
    audit["raw_source_sha256"]=candidates.raw_source_sha256
    verified={(r["source"],r["file"]) for r in inventory if r["raw_state"]=="verified"}
    audit["raw_verified"]=[(a,b) in verified for a,b in zip(audit.source_dataset,audit.source_file)]
    audit["source_li_method"]=candidates.source_li_method
    audit["source_coordinate_crs"]=candidates.source_coordinate_crs
    # Deterministic production feature engineering, not fitting/imputation.
    features=prepare_prediction_data(candidates,"common",prepared.schema)
    if not features.index.equals(audit.index) or list(features.columns)!=list(x.columns):
        raise ValueError("Expansion predictor schema changed")
    # Verify canonical group semantics on valid coordinates, including candidates.
    valid=features[["Longitude","Latitude"]].notna().all(axis=1)
    audit["prepared_coordinates_valid"]=valid
    computed=build_spatial_groups(features.loc[valid],degrees=.5)
    if not np.array_equal(computed.to_numpy(str),audit.loc[valid,"spatial_group"].to_numpy(str)):
        raise ValueError("Candidate spatial groups differ from production")
    audit=audit.join(describe_domain(features,x,prepared.schema.numeric))
    audit["lithium_band"]=pd.cut(audit.Li_icpms,[-np.inf,5,20,50,100,696,1000,np.inf]).astype(str)
    review=None
    review_digest=None
    if review_path is not None:
        review_path=Path(review_path)
        review_digest=file_sha256(review_path)
        review=pd.read_csv(review_path,dtype=str,keep_default_na=False)
        if file_sha256(review_path)!=review_digest:
            raise ValueError("Review ledger changed during read")
    audit=review_candidates(audit,review)
    publications=text(prepared.frame,"source_publication")
    audit=reserve_holdout(audit,set(groups),set(publications)-{""})
    membership=fold_membership(audit,features,x,folds,publications,prepared.schema.numeric)
    summary=(audit.groupby(["source_dataset","screening_state","review_decision","candidate_partition"],dropna=False)
             .agg(rows=("candidate_id","size"),groups=("spatial_group","nunique"),
                  eligible=("experimental_eligible","sum")).reset_index())
    coverage=(audit.groupby(["source_dataset","screening_state","lithium_band"],observed=True)
              .agg(rows=("candidate_id","size"),groups=("spatial_group","nunique"),
                   median_predictors=("predictors_present","median"),
                   median_marginal_ood=("marginal_ood_fraction","median")).reset_index())
    proposed=audit.loc[audit.screening_state.isin(GOOD_STATES)].copy()
    requests=(proposed.groupby(["source_dataset","source_publication","screening_state"],dropna=False)
              .agg(rows=("candidate_id","size"),method_known=("method_known","sum"),
                   crs_known=("crs_known","sum"),li_max=("Li_icpms","max")).reset_index()
              .sort_values(["li_max","rows"],ascending=False))
    ledger=proposed[["candidate_id","raw_source_sha256","source_dataset","source_file",
                    "source_row_position","source_publication","source_sample","Li_icpms",
                    "screening_state","source_li_method","source_coordinate_crs"]].copy()
    for key in REVIEW_COLUMNS[2:]:
        ledger[key]=""
    feature_export=pd.concat([audit[["candidate_id","Li_icpms","spatial_group","source_dataset",
                                    "source_publication"]],features],axis=1)
    counts=membership.groupby(["fold","cohort"]).candidate_train.sum().reset_index(name="candidate_train_rows")
    root=Path(output_root) if output_root is not None else RESULTS_DIR/"model_1/expansion"
    destination=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    outputs={"row_audit.csv":audit,"candidate_features.csv":feature_export,
             "summary.csv":summary,"coverage.csv":coverage,"publication_review.csv":requests,
             "review_template.csv":ledger,"fold_membership.csv":membership,"fold_counts.csv":counts}
    for name,frame in outputs.items():
        write_table(destination/name,frame)
    write_legacy_evidence_json(destination/"raw_metadata_inventory.json",inventory)
    if file_sha256(source)!=source_digest:
        raise ValueError("Frozen source changed")
    manifest={"state":"review_required" if not audit.experimental_eligible.any() else "experimental_plan_ready",
        "scope":"M1_expansion_preparation_not_training","source_sha256":source_digest,
        "historical_rows":len(x),"historical_groups":int(groups.nunique()),"historical_folds":5,
        "historical_test_preserved":True,"candidate_rows":len(audit),
        "historical_fold_sha256":FOLD_SHA256,
        "exact_candidate_copies":int(audit.exact_copy.sum()),
        "experimental_eligible":int(audit.experimental_eligible.sum()),
        "production_approved":0,"reserved_development_rows":int(audit.candidate_partition.eq("reserved_development_check").sum()),
        "historical_publication_known_rows":int(publications.ne("").sum()),
        "features":list(x.columns),"numeric":list(prepared.schema.numeric),"categorical":list(prepared.schema.categorical),
        "review_sha256":review_digest,"inputs":inputs,
        "training":False,"acquisition":False,"promotion":False,"independent_external_holdout":False,
        "outputs":{name:file_sha256(destination/name) for name in [*outputs,"raw_metadata_inventory.json"]}}
    write_json(destination/"manifest.json",manifest)
    print(summary.to_string(index=False))
    print(counts.to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {destination}")
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review",type=Path)
    args=parser.parse_args()
    run(review_path=args.review)


if __name__=="__main__":
    main()
