"""Build a one-change M1 lithology ablation input; preserve every historical row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.preparation.source_reconciliation import (
    reconcile_georoc, reported_rock_label, oxide_balance,
)

EXPECTED_HISTORICAL_SHA="94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94"
EXPECTED_GEOROC_SHA="dd95db8983337611393a3e19698ade39c6825565ae32f876f3a8801e0e9c3f16"


def run(dataset, source, output):
    if file_sha256(dataset)!=EXPECTED_HISTORICAL_SHA or file_sha256(source)!=EXPECTED_GEOROC_SHA:
        raise ValueError("Frozen historical data or reconciled GEOROC input changed")
    output=Path(output)
    if output.exists():
        raise FileExistsError("Use a new audit directory")
    prepared=prepare_training_data(dataset)
    historical=prepared.raw_frame
    georoc=pd.read_csv(source,dtype={"source_unique_id":str})
    selected=historical.source_dataset.eq("GEOROC")
    linkage=reconcile_georoc(historical.loc[selected],georoc)
    linkage["reported_rock_label"]=linkage.source_rock_class.map(reported_rock_label)
    linkage["historical_rock_type"]=historical.loc[linkage.source_row,"Rock_type"].to_numpy()
    linkage["eligible_in_original_training"]=linkage.source_row.isin(prepared.frame.index)
    linkage["restore"]=linkage.state.eq("unique") & linkage.reported_rock_label.notna() & linkage.historical_rock_type.isna()
    balance=oxide_balance(historical.loc[prepared.frame.index])
    balance.insert(0,"source_dataset",historical.loc[prepared.frame.index,"source_dataset"])
    balance.insert(0,"source_row",balance.index)
    output.mkdir(parents=True,exist_ok=False)
    # Keep all raw values as text, modifying only the one feature for a matched source.
    original=pd.read_csv(dataset,dtype=str,keep_default_na=False)
    candidate=original.copy()
    restore=linkage.loc[linkage.restore]
    candidate.loc[restore.source_row,"Rock_type"]=restore.reported_rock_label.to_numpy()
    if not original.drop(columns="Rock_type").equals(candidate.drop(columns="Rock_type")):
        raise ValueError("Ablation changed columns other than Rock_type")
    candidate_path=output/"training_combined_lithology_candidate.csv"
    candidate.to_csv(candidate_path,index=False,encoding="utf-8-sig")
    recovered=prepare_training_data(candidate_path)
    if not (prepared.frame.index.equals(recovered.frame.index)
            and prepared.y.equals(recovered.y) and prepared.groups.equals(recovered.groups)
            and prepared.x.drop(columns="Rock_type").equals(recovered.x.drop(columns="Rock_type"))
            and prepared.schema==recovered.schema):
        raise ValueError("Lithology ablation changed cohort, group, target or other predictors")
    changed=prepared.x.Rock_type.ne(recovered.x.Rock_type)
    duplicate=prepared.x.assign(_target=prepared.y).duplicated(keep=False)
    duplicates=historical.loc[prepared.x.index[duplicate],
                              ["source_dataset","source_sample","Sample","Li_icpms","Longitude","Latitude"]].copy()
    duplicates.insert(0,"source_row",duplicates.index)
    # Do not infer duplicate physical sample from identical feature vector.
    duplicates["disposition"]="retained_pending_documentary_sample_identity_review"
    linkage.to_csv(output/"chemical_source_linkage.csv",index=False)
    balance.to_csv(output/"oxide_balance.csv",index=False)
    duplicates.to_csv(output/"exact_predictor_target_repetitions.csv",index=False)
    pd.DataFrame({"source_row":prepared.x.index,
                  "historical_rock_category":prepared.x.Rock_type.to_numpy(),
                  "candidate_rock_category":recovered.x.Rock_type.to_numpy(),
                  "source_dataset":prepared.frame.source_dataset.to_numpy(),
                  "changed":changed.to_numpy()}).to_csv(output/"prepared_category_comparison.csv",index=False)
    balance_summary={}
    for name,group in balance.groupby("source_dataset"):
        balance_summary[name]={"rows":len(group),"feo_reported":int(group.reported_feo.notna().sum()),
            "sum10_agrees":int(group.matches_10_within_0_05.sum()),
            "sum10_plus_feo_agrees":int(group.matches_10_plus_feo_within_0_05.sum())}
    final=linkage.loc[linkage.eligible_in_original_training]
    report={"scope":"Retrospective provenance and one-feature input ablation, no estimator fitting",
            "historical_sha256":EXPECTED_HISTORICAL_SHA,"georoc_sha256":EXPECTED_GEOROC_SHA,
            "historical_raw_rows":len(historical),"historical_prepared_rows":len(prepared.y),
            "raw_georoc_match_states":linkage.state.value_counts().to_dict(),
            "prepared_georoc_match_states":final.state.value_counts().to_dict(),
            "raw_lithology_restored":len(restore),
            "prepared_lithology_restored":int(final.restore.sum()),
            "prepared_category_changed":int(changed.sum()),
            "prepared_georoc_without_unambiguous_reported_lithology":int((~final.restore).sum()),
            "raw_and_prepared_other_fields_unchanged":True,
            "groups_unchanged":True,"labels_unchanged":True,"predictor_count":len(prepared.x.columns),
            "exact_predictor_target_repeated_rows":len(duplicates),"rows_deleted":0,
            "oxide_balance":balance_summary,"qc_tolerance_for_comparison":.05,
            "analytical_method_equivalence_certified":False,"candidate_trained":False,
            "artifacts":{p.name:file_sha256(p) for p in output.glob("*.csv")}}
    if file_sha256(dataset)!=EXPECTED_HISTORICAL_SHA:
        raise ValueError("Historical input changed")
    (output/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","source","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.source,args.output),indent=2,ensure_ascii=False))
