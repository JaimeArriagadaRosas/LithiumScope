"""Trace the 3420-candidate's largest historical errors to GEOROC provenance.

Retrospective only. No fitting, no source edits, no exclusion of high lithium.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.scientific_checks import assert_group_isolation, assert_partition_coverage
from lithiumscope.model_1.pipeline import prepare_training_data

SOURCE_SHA = "94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94"
OOF_SHA = "a526bfa1985b0de34c3e4bf7dbbd7e4ca428f16630416b6d57fdd13a7c0e8b06"
KEYS = ["_sample", "_li", "_lon", "_lat"]
PROVENANCE = ["source_unique_id", "source_publication", "source_rock_class",
              "source_location", "source_tectonic_setting", "source_li_raw",
              "source_li_method", "source_li_unit", "source_feot_raw", "source_fe2o3_raw"]


def keys(frame):
    result = frame.copy()
    result["_sample"] = result.source_sample.fillna("").astype(str).str.strip()
    for name, target in (("Li_icpms", "_li"), ("Longitude", "_lon"), ("Latitude", "_lat")):
        result[target] = pd.to_numeric(result[name], errors="coerce").round(8)
    return result


def reconcile(raw_rows, geo_source):
    """Match sample, concentration and coordinates; never choose among ambiguous IDs."""
    left = keys(raw_rows.loc[raw_rows.source_dataset == "GEOROC"])
    left["row"] = left.index
    right = keys(geo_source)
    right = right.loc[right._sample.ne("") & right[KEYS[1:]].notna().all(axis=1)]
    if right.source_unique_id.isna().any() or right.source_unique_id.duplicated().any():
        raise ValueError("Source IDs missing or duplicated")
    valid = left._sample.ne("") & left[KEYS[1:]].notna().all(axis=1)
    candidates = left.loc[valid, ["row", *KEYS]].merge(right[[*KEYS, *PROVENANCE]], on=KEYS, how="left")
    records = []
    for row, matches in candidates.groupby("row"):
        found = matches.loc[matches.source_unique_id.notna()]
        record = {"row": int(row), "matches": len(found),
                  "matched_ids": ";".join(found.source_unique_id.astype(str)),
                  "match_state": "unique" if len(found)==1 else "ambiguous" if len(found)>1 else "unmatched"}
        if len(found)==1:
            record.update({c: found.iloc[0][c] for c in PROVENANCE})
        records.append(record)
    for row in left.loc[~valid,"row"]:
        records.append({"row": int(row), "matches": 0, "match_state": "missing_match_key"})
    return pd.DataFrame(records)


def verify_historical_oof(prepared, oof, checkpoint_folder):
    oof = oof.loc[oof.version == "candidate_3420"].copy()
    n = len(prepared.y)
    if oof[["algorithm","row"]].duplicated().any() or oof.algorithm.nunique()!=6:
        raise ValueError("Unexpected historical OOF identity")
    folds = []
    fold_ids = np.zeros(n, dtype=int)
    for fold in range(1,6):
        record = json.loads((checkpoint_folder/f"fold_{fold:02d}.json").read_text(encoding="utf-8"))
        test = np.asarray(record["test_indices"],dtype=int)
        train = np.setdiff1d(np.arange(n),test)
        assert_group_isolation(train,test,prepared.groups)
        folds.append((train,test))
        fold_ids[test] = fold
    assert_partition_coverage(folds,n)
    for _, part in oof.groupby("algorithm"):
        part = part.sort_values("row")
        if not (np.array_equal(part.row, np.arange(n))
                and np.allclose(part.real, prepared.y.to_numpy(float), rtol=0, atol=1e-10)
                and np.array_equal(part.group.astype(str),prepared.groups.astype(str))
                and np.array_equal(part.source.astype(str),prepared.frame.source_dataset.astype(str))
                and np.array_equal(part.fold,fold_ids)):
            raise ValueError("OOF rows do not match the hash-verified prepared dataset and folds")
    return oof,folds


def run(dataset, oof_path, checkpoint_folder, geo_source_path, output):
    if file_sha256(dataset)!=SOURCE_SHA or file_sha256(oof_path)!=OOF_SHA:
        raise ValueError("Source or historical OOF hash changed")
    prepared=prepare_training_data(dataset)
    oof,folds=verify_historical_oof(prepared,pd.read_csv(oof_path),checkpoint_folder)
    raw=prepared.raw_frame.loc[prepared.frame.index].reset_index(drop=True)
    geo=pd.read_csv(geo_source_path,dtype={"source_unique_id":str})
    linkage=reconcile(raw,geo)
    linked=raw.reset_index(names="row").merge(linkage,on="row",how="left",validate="one_to_one")
    linked["source_row_position"]=prepared.frame.index.to_numpy()
    linked["numeric_predictors_missing"]=prepared.x[prepared.schema.numeric].isna().sum(axis=1).to_numpy()
    linked["raw_rock_type_missing"]=raw.Rock_type.isna()
    linked["prepared_rock_type"]=prepared.x.Rock_type.to_numpy()
    top=pd.concat([part.nlargest(10,"abs_error") for _,part in oof.groupby("algorithm")])
    union=sorted(top.row.unique())
    ids=linked.set_index("row")
    if not ids.loc[union,"match_state"].eq("unique").all():
        raise ValueError("Cannot certify source identity for every top-error sample")
    findings=[]
    for row in union:
        sample=ids.loc[row]
        row_oof=oof.loc[oof.row==row]
        fold=int(row_oof.fold.iloc[0])
        train,_=folds[fold-1]
        ytrain=prepared.y.iloc[train].to_numpy(float)
        train_rows=linked.iloc[train]
        known=train_rows.match_state.eq("unique")
        same_publication=known & train_rows.source_publication.eq(sample.source_publication)
        same_location=known & train_rows.source_location.eq(sample.source_location)
        findings.append({
            "row": int(row),"fold":fold,"source_row_position":int(sample.source_row_position),
            "source_sample":sample.source_sample,"source_unique_id":sample.source_unique_id,
            "source_publication_code":sample.source_publication,
            "source_rock_class":sample.source_rock_class,"source_location":sample.source_location,
            "li_ppm":float(sample.Li_icpms),"li_unit":sample.source_li_unit,"li_method":sample.source_li_method,
            "raw_rock_type_missing":bool(sample.raw_rock_type_missing),"prepared_rock_type":sample.prepared_rock_type,
            "numeric_predictors_missing":int(sample.numeric_predictors_missing),
            "qc_sum_no_water":float(sample["SUM (no water)"]),"fe_convention":sample.Fe_total_provenance,
            "feot_raw":sample.source_feot_raw,"fe2o3_raw":sample.source_fe2o3_raw,
            "train_count":len(train),"train_max_li_ppm":float(ytrain.max()),"train_q99_li_ppm":float(np.quantile(ytrain,.99)),
            "train_count_ge_observed_li":int((ytrain>=sample.Li_icpms).sum()),
            "train_count_ge_1000_ppm":int((ytrain>=1000).sum()),
            "train_same_publication_confirmed":int(same_publication.sum()),
            "train_same_location_confirmed":int(same_location.sum()),
            "train_georoc_unresolved_provenance":int((train_rows.source_dataset.eq("GEOROC") & ~known).sum()),
            "top10_algorithms":int(top.loc[top.row==row,"algorithm"].nunique()),
        })
    dossier=pd.DataFrame(findings)
    output.mkdir(parents=True,exist_ok=True)
    dossier.to_csv(output/"extreme_sample_dossier.csv",index=False)
    linkage.to_csv(output/"georoc_linkage.csv",index=False)
    top.merge(dossier,on=["row","fold"],validate="many_to_one").to_csv(output/"extreme_errors_by_algorithm.csv",index=False)
    summary={"dataset_sha256":SOURCE_SHA,"oof_sha256":OOF_SHA,"georoc_harmonization_sha256":file_sha256(geo_source_path),
             "historical_checkpoint_sha256":{p.name:file_sha256(p) for p in sorted(checkpoint_folder.glob("fold_*.json"))},
             "georoc_linkage":linkage.match_state.value_counts().to_dict(),"top_distinct_samples":len(dossier),
             "top_by_publication":dossier.source_publication_code.value_counts().to_dict(),
             "top_without_historical_rock_type":int(dossier.raw_rock_type_missing.sum()),
             "top_above_training_maximum":int((dossier.li_ppm>dossier.train_max_li_ppm).sum()),
             "top_zero_confirmed_publication_train":int(dossier.train_same_publication_confirmed.eq(0).sum()),
             "scope":"Retrospective; citation codes not full bibliographic records; analytical method/CRS not certified; no causal proof or independent holdout."}
    (output/"provenance_audit.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","oof","checkpoints","georoc","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.oof,args.checkpoints,args.georoc,args.output),indent=2,ensure_ascii=False))
