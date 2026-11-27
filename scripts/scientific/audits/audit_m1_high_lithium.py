"""Retrospective auxiliary confusion matrices for both archived M1 cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.diagnostics.high_lithium import high_lithium_diagnostics

ARCHIVE_SHA="a526bfa1985b0de34c3e4bf7dbbd7e4ca428f16630416b6d57fdd13a7c0e8b06"


def run(archive,output):
    if file_sha256(archive)!=ARCHIVE_SHA:
        raise ValueError("Frozen historical OOF archive changed")
    source=pd.read_csv(archive)
    tables=[];strata=[]
    for (version,algorithm),part in source.groupby(["version","algorithm"],sort=True):
        part=part.sort_values("row")
        expected=3420 if version=="candidate_3420" else 1120
        if len(part)!=expected or not np.array_equal(part.row,np.arange(expected)) or part.fold.nunique()!=5:
            raise ValueError("Archived row coverage differs")
        # Reconstruct the complement of each outer TEST from the complete archived OOF cohort.
        for fold,g in part.groupby("fold"):
            train=part.loc[part.fold!=fold]
            if set(g.group.astype(str))&set(train.group.astype(str)):
                raise ValueError("Archived spatial groups overlap")
            if not np.allclose(g.train_q75,np.quantile(train.real,.75),rtol=0,atol=1e-10):
                raise ValueError("Archived cutoff does not match TRAIN")
        frame=pd.DataFrame({"row_position":part.row,"outer_fold":part.fold,
            "fold_train_q75_ppm":part.train_q75,"Li_icpms_real":part.real,"Li_icpms_predicted":part.pred,
            "observed_high":(part.real>=part.train_q75).astype(int),
            "predicted_high":(part.pred>=part.train_q75).astype(int)})
        for col in ("source","source_dataset"):
            if col in part:
                frame["source_dataset"]=part[col]
                break
        t,s=high_lithium_diagnostics(frame)
        tables.append(t.assign(version=version,algorithm=algorithm))
        strata.append(s.assign(version=version,algorithm=algorithm))
    table=pd.concat(tables,ignore_index=True)
    if len(tables)!=12:
        raise ValueError("Expected six families in two cohorts")
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/"confusion_by_fold_and_pooled.csv",index=False)
    pd.concat(strata,ignore_index=True).to_csv(output/"residuals_by_source_and_high_li.csv",index=False)
    if file_sha256(archive)!=ARCHIVE_SHA:
        raise ValueError("Archive changed")
    pooled=table.loc[(table.scope=="pooled_fold_specific_endpoints")&(table.version=="candidate_3420")]
    print(pooled[["algorithm","n","tn","fp","fn","tp","sensitivity","specificity","balanced_accuracy"]].to_string(index=False))
    summary={"archive_sha256":ARCHIVE_SHA,"cohort_algorithm_pairs":len(tables),
             "fold_confusion_matrices":int((table.scope=="outer_fold").sum()),
             "pooled_confusion_matrices":len(tables),"predictions":len(source),
             "fit":False,"scope":"Retrospective auxiliary diagnostic; fold TRAIN Q75, not classification target or independent holdout"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.archive,args.output)
