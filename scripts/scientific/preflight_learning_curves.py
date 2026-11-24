"""Audit curve partitions on frozen cohorts without fitting an estimator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.scientific.learning_curves import build_curve_plan,curve_plan_records
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.evaluation.baseline import baseline_splits
from lithiumscope.model_2.validation.spatial import make_scientific_folds
from lithiumscope.model_2.validation.quality import training_quality_mask


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    m1=PROJECT_ROOT/"data/processed/model_1/training_combined.csv"
    cfg1=load_config("model_1");cfg2=load_config("model_2")
    p=prepare_training_data(m1)
    splits,_=baseline_splits(p,cfg1)
    m2=PROJECT_ROOT/cfg2["training"]["manifest_path"]
    frame=pd.read_csv(m2)
    quality,_=training_quality_mask(m2,inventory_path=PROJECT_ROOT/cfg2["quality_training"]["inventory_path"],
                                    minimum_joint_valid=cfg2["quality_training"]["minimum_joint_valid_fraction"])
    folds=make_scientific_folds(frame.Li_icpms,frame.spatial_group,n_splits=5,quantile=.75)
    definitions=[("model_1",p.y,p.groups,splits,None,None),
        ("model_2",frame.Li_icpms,frame.spatial_group,[(f.train_idx,f.test_idx) for f in folds],quality,
         [f.label_threshold_ppm for f in folds])]
    summary={"scope":"preflight only; no predictor fit; no external validation","models":{}}
    for name,y,g,splits,keep,endpoints in definitions:
        plan=build_curve_plan(y,g,splits,kind="regression" if name=="model_1" else "classification",
                              eligible=keep,endpoints=endpoints)
        rows=curve_plan_records(plan)
        (out/f"{name}_partitions.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        table=pd.DataFrame(rows).drop(columns=["train_positions","test_positions"])
        table.to_csv(out/f"{name}_plan.csv",index=False)
        summary["models"][name]={"observations":len(y),"groups":int(pd.Series(g).nunique()),"partitions":len(plan),
            "ready":int((table.status=="ready").sum()),"unavailable":int((table.status!="ready").sum()),
            "min_train_rows":int(table.n_train.min()),"max_train_rows":int(table.n_train.max()),
            "min_train_groups":int(table.train_groups.min()),"all_test_preserved":True,
            "eligible":int(np.sum(keep)) if keep is not None else len(y),
            "dataset_sha256":file_sha256(m1 if name=="model_1" else m2)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    run(parser.parse_args().output)
