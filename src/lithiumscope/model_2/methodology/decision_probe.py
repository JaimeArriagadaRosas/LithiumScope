"""Phase C: train-only inner-group OOF decision-threshold probe.

Uses already saved outer-model scores; fits ONLY small inner models to
estimate the selection rule. No calibration, no ten-case demo, no promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.ablations import (
    build_inputs, nested_decision_probe,
)

ARMS=("legacy_all","expanded_buffer_5km")


def run_probe(root:Path, pilot:Path):
    if not (pilot/"experiment_summary.json").is_file():
        raise FileNotFoundError("Run the bounded seven-arm ablations first")
    summary=json.loads((pilot/"experiment_summary.json").read_text(encoding="utf-8"))
    if summary.get("state")!="completed" or not summary.get("same_706_held_out"):
        raise ValueError("Pilot is incomplete or outer test indices changed")
    contract,predictors,folds,features,hashes=build_inputs(root)
    results=[]
    rowdata=[]
    for arm in ARMS:
        print("NESTED_START",arm,flush=True)
        oof=pd.read_csv(pilot/("oof_"+arm+".csv"))
        if len(oof)!=706 or oof.test_index.sort_values().tolist()!=list(range(706)):
            raise ValueError("Expected the exact frozen 706 outer test rows")
        rules=nested_decision_probe(contract,predictors,folds,arm=arm)
        rules.to_csv(pilot/("inner_frozen_thresholds_"+arm+".csv"),index=False)
        look=rules.set_index("fold").decision_threshold
        oof["train_only_threshold"]=oof.fold.map(look).astype(float)
        oof["native_threshold_prediction"]=(oof.score>=oof.train_only_threshold).astype(int)
        oof["fixed_half_prediction"]=(oof.score>=.5).astype(int)
        y=oof.y_true.to_numpy(int)
        native=oof.native_threshold_prediction.to_numpy(int)
        fixed=oof.fixed_half_prediction.to_numpy(int)
        stats={
            "arm":arm,
            "samples":len(oof),
            "positives":int(y.sum()),
            "n_inner_folds_per_outer":3,
            "selection_rule":"inner group out-of-fold, outer-train Li only",
            "outer_test_used_for_threshold_selection":False,
            "thresholds":rules.decision_threshold.tolist(),
            "native_recall":float(recall_score(y,native,zero_division=0)),
            "native_precision":float(precision_score(y,native,zero_division=0)),
            "native_balanced_accuracy":float(balanced_accuracy_score(y,native)),
            "native_f1":float(f1_score(y,native,zero_division=0)),
            "fixed_half_recall":float(recall_score(y,fixed,zero_division=0)),
            "fixed_half_balanced_accuracy":float(balanced_accuracy_score(y,fixed)),
        }
        for fold,group in oof.groupby("fold"):
            yy=group.y_true.to_numpy(int)
            dd=group.native_threshold_prediction.to_numpy(int)
            stats_fold={
                "arm":arm,"fold":int(fold),"threshold":float(look.loc[fold]),
                "n_test":len(group),
                "positives":int(yy.sum()),
                "recall":float(recall_score(yy,dd,zero_division=0)),
                "precision":float(precision_score(yy,dd,zero_division=0)),
                "balanced_accuracy":float(balanced_accuracy_score(yy,dd)),
            }
            rowdata.append(stats_fold)
        oof.to_csv(pilot/("oof_with_train_only_decisions_"+arm+".csv"),index=False)
        results.append(stats)
        (pilot/"decision_probe_progress.json").write_text(
            json.dumps({"state":"running","completed":results},indent=2),encoding="utf-8")
        print("NESTED_DONE",json.dumps(stats),flush=True)
    pd.DataFrame(rowdata).to_csv(pilot/"inner_decision_fold_metrics.csv",index=False)
    (pilot/"decision_probe_summary.json").write_text(
        json.dumps({"state":"completed","results":results,
            "limitations":[
                "No probability calibration was fitted.",
                "Outer results are exploratory after prior model selection.",
                "The ten known demo cases were never read.",
                "Inner folds require group-disjoint fitting, enforced by StratifiedGroupKFold.",
                "Li thresholds are train-derived within the inner folds.",
            ]},indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print("DECISION_PROBE_COMPLETE",pilot,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True)
    args=p.parse_args()
    path=Path(args.pilot)
    if not path.is_absolute():path=PROJECT_ROOT/path
    run_probe(PROJECT_ROOT,path)

if __name__=="__main__":
    main()
