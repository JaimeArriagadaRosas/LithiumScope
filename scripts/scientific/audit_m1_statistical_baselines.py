"""Read-only retrospective comparison of train-only baselines and archived OOF."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.evaluation.baseline import evaluate_baselines
from lithiumscope.model_1.evaluation.selection import baseline_eligibility
from lithiumscope.model_1.evaluation.metrics import regression_metrics


def run(dataset, archive, output):
    if file_sha256(dataset)!="94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94":
        raise ValueError("Historical dataset changed")
    if file_sha256(archive)!="a526bfa1985b0de34c3e4bf7dbbd7e4ca428f16630416b6d57fdd13a7c0e8b06":
        raise ValueError("Historical OOF evidence changed")
    p=prepare_training_data(dataset)
    rows,folds,oof=evaluate_baselines(p,load_config("model_1"))
    hist=pd.read_csv(archive)
    hist=hist[hist.version=="candidate_3420"]
    ref=oof[oof.algorithm=="baseline_mean"].sort_values("row_position")
    if hist.algorithm.nunique()!=6 or hist[["algorithm","row"]].duplicated().any():
        raise ValueError("Invalid historical algorithm/row coverage")
    for alg,g in hist.groupby("algorithm"):
        g=g.sort_values("row")
        if not (np.array_equal(g.row,ref.row_position)
                and np.allclose(g.real,ref.Li_icpms_real,rtol=0,atol=1e-10)
                and np.array_equal(g.fold,ref.outer_fold)
                and np.array_equal(g.group.astype(str),ref.spatial_group.astype(str))
                and np.allclose(g.train_q75,ref.fold_train_q75_ppm,rtol=0,atol=1e-10)):
            raise ValueError("Historical fold/label identity changed")
        metric=regression_metrics(g.real,g.pred).to_dict()
        byfold=[regression_metrics(f.real,f.pred).to_dict() for _,f in g.groupby("fold")]
        rows.append({"algorithm":alg,"is_baseline":False,"status":"ok",
                     **{f"{m}_mean":float(np.mean([v[m] for v in byfold])) for m in ("rmse","mae","r2")},
                     **{f"oof_{k}":v for k,v in metric.items()}})
    output.mkdir(parents=True,exist_ok=False)
    folds.to_csv(output/"baseline_fold_metrics.csv",index=False)
    oof.to_csv(output/"baseline_oof_predictions.csv",index=False)
    table=baseline_eligibility(pd.DataFrame(rows),preflight_ok=True)
    # No current preflight or release claim for archived candidates.
    table["scientific_candidate_eligible"]=False
    table["scientific_candidate_reason"]="retrospective_only"
    table.to_csv(output/"retrospective_baseline_comparison.csv",index=False)
    print(table[["algorithm","rmse_mean","mae_mean","oof_rmse","oof_mae",
                 "rmse_gain_over_best_baseline","baseline_gate_pass"]].to_string(index=False))
    (output/"summary.json").write_text(json.dumps({
        "dataset_sha256":file_sha256(dataset),"historical_oof_sha256":file_sha256(archive),
        "rows":len(p.y),"folds":5,"scope":"retrospective comparison, no fit, no release",
        "artifacts":{p.name:file_sha256(p) for p in output.glob("*.csv")}},indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","archive","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    run(args.dataset,args.archive,args.output)
