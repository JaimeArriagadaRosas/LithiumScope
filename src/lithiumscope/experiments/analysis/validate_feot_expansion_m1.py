"""Development-only spatial test of M1 GEOROC FEOT expansion on novel B groups."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.core.config import load_config
from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks
from lithiumscope.experiments.analysis.run_vnext_m1 import (
    _frame, _metrics, quality_candidate_frames,
)
from lithiumscope.experiments.analysis.run_vnext_m1_diagnostics import (
    cluster_bootstrap_improvement,
)
from lithiumscope.model_1.pipeline import prepare_prediction_data
from lithiumscope.model_1.steps.step_05_target_filtering import filter_target
from lithiumscope.model_1.steps.step_09_validation import build_spatial_groups

DESTINATION = (
    RESULTS_DIR / "scientific_simulations" / "vnext_20261106"
    / "stage4_m1_feot_verification"
)


def heldout_new_geography(a: pd.DataFrame, b: pd.DataFrame):
    ga=build_spatial_groups(a)
    gb=build_spatial_groups(b)
    if ga is None or gb is None:
        raise RuntimeError("Spatial groups required")
    data=b.loc[~gb.isin(set(ga))].copy()
    groups=gb.loc[data.index].copy()
    if groups.nunique()<5:
        raise ValueError("Need at least five new spatial groups")
    return data, groups


def evaluate(root: Path=PROJECT_ROOT, output: Path=DESTINATION):
    a_path=root/"data/processed/model_1/training_combined.csv"
    b_path=(root/"results/scientific_simulations/vnext_20261106"
            /"stage2_m1/feot_combined_experiment_only.csv")
    a=_frame(pd.read_csv(a_path,low_memory=False))
    a,_=filter_target(a,list(load_config("model_1")["data"]["target_candidates"]))
    a=a.loc[protocol_masks(a)["A_original"]].reset_index(drop=True)
    b=quality_candidate_frames(pd.read_csv(b_path,low_memory=False))["B"]
    b=b.reset_index(drop=True)
    new,groups=heldout_new_geography(a,b)
    new=new.reset_index(drop=True)
    groups=groups.reset_index(drop=True)
    assert groups.nunique()>0
    old_groups=set(build_spatial_groups(a))
    assert not old_groups.intersection(set(groups))

    bundle=joblib.load(
        root/"models/model_1/trained/training_20261105_204029_m0300/model.joblib"
    )
    schema=bundle["schema"]
    est=bundle["estimator"]
    xa=prepare_prediction_data(a,"svm",schema)
    xb=prepare_prediction_data(b,"svm",schema)
    xt=prepare_prediction_data(new,"svm",schema)
    ya=a.Li_icpms.to_numpy(dtype=float)
    yb=b.Li_icpms.to_numpy(dtype=float)
    yt=new.Li_icpms.to_numpy(dtype=float)
    model_a=clone(est)
    model_a.fit(xa,ya)
    pred_a=np.asarray(model_a.predict(xt),dtype=float)
    pred_b=np.full(len(new),np.nan)
    bg=build_spatial_groups(b)
    records=[]
    for index,(_,test) in enumerate(
        GroupKFold(n_splits=5).split(xt,groups=groups),start=1
    ):
        heldout=set(groups.iloc[test])
        train=np.flatnonzero(~bg.isin(heldout).to_numpy())
        assert heldout.isdisjoint(set(bg.iloc[train]))
        model_b=clone(est)
        model_b.fit(xb.iloc[train],yb[train])
        pred_b[test]=model_b.predict(xt.iloc[test])
        ma=_metrics(yt[test],pred_a[test])
        mb=_metrics(yt[test],pred_b[test])
        records.append({
            "fold":index,"n_test":len(test),"n_test_groups":len(heldout),
            "n_train_A":len(a),"n_train_B":len(train),
            "RMSE_A":ma["rmse"],"RMSE_B":mb["rmse"],
            "MAE_A":ma["mae"],"MAE_B":mb["mae"],
        })
        print("M1_NEW_GEO_FOLD",index,"n",len(test),
              "A",round(ma["rmse"],3),"B",round(mb["rmse"],3),flush=True)
    if not np.isfinite(pred_b).all():
        raise RuntimeError("Missing predictions for newly recovered groups")
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output/"fold_metrics.csv",index=False)
    pd.DataFrame({
        "source_citation":new.CITATIONS.to_numpy(),
        "spatial_group":groups.to_numpy(),
        "y_true":yt,"prediction_A":pred_a,"prediction_B":pred_b,
    }).to_csv(output/"paired_predictions.csv",index=False)
    conclusion={
        "test_labels":"GEOROC FEOT-B samples: original publications not independently certified",
        "spatial_holdout_groups":int(groups.nunique()),
        "test_samples":len(new),
        "metrics_A":_metrics(yt,pred_a),
        "metrics_B":_metrics(yt,pred_b),
        "gain_ci_by_group":cluster_bootstrap_improvement(
            yt,pred_a,pred_b,groups,repeats=400,seed=123
        ),
        "baseline_A_training":len(a),
        "B_training_total_excluding_groups_each_fold":len(b),
        "external_independence":False,
        "no_sentinel_features_added":True,
        "source_sha256":hashlib.sha256(b_path.read_bytes()).hexdigest(),
    }
    (output/"summary.json").write_text(
        json.dumps(conclusion,ensure_ascii=False,indent=2),encoding="utf8"
    )
    print("M1_NEW_GEO_SUMMARY",json.dumps(conclusion,ensure_ascii=False),flush=True)
    return conclusion


if __name__ == "__main__":
    evaluate()
