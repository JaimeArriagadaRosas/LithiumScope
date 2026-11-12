"""Bounded M2 Phase C scientific ablations, no main menu or model promotion.

Seven predeclared paired CatBoost arms, same 706 historical held-out rows.
Five frozen anchored outer folds and fold-specific historical TRAIN-only Li q75.
Extra GEOROC rows are only admitted to train if originally in that outer fold train.
No Optuna, hyperparameter or threshold selection on outer scores.
Ten known demonstration cases are intentionally NEVER read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    balanced_accuracy_score, recall_score, precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.contracts import sample_contract
from lithiumscope.model_2.methodology.spatial import EARTH_RADIUS_M, _positions
from lithiumscope.model_2.methodology.decision import (
    freeze_inner_oof_threshold, evaluate_frozen_threshold,
)
from sklearn.neighbors import BallTree


HISTORICAL_N=706
FOLD_REL="results/model_2/runs/training_20261111_170832_m0300/manifests/scientific_folds.json"
MAN_REL="data/processed/model_2/vnext_expanded/training_manifest.csv"
FEATURE_REL="data/processed/model_2/vnext_expanded/spectral_features.csv"
ARMS=(
    "legacy_all",
    "expanded_all",
    "expanded_buffer_5km",
    "expanded_buffer_10km",
    "expanded_exclude_shared_scenes",
    "expanded_drop_mixed_train_sites",
    "expanded_raw_spectral_only",
    "expanded_inverse_site_weights",
)
PARAMS={
    "iterations":160,
    "depth":5,
    "learning_rate":0.065,
    "l2_leaf_reg":5.0,
    "loss_function":"Logloss",
    "random_seed":42,
    "verbose":False,
    "allow_writing_files":False,
    "task_type":"CPU",
    "thread_count":4,
}
# Only predeclared direct-band summary columns; no data-driven feature selection.
RAW_PREFIX=("red", "green", "blue", "nir", "swir16", "swir22")
RAW_SUFFIX=("mean", "std", "p10", "p50", "p90", "iqr", "texture_mean", "texture_p90")
SCORE_T=0.5


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def score_metrics(y,s, threshold=.5):
    y=np.asarray(y,int);s=np.asarray(s,float)
    if not len(y) or not set(y).issubset({0,1}) or not np.isfinite(s).all():
        raise ValueError("Invalid evaluation labels/scores")
    d=(s>=float(threshold)).astype(int)
    return {
        "n":int(len(y)), "positives":int(y.sum()),
        "auc":float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
        "ap":float(average_precision_score(y,s)) if len(np.unique(y))==2 else None,
        "brier":float(brier_score_loss(y,s)),
        "balanced_accuracy_at_threshold":float(balanced_accuracy_score(y,d)) if len(np.unique(y))==2 else None,
        "recall_at_threshold":float(recall_score(y,d,zero_division=0)),
        "precision_at_threshold":float(precision_score(y,d,zero_division=0)),
        "evaluated_threshold":float(threshold),
        "median_score":float(np.median(s)),
    }


def define_outer_fold(fold:dict, manifest:pd.DataFrame):
    """Frozen historical TEST; labels learned from frozen historical TRAIN Li only."""
    train=np.asarray(fold["train_indices"],int)
    hist_train=train[train<HISTORICAL_N]
    test=np.asarray(fold["test_indices"],int)
    test=test[test<HISTORICAL_N]
    if not len(hist_train) or not len(test):
        raise ValueError("Empty frozen historical training or test partition")
    threshold=float(np.quantile(
        manifest.Li_icpms.iloc[hist_train].to_numpy(float),.75))
    if abs(threshold-float(fold["label_threshold_ppm"]))>1e-8:
        raise ValueError("The historical fold threshold was changed")
    if set(train)&set(test):
        raise ValueError("Direct train/test overlap")
    groups=manifest.spatial_group.astype(str)
    if set(groups.iloc[train])&set(groups.iloc[test]):
        raise ValueError("Frozen fold spatial groups overlap")
    return train,hist_train,test,threshold


def select_train_indices(
    arm:str,fold:dict,contract:pd.DataFrame, *, threshold:float
):
    """Only training data (and frozen test coordinates for distance) influence selection.

    A fixed nominal q75 label is only used to detect mutually inconsistent
    Li at an identical screening site within the CURRENT TRAIN subset.
    """
    if arm not in ARMS:
        raise ValueError("Unknown predeclared ablation")
    original=np.asarray(fold["train_indices"],int)
    test=np.asarray(fold["test_indices"],int)
    hist_test=test[test<HISTORICAL_N]
    train=original[original<HISTORICAL_N] if arm=="legacy_all" else original
    if arm=="expanded_exclude_shared_scenes":
        # A held-out satellite scene cannot be used in this diagnostic arm.
        # The scene IDs are metadata; no test Li or test labels are consulted.
        test_scenes=set(contract.scene_id.iloc[hist_test].astype(str))-{""}
        train=train[~contract.scene_id.iloc[train].astype(str).isin(test_scenes).to_numpy()]
    if arm=="expanded_drop_mixed_train_sites":
        selected=contract.iloc[train]
        labels=(selected.Li_icpms.to_numpy(float)>=threshold)
        info=pd.DataFrame({
            "site":selected.site_screening_id.to_numpy(),
            "label":labels,
        })
        mixed=info.groupby("site").label.nunique()
        problematic=set(mixed.index[mixed>1])
        keep=~selected.site_screening_id.isin(problematic).to_numpy()
        train=train[keep]
    if arm in ("expanded_buffer_5km","expanded_buffer_10km"):
        # Drop train rows near held-out historical TEST coordinates only.
        distance=5000.0 if arm.endswith("5km") else 10000.0
        locations=_positions(contract)
        test_tree=BallTree(locations[hist_test],metric="haversine")
        close=test_tree.query_radius(
            locations[train],r=distance/EARTH_RADIUS_M,count_only=True
        )>0
        train=train[~close]
    return train


def site_weights(contract:pd.DataFrame,train:np.ndarray)->np.ndarray:
    site=contract.site_screening_id.iloc[train].reset_index(drop=True)
    weights=1.0/site.map(site.value_counts()).to_numpy(float)
    weights=weights/weights.mean()
    if not (np.isfinite(weights).all() and (weights>0).all()):
        raise ValueError("Invalid site-weighted training weights")
    return weights


def fit_classifier(X_train,y_train,X_test, *, weights=None):
    from catboost import CatBoostClassifier
    if np.unique(y_train).size!=2:
        raise ValueError("Training subset contains only one class")
    est=CatBoostClassifier(**PARAMS)
    est.fit(X_train,y_train,sample_weight=weights)
    return est.predict_proba(X_test)[:,1]


def build_inputs(root:Path):
    man=root/MAN_REL;feat=root/FEATURE_REL;fold=root/FOLD_REL
    contract=sample_contract(pd.read_csv(man))
    x=pd.read_csv(feat)
    assert len(contract)==len(x)==783
    assert contract.spatial_group.astype(str).reset_index(drop=True).equals(
        x.spatial_group.astype(str).reset_index(drop=True)
    )
    assert np.allclose(
        contract.Li_icpms.to_numpy(float),x.Li_icpms.to_numpy(float),
        atol=1e-9
    )
    predictors=x.drop(columns=["Li_icpms","spatial_group"])
    assert predictors.shape[1]==126 and np.isfinite(predictors.to_numpy(float)).all()
    names=[f"{prefix}_{suffix}" for prefix in RAW_PREFIX for suffix in RAW_SUFFIX]
    if not set(names).issubset(predictors.columns):
        raise ValueError("The frozen direct-band feature schema changed")
    assert len(names)==48
    folds=json.loads(fold.read_text(encoding="utf-8"))
    if len(folds)!=5:raise ValueError("Expected 5 frozen scientific folds")
    expected=np.zeros(HISTORICAL_N,int)
    for f in folds:
        _,_,test,_=define_outer_fold(f,contract)
        expected[test]+=1
    if not np.all(expected==1):
        raise ValueError("Historical 706 test indices are not a partition")
    return contract,predictors,folds,names,{p.name:sha(p) for p in (man,feat,fold)}


def bootstrap_group_delta(predictions:pd.DataFrame,reference:str,arm:str,
                          *,repeats=320, seed=20261112):
    # Group bootstrap is descriptive; models were chosen in another development
    # competition. Multiple comparisons and dependence remain.
    df=predictions[predictions.arm.isin((reference,arm))]
    pv=df.pivot(index="test_index",columns="arm",values="score").sort_index()
    ref=pv[reference].to_numpy(float)
    alternative=pv[arm].to_numpy(float)
    md=df[df.arm==reference].set_index("test_index").loc[pv.index]
    y=md.y_true.to_numpy(int)
    gr=md.spatial_group.astype(str).to_numpy()
    unique=np.unique(gr)
    by_group=[np.flatnonzero(gr==g) for g in unique]
    rng=np.random.default_rng(seed)
    deltas={"auc":[],"ap":[],"brier":[]}
    for _ in range(repeats):
        sampled=rng.integers(0,len(unique),len(unique))
        ix=np.concatenate([by_group[j] for j in sampled])
        if np.unique(y[ix]).size<2:
            continue
        deltas["auc"].append(float(roc_auc_score(y[ix],alternative[ix])
                                  -roc_auc_score(y[ix],ref[ix])))
        deltas["ap"].append(float(average_precision_score(y[ix],alternative[ix])
                                 -average_precision_score(y[ix],ref[ix])))
        deltas["brier"].append(float(brier_score_loss(y[ix],alternative[ix])
                                    -brier_score_loss(y[ix],ref[ix])))
    out={"reference":reference,"arm":arm,"iterations_requested":repeats,
         "groups":len(unique)}
    for key,values in deltas.items():
        out[key+"_delta_ci95_low"]=float(np.quantile(values,.025))
        out[key+"_delta_ci95_high"]=float(np.quantile(values,.975))
        out[key+"_bootstrap_valid"]=len(values)
    return out


def nested_decision_probe(contract,predictors,folds, *,arm):
    """Before outer test, determine threshold via 3 geographically grouped inner fits.

    Thresholds, including reference Li q75, use inner TRAIN-only information.
    Decision estimates still exploratory because these arms are reviewed on
    the same 706 outer observations. No probability calibrator is fitted.
    """
    from lithiumscope.model_2.methodology.decision import freeze_inner_oof_threshold
    records=[]
    all_outer=[]
    for k,fold in enumerate(folds,1):
        full_train,hist_train,outer_test,outer_threshold=define_outer_fold(
            fold,contract)
        tr=select_train_indices(arm,fold,contract,threshold=outer_threshold)
        # Use only sites in current outer train for nested validation.
        train_group=contract.spatial_group.iloc[tr].astype(str).reset_index(drop=True)
        split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
        outer_labels=(contract.Li_icpms.iloc[tr].to_numpy(float)>=outer_threshold).astype(int)
        oof_rows=[]
        # No target from outer TEST used in inner training, labels, or threshold selection.
        for inner,(i_tr,i_val) in enumerate(split.split(
                predictors.iloc[tr],outer_labels,groups=train_group),1):
            inner_global_tr=tr[i_tr]
            inner_global_valid=tr[i_val]
            inner_hist=inner_global_tr[inner_global_tr<HISTORICAL_N]
            if len(inner_hist)<10:
                raise ValueError("Too few historical inner train rows")
            inner_threshold=float(np.quantile(
                contract.Li_icpms.iloc[inner_hist].to_numpy(float),.75))
            y_train=(contract.Li_icpms.iloc[inner_global_tr].to_numpy(float)>=inner_threshold).astype(int)
            y_valid=(contract.Li_icpms.iloc[inner_global_valid].to_numpy(float)>=inner_threshold).astype(int)
            if len(np.unique(y_valid))<2:
                raise ValueError("Inner fold has one validation class; no random fallback")
            w=site_weights(contract,inner_global_tr) if arm=="expanded_inverse_site_weights" else None
            score=fit_classifier(predictors.iloc[inner_global_tr],
                y_train,predictors.iloc[inner_global_valid],weights=w)
            for orig,yt,s in zip(inner_global_valid,y_valid,score):
                oof_rows.append({"row_index":int(orig),
                    "group":str(contract.spatial_group.iloc[orig]),
                    "inner_fold":inner,"y_true":int(yt),"score":float(s)})
        frozen=freeze_inner_oof_threshold(
            pd.DataFrame(oof_rows),permitted_outer_train_indices=tr,
            reserved_outer_test_indices=np.asarray(fold["test_indices"],int))
        # Use the outer TEST score of the already fitted outer model.
        records.append({"fold":k,"arm":arm,"decision_threshold":frozen.selection.threshold,
                        "train_rows":len(tr),"inner_groups":frozen.group_count,
                        "inner_oof_sha256":frozen.inner_rows_sha256})
    return pd.DataFrame(records)


def run(root:Path,output:Path, *, nested=False):
    contract,predictors,folds,raw_names,hashes=build_inputs(root)
    output.mkdir(parents=True,exist_ok=True)
    configuration={
        "schema_version":1,"scope":"Phase C fixed ablations; historical holdout only",
        "arms":list(ARMS),"params":PARAMS,
        "historical_test_rows":HISTORICAL_N,
        "feature_count_full":len(predictors.columns),
        "feature_count_raw":len(raw_names),
        "raw_feature_columns":raw_names,
        "folds":len(folds),"random_seed":42,
        "known_ten_cases_read":False,"new_sources_downloaded":False,
        "source_sha256":hashes,
        "no_hyperparameter_selection":True,
        "no_model_activation":True,"no_final_refit":True,
        "outer_threshold_provenance":"q75 historical TRAIN per fold",
        "warning":"Outer results exploratory for arm development; no confirmatory holdout.",
    }
    (output/"experiment_design.json").write_text(
        json.dumps(configuration,indent=2,ensure_ascii=False),encoding="utf-8")
    allrows=[]
    completed={}
    for arm in ARMS:
        t0=time.monotonic()
        arm_rows=[]
        for k,fold in enumerate(folds,1):
            base,legacy,te,thr=define_outer_fold(fold,contract)
            tr=select_train_indices(arm,fold,contract,threshold=thr)
            columns=raw_names if arm=="expanded_raw_spectral_only" else list(predictors.columns)
            y_tr=(contract.Li_icpms.iloc[tr].to_numpy(float)>=thr).astype(int)
            y_te=(contract.Li_icpms.iloc[te].to_numpy(float)>=thr).astype(int)
            weights=site_weights(contract,tr) if arm=="expanded_inverse_site_weights" else None
            score=fit_classifier(predictors.iloc[tr][columns],y_tr,
                                 predictors.iloc[te][columns],weights=weights)
            fold_metrics=score_metrics(y_te,score)
            print("ABLATION",arm,"fold",k,"train",len(tr),"historical_test",len(te),
                  "auc",round(fold_metrics["auc"],4) if fold_metrics["auc"] is not None else None,
                  "ap",round(fold_metrics["ap"],4) if fold_metrics["ap"] is not None else None,
                  flush=True)
            for idx,y,s in zip(te,y_te,score):
                arm_rows.append({"arm":arm,"fold":k,"test_index":int(idx),
                    "sample_id":str(contract.sample_id.iloc[idx]),
                    "spatial_group":str(contract.spatial_group.iloc[idx]),
                    "site_screening_id":str(contract.site_screening_id.iloc[idx]),
                    "latitude":float(contract.latitude.iloc[idx]),
                    "li_ppm":float(contract.Li_icpms.iloc[idx]),
                    "fold_li_threshold_ppm":float(thr),
                    "y_true":int(y),"score":float(s),
                    "n_train":len(tr),"n_features":len(columns)})
        df=pd.DataFrame(arm_rows).sort_values("test_index")
        if df.test_index.tolist()!=list(range(HISTORICAL_N)):
            raise AssertionError("Each arm must score the SAME 706 historical cases once")
        allrows.extend(arm_rows)
        df.to_csv(output/("oof_"+arm+".csv"),index=False)
        completed[arm]={"seconds":time.monotonic()-t0,**score_metrics(df.y_true,df.score)}
        (output/"run_progress.json").write_text(json.dumps({
            "state":"running","finished_arms":completed,"expected_arms":list(ARMS)
        },indent=2),encoding="utf-8")
        print("ARM_DONE",arm,"pooled",completed[arm],flush=True)
    pred=pd.DataFrame(allrows)
    pred.to_csv(output/"all_oof_arms.csv",index=False)
    rows=[]
    for arm,df in pred.groupby("arm",sort=False):
        for fold,sub in df.groupby("fold",sort=False):
            rows.append({"arm":arm,"fold":int(fold),"n_train":int(sub.n_train.iloc[0]),
                         "n_features":int(sub.n_features.iloc[0]),**score_metrics(sub.y_true,sub.score)})
    pd.DataFrame(rows).to_csv(output/"fold_metrics.csv",index=False)
    rank=[]
    for arm,info in completed.items():
        reference=completed["expanded_all"] if arm!="legacy_all" else completed["legacy_all"]
        rank.append({"arm":arm,**info,
            "delta_auc_vs_expanded":info["auc"]-completed["expanded_all"]["auc"],
            "delta_ap_vs_expanded":info["ap"]-completed["expanded_all"]["ap"],
            "delta_brier_vs_expanded":info["brier"]-completed["expanded_all"]["brier"]})
    ranking=pd.DataFrame(rank)
    ranking.to_csv(output/"exploratory_comparison.csv",index=False)
    # Fixed baseline to augmented contrast and other single-factor contrasts.
    pairs=[
        ("legacy_all","expanded_all"),
        ("expanded_all","expanded_buffer_5km"),
        ("expanded_all","expanded_buffer_10km"),
        ("expanded_all","expanded_exclude_shared_scenes"),
        ("expanded_all","expanded_drop_mixed_train_sites"),
        ("expanded_all","expanded_raw_spectral_only"),
        ("expanded_all","expanded_inverse_site_weights"),
    ]
    boots=[]
    for ref,arm in pairs:
        boots.append(bootstrap_group_delta(pred,ref,arm,repeats=320))
    pd.DataFrame(boots).to_csv(output/"cluster_bootstrap_exploratory.csv",index=False)
    if nested:
        choices=("legacy_all","expanded_buffer_5km")
        nested_records=[]
        for arm in choices:
            print("NESTED_DECISION_BEGIN",arm,flush=True)
            rules=nested_decision_probe(contract,predictors,folds,arm=arm)
            rulelookup=rules.set_index("fold").decision_threshold
            subset=pred[pred.arm==arm]
            for fold,sub in subset.groupby("fold"):
                scores=sub.score.to_numpy(float)
                metrics=score_metrics(sub.y_true,scores,float(rulelookup.loc[fold]))
                nested_records.append({"arm":arm,"fold":int(fold),
                    "train_only_decision_threshold":float(rulelookup.loc[fold]),
                    **{("decision_"+k):v for k,v in metrics.items()}})
            rules.to_csv(output/("inner_frozen_thresholds_"+arm+".csv"),index=False)
        pd.DataFrame(nested_records).to_csv(output/"inner_threshold_outer_assessment.csv",index=False)
    summary={"state":"completed","results":completed,
             "same_706_held_out":True,"no_activation":True,
             "bootstrap_pairs":[{"reference":r,"arm":v} for r,v in pairs],
             "nested_probe":bool(nested)}
    (output/"experiment_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (output/"run_progress.json").write_text(
        json.dumps({"state":"completed","finished_arms":completed,
                    "nested_probe":bool(nested)},indent=2),encoding="utf-8")
    print("PHASE_C_COMPLETE",output,flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--nested-probe",action="store_true")
    args=parser.parse_args(argv)
    dest=Path(args.output)
    if not dest.is_absolute():dest=PROJECT_ROOT/dest
    run(PROJECT_ROOT,dest,nested=args.nested_probe)


if __name__=="__main__":
    main()
