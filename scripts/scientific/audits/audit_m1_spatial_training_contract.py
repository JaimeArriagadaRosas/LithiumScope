"""Audit M1 spatial early stopping and synthetic stress on frozen observations, no fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.evaluation.baseline import baseline_splits
from lithiumscope.model_1.diagnostics.robustness import SCENARIOS, SEED_OFFSETS, POLICY, apply_stress
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.training.preprocessing.spatial_tabnet import early_stopping_partition
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES,TRACE_ELEMENTS,DERIVED
from lithiumscope.model_1.validation.spatial import materialize_regression_splits

EXPECTED="94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94"


def audit_split(p, allowed, forbidden, *, stage, outer, inner, seed):
    x,y,g=p.x.iloc[allowed],p.y.iloc[allowed],p.groups.iloc[allowed]
    tr,va,_=early_stopping_partition(x,y,g,seed=seed,validation_fraction=.15)
    train,valid=allowed[tr],allowed[va]
    assert not np.intersect1d(train,valid).size
    assert not np.intersect1d(np.r_[train,valid],forbidden).size
    assert np.array_equal(np.sort(np.r_[train,valid]),np.sort(allowed))
    tg,vg=set(p.groups.iloc[train]),set(p.groups.iloc[valid])
    fg=set(p.groups.iloc[forbidden])
    if tg&vg or (tg|vg)&fg:
        raise ValueError("Spatial group leakage in early stopping")
    return {"stage":stage,"outer_fold":outer,"inner_fold":inner,"n_available":len(allowed),
            "n_early_train":len(train),"n_early_validation":len(valid),"n_forbidden":len(forbidden),
            "train_groups":len(tg),"validation_groups":len(vg),"forbidden_groups":len(fg),
            "validation_fraction":len(valid)/len(allowed),"early_train_positions":train.tolist(),
            "early_validation_positions":valid.tolist(),"forbidden_positions":forbidden.tolist()}


def check_scenario(x,schema,scenario,magnitude,seed):
    stress=apply_stress(x,schema,scenario,magnitude,seed)
    out=stress.frame
    unaffected=set(x)-(
        set(OXIDES+DERIVED) if scenario=="oxide_relative" else set(TRACE_ELEMENTS) if scenario=="trace_relative"
        else set(OXIDES+TRACE_ELEMENTS+DERIVED) if scenario=="missingness" else {"Latitude","Longitude"})
    cols=[c for c in x if c in unaffected]
    pd.testing.assert_frame_equal(x[cols],out[cols])
    oxide=[c for c in OXIDES if c in x]
    if scenario=="oxide_relative":
        np.testing.assert_allclose(x[oxide].sum(axis=1),out[oxide].sum(axis=1),rtol=0,atol=1e-10)
        assert x[oxide].isna().equals(out[oxide].isna())
    if scenario in {"oxide_relative","missingness"}:
        calculated=add_geochemical_features(out,version=schema.preparation_version)
        cols=[c for c in DERIVED if c in x]
        pd.testing.assert_frame_equal(calculated[cols],out[cols])
    if scenario=="missingness":
        assert (x[oxide].isna()<=out[oxide].isna()).all().all()
    if scenario=="spatial_m":
        usable=x[["Latitude","Longitude"]].notna().all(axis=1)
        a=np.deg2rad(x.loc[usable,["Latitude","Longitude"]])
        b=np.deg2rad(out.loc[usable,["Latitude","Longitude"]])
        distances=haversine_distances(a,b).diagonal()*6371008.8
        np.testing.assert_allclose(distances,magnitude,rtol=0,atol=.001)
    return {k:v for k,v in vars(stress).items() if k!="frame"}


def run(dataset,output):
    if file_sha256(dataset)!=EXPECTED:
        raise ValueError("Frozen M1 data changed")
    config=load_config("model_1")
    p=prepare_training_data(dataset)
    seed=config["validation"]["random_seed"]
    splits,strategy=baseline_splits(p,config)
    if len(p.y)!=3420 or len(splits)!=5 or strategy!="group_kfold_spatial":
        raise ValueError("Unexpected frozen M1 protocol")
    rows=[]
    stress_rows=[]
    for k,(tr,te) in enumerate(splits,1):
        rows.append(audit_split(p,tr,te,stage="outer_fit",outer=k,inner=None,seed=seed))
        inner,_=materialize_regression_splits(p.x.iloc[tr],p.y.iloc[tr],p.groups.iloc[tr],5,seed+k,require_groups=True)
        for j,(itr,iva) in enumerate(inner,1):
            rows.append(audit_split(p,tr[itr],np.r_[te,tr[iva]],stage="inner_trial_fit",outer=k,inner=j,seed=seed))
        x=p.x.iloc[te]
        for scenario,magnitudes in SCENARIOS.items():
            for magnitude in magnitudes:
                for offset in SEED_OFFSETS:
                    s=seed+10000+k*100+offset
                    stress_rows.append({"fold":k,"n_test":len(te),"policy":POLICY,"scenario":scenario,
                                        "magnitude":magnitude,"seed":s,
                                        **check_scenario(x,p.schema,scenario,magnitude,s)})
    all_indices=np.arange(len(p.y))
    final,_=materialize_regression_splits(p.x,p.y,p.groups,5,seed+5000,require_groups=True)
    for j,(tr,va) in enumerate(final,1):
        rows.append(audit_split(p,tr,va,stage="final_optuna_trial_fit",outer=None,inner=j,seed=seed))
    rows.append(audit_split(p,all_indices,np.array([],dtype=int),stage="final_fit",outer=None,inner=None,seed=seed))
    if file_sha256(dataset)!=EXPECTED:
        raise ValueError("Frozen M1 data changed during audit")
    output.mkdir(parents=True,exist_ok=False)
    (output/"spatial_early_stopping_partitions.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    table=pd.DataFrame(stress_rows)
    table.to_csv(output/"stress_input_audit.csv",index=False)
    summary={"dataset_sha256":EXPECTED,"observations":len(p.y),"spatial_groups":p.groups.nunique(),
             "early_stopping_partitions":len(rows),"minimum_train_rows":min(r["n_early_train"] for r in rows),
             "minimum_validation_rows":min(r["n_early_validation"] for r in rows),
             "validation_fraction_range":[min(r["validation_fraction"] for r in rows),max(r["validation_fraction"] for r in rows)],
             "stress_scenarios":len(table),"preexisting_invalid_cells_by_scenario":table.groupby("scenario").preexisting_invalid_cells.sum().to_dict(),
             "rejected_cells_by_scenario":table.groupby("scenario").rejected_cells.sum().to_dict(),
             "all_input_invariants_pass":True,"model_fit":False,"predictive_performance_measured":False}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,default=int),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=int))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.dataset,args.output)
