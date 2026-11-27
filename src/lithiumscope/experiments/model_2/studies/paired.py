"""Controlled M2 representation, QA, calibration and tuning comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from sklearn.model_selection import GroupKFold

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.shared.design import budget_arms, choose_algorithms
from lithiumscope.experiments.shared.paired_suite import Arm, run_paired_suite
from lithiumscope.model_2.training.calibration.operating import fit_policy
from lithiumscope.model_2.training.calibration.policy import calibration_policy
from lithiumscope.model_2.training.execution.competition_prepare_scientific_data import _prepare_scientific_data
from lithiumscope.model_2.training.factory import create_model, get_algorithm
from lithiumscope.model_2.training.optimization.optimizer import optimize_classifier_with_optuna
from lithiumscope.model_2.training.optimization.search_policy import parameter_space_for
from lithiumscope.resources.detection.device import detect_device


def representation_arms(x, algorithms, mask, *, seed, mode, physical=None):
    if x.shape[1]!=170 or len([c for c in x if c.startswith("nir_swir22_nd_")])!=6:
        raise ValueError("Verified v4 170-column representation required")
    legacy=x.iloc[:,:126]
    removed=[c for c in x if c.startswith("nir_swir22_nd_")]
    for c in removed:
        other=c.replace("nir_swir22_nd_","nbr_",1)
        if other not in x or not np.array_equal(x[c],x[other]):
            raise ValueError("Expected exact redundant spectral descriptors")
    dedup=x.drop(columns=removed)
    if physical is not None and (not physical.index.equals(x.index)
            or list(physical.columns)!=list(dedup.columns) or not np.isfinite(physical.to_numpy(float)).all()):
        raise ValueError("Physical representation must retain all comparable test rows")
    arms=[]
    for algorithm in algorithms:
        ref=f"{algorithm}_v4"
        policy={"seed":seed,"calibration":mode}
        arms.extend([
            Arm(ref,algorithm,None,x,{**policy,"representation":"v4_170","quality":"common_QA"},mask),
            Arm(f"{algorithm}_legacy",algorithm,ref,legacy,{**policy,"representation":"legacy_126","quality":"common_QA"},mask),
            Arm(f"{algorithm}_deduplicated",algorithm,ref,dedup,{**policy,"representation":"v4_164","quality":"common_QA"},mask),
        ])
        if physical is not None:
            arms.append(Arm(f"{algorithm}_physical",algorithm,f"{algorithm}_deduplicated",physical,
                            {**policy,"representation":"v5_physical_164","quality":"common_QA"},mask))
    # Each arm must reference a primary comparison; physical uses dedup as primary.
    # Use a separate identical dedup reference rather than chaining references.
    if physical is not None:
        corrected=[]
        for arm in arms:
            if arm.policy["representation"]=="v5_physical_164":
                name=f"{arm.algorithm}_dedup_reference"
                corrected.append(Arm(name,arm.algorithm,None,dedup,
                                    {"seed":seed,"calibration":mode,"representation":"v4_164","quality":"common_QA"},mask))
                arm=Arm(arm.name,arm.algorithm,name,arm.x,arm.policy,mask)
            corrected.append(arm)
        arms=corrected
    return arms


def policy_arms(x, algorithms, mask, *, seed, mode):
    arms=[]
    for algorithm in algorithms:
        ref=f"{algorithm}_legacy_qa"
        arms.extend([
            Arm(ref,algorithm,None,x,{"seed":seed,"calibration":"legacy","quality":"QA"},mask),
            Arm(f"{algorithm}_sigmoid_qa",algorithm,ref,x,{"seed":seed,"calibration":"spatial_sigmoid","quality":"QA"},mask),
            Arm(f"{algorithm}_legacy_all_train",algorithm,ref,x,
                {"seed":seed,"calibration":"legacy","quality":"all_train"},None),
        ])
    return arms


def run(*, output: Path, study="representation", algorithms=None, execute=False, prefer_gpu=False,
        physical_directory=None, temporal_manifest=None, temporal_interval=None, temporal_root=None):
    config=load_config("model_2")
    model,training,validation=config["model"],config["training"],config["validation"]
    selected=choose_algorithms(model["algorithms"],algorithms)
    source=local_training_source("model_2")
    preparation=output.parent/(output.name+"_preparation")
    preparation.mkdir(parents=True,exist_ok=False)
    _,frame,groups,_,_,mask,folds,_,x,_,_=_prepare_scientific_data(
        config,SimpleNamespace(manifests=preparation),model.get("fixed_threshold_ppm"),
        training["lithium_column"],source,float(model["high_lithium_quantile"]),
        int(validation["folds"]),int(model["random_seed"]),training["spatial_group_column"])
    seed=int(model["random_seed"])
    mode=calibration_policy(config["optimization"])
    if study=="representation":
        physical=None
        if physical_directory is not None:
            from lithiumscope.experiments.model_2.representation.physical import load_physical_comparison
            physical,mask=load_physical_comparison(physical_directory,source,frame,x,mask)
        arms=representation_arms(x,selected,mask,seed=seed,mode=mode,physical=physical)
    elif study=="texture":
        from lithiumscope.experiments.model_2.representation.texture import texture_arms
        physical = physical_directory is not None
        if physical:
            from lithiumscope.experiments.model_2.representation.physical import load_physical_comparison
            x, mask = load_physical_comparison(physical_directory, source, frame, x, mask)
        arms = texture_arms(x, selected, mask, seed=seed, mode=mode, physical=physical)
    elif study=="temporal":
        if temporal_manifest is None or temporal_interval is None or temporal_root is None:
            raise ValueError("Temporal study requires explicit staged manifest, root and fixed interval")
        import json
        import pandas as pd
        from lithiumscope.experiments.model_2.temporal.comparison import temporal_arms
        cohort=pd.read_csv(source,dtype={"sample_id":str})
        if (len(cohort)!=len(frame) or not np.array_equal(cohort[training["lithium_column"]].to_numpy(),frame[training["lithium_column"]].to_numpy())
                or not np.array_equal(cohort[training["spatial_group_column"]].astype(str).to_numpy(),groups.astype(str).to_numpy())):
            raise ValueError("Temporal cohort must align with canonical labels and groups")
        cohort.index=frame.index
        arms,audit=temporal_arms(cohort,pd.read_csv(temporal_manifest,dtype={"sample_id":str}),selected,interval=temporal_interval,
                                root=temporal_root,seed=seed,mode=mode,mask=mask)
        (preparation/"temporal_acquisition_evidence.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    elif study=="policy":
        arms=policy_arms(x,selected,mask,seed=seed,mode=mode)
    elif study=="budget":
        arms=budget_arms(selected,x,eligible=mask)
    else:
        raise ValueError("Unknown M2 study")
    device=detect_device(prefer_gpu=prefer_gpu)
    def fit(arm,fold,train,labels,train_groups,raw):
        policy=arm.policy
        model_seed=policy["seed"]+fold
        inner=int(validation["inner_folds"])
        params={}
        if "attempts" in policy:
            def factory(candidate):
                return create_model(arm.algorithm,device,model_seed,candidate,calibrated=False)
            splits=list(GroupKFold(n_splits=inner).split(train,labels,train_groups))
            params=optimize_classifier_with_optuna(factory,
                parameter_space_for(arm.algorithm,get_algorithm(arm.algorithm),config["optimization"]),
                train,labels,splits,n_trials=policy["attempts"],random_seed=model_seed,
                storage_path=output/"studies"/arm.name/f"fold_{fold}.db",study_name=arm.name,
                pruning_enabled=policy["pruning"],lithium_values=raw,
                label_quantile=float(model["high_lithium_quantile"]),
                fixed_threshold_ppm=model.get("fixed_threshold_ppm"),groups=train_groups)
        return fit_policy(arm.algorithm,device,model_seed,params,train,labels,train_groups,
                          inner,policy.get("calibration",mode))
    return run_paired_suite(arms=arms,y=frame[training["lithium_column"]],groups=groups,
        splits=[(f.train_idx,f.test_idx) for f in folds],kind="classification",output=output,
        endpoints=[f.label_threshold_ppm for f in folds],fit=fit,
        predict=lambda m,x:m.predict_proba(x)[:,1],execute=execute)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",choices=("representation","texture","temporal","policy","budget"),default="representation")
    parser.add_argument("--algorithm",action="append")
    parser.add_argument("--physical-directory",type=Path)
    parser.add_argument("--temporal-manifest",type=Path)
    parser.add_argument("--temporal-interval")
    parser.add_argument("--temporal-root",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--prefer-gpu",action="store_true")
    args=parser.parse_args()
    output=RESULTS_DIR/"model_2"/"methodology"/("paired_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8])
    result=run(output=output,study=args.study,algorithms=args.algorithm,execute=args.execute,
               prefer_gpu=args.prefer_gpu,physical_directory=args.physical_directory,
               temporal_manifest=args.temporal_manifest,temporal_interval=args.temporal_interval,temporal_root=args.temporal_root)
    print(f"M2 {args.study}: {result['state']} — {output}")


if __name__=="__main__":
    main()
