"""Run predictor/preprocessor and nested tuning ablations for M1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sklearn.model_selection import GroupKFold

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.shared.design import budget_arms, choose_algorithms
from lithiumscope.experiments.shared.paired_suite import Arm, run_paired_suite
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.training.factory import build_pipeline, get_algorithm
from lithiumscope.model_1.training.optimization.optimizer import optimize_with_optuna
from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
from lithiumscope.model_1.training.preprocessing.variants import ABLATIONS, feature_variant
from lithiumscope.model_1.validation.spatial import materialize_regression_splits
from lithiumscope.resources.detection.device import detect_device


from lithiumscope.model_1.methodology.feature_comparison import build_feature_arms


def run(*, output: Path, study="features", algorithms=None, execute=False, prefer_gpu=False,
        review_ledger=None, evidence_root=None):
    config=load_config("model_1")
    prepared=prepare_training_data(local_training_source("model_1"))
    validation=config["validation"]
    selected=choose_algorithms(config["model"]["algorithms"],algorithms)
    seed=int(validation["random_seed"])
    splits,_=materialize_regression_splits(prepared.x,prepared.y,prepared.groups,
        int(validation["outer_folds"]),seed,require_groups=True)
    if study=="features":
        arms=build_feature_arms(prepared,selected,seed=seed)
    elif study=="source_conflicts":
        if review_ledger is None or evidence_root is None:
            raise ValueError("Documentary review ledger and evidence root required")
        import json
        import pandas as pd
        from lithiumscope.model_1.validation.conflict_review import conflict_arms
        arms,audit=conflict_arms(prepared,pd.read_csv(review_ledger,dtype=str,keep_default_na=False),selected,
                                 evidence_root=evidence_root,seed=seed)
        preparation=output.parent/(output.name+"_preparation")
        preparation.mkdir(parents=True,exist_ok=False)
        (preparation/"documentary_review.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    elif study=="budget":
        arms=budget_arms(selected,prepared.x)
    else:
        raise ValueError("Unknown M1 study")
    device=detect_device(prefer_gpu=prefer_gpu)
    def fit(arm,fold,x,y,groups,raw):
        del raw
        policy=arm.policy
        base=feature_variant(prepared,policy.get("variant","reference"),
            preprocessing_policy=policy.get("preprocessing",prepared.schema.preprocessing_policy))
        model_seed=policy["seed"]+fold
        def factory(params):
            return build_pipeline(arm.algorithm,device,model_seed,base.schema,params,
                                  target_transform="identity",final_fit=False)
        params={}
        if "attempts" in policy:
            params=optimize_with_optuna(factory,get_algorithm(arm.algorithm).module.optuna_space,x,y,
                GroupKFold(n_splits=int(validation["inner_folds"])),
                n_trials=policy["attempts"],storage_path=output/"studies"/arm.name/f"fold_{fold}.db",
                study_name=arm.name,random_seed=model_seed,pruning_enabled=policy["pruning"],groups=groups)
        return fit_regression_estimator(factory(params),x,y,groups)
    return run_paired_suite(arms=arms,y=prepared.y,groups=prepared.groups,splits=splits,
        kind="regression",output=output,fit=fit,predict=lambda m,x:m.predict(x),execute=execute)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",choices=("features","budget","source_conflicts","review_register"),default="features")
    parser.add_argument("--algorithm",action="append")
    parser.add_argument("--review-ledger",type=Path)
    parser.add_argument("--evidence-root",type=Path)
    parser.add_argument("--execute",action="store_true",help="Explicitly fit diagnostic models")
    parser.add_argument("--prefer-gpu",action="store_true")
    args=parser.parse_args()
    output=RESULTS_DIR/"model_1"/"methodology"/("paired_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8])
    if args.study=="review_register":
        from lithiumscope.model_1.validation.conflict_review import review_register
        output.mkdir(parents=True,exist_ok=False)
        review_register(prepare_training_data(local_training_source("model_1"))).to_csv(output/"review_register.csv",index=False)
        print(output)
        return
    result=run(output=output,study=args.study,algorithms=args.algorithm,execute=args.execute,prefer_gpu=args.prefer_gpu,
               review_ledger=args.review_ledger,evidence_root=args.evidence_root)
    print(f"M1 {args.study}: {result['state']} — {output}")


if __name__=="__main__":
    main()
