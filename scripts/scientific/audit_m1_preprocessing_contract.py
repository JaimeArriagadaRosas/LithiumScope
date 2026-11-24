"""Verify preprocessing and predeclared feature variants on frozen M1; never fit predictors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.evaluation.baseline import baseline_splits
from lithiumscope.model_1.training.preprocessing.tabular import build_preprocessor
from lithiumscope.model_1.training.preprocessing.variants import feature_variant, ABLATIONS
from lithiumscope.model_1.training.preprocessing.evidence import preprocessing_evidence

EXPECTED="94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94"


def run(dataset,output):
    if file_sha256(dataset)!=EXPECTED:
        raise ValueError("Expected frozen M1 combined dataset")
    p=prepare_training_data(dataset)
    config=load_config("model_1")
    splits,strategy=baseline_splits(p,config)
    if len(p.y)!=3420 or p.groups.nunique()!=244 or strategy!="group_kfold_spatial":
        raise ValueError("Unexpected frozen protocol")
    rows=[];details=[];features=[]
    for variant in ABLATIONS:
        q=feature_variant(p,variant)
        assert q.y is p.y and q.groups is p.groups and q.x.index.equals(p.x.index)
        for fold,(tr,te) in enumerate(splits,1):
            for col in q.x:
                x=q.x.iloc[tr][col]
                features.append({"variant":variant,"fold":fold,"feature":col,
                    "train_available":int(x.notna().sum()),"train_unique":int(x.nunique()),
                    "train_variance":float(x.var(ddof=0)) if pd.api.types.is_numeric_dtype(x) else None,
                    "test_available":int(q.x.iloc[te][col].notna().sum())})
        # All families use same raw reference and splits; ablations audit dimensions with SVM.
        algorithms=config["model"]["algorithms"] if variant=="reference" else ["svm"]
        for algorithm in algorithms:
            policies=["legacy","stable_v1"] if variant=="reference" else ["stable_v1"]
            if algorithm=="tabnet":
                policies.append("tabnet_mask_v1")
            for policy in policies:
                r=feature_variant(q,"reference",preprocessing_policy=policy)
                for fold,(tr,te) in enumerate(splits,1):
                    a,b=r.x.iloc[tr],r.x.iloc[te]
                    transformer=build_preprocessor(r.schema,algorithm).fit(a)
                    model=Pipeline([("preprocess",transformer)])
                    before=pickle.dumps(transformer)
                    evidence=preprocessing_evidence(model,a,b)
                    if pickle.dumps(transformer)!=before:
                        raise ValueError("TEST transform altered TRAIN-fitted preprocessing")
                    details.append({"algorithm":algorithm,"variant":variant,"policy":policy,"fold":fold,**evidence})
                    rows.append({"algorithm":algorithm,"variant":variant,"policy":policy,"fold":fold,
                        "raw_features":r.x.shape[1],"transformed_features":evidence["feature_count"],
                        "train_rows":len(a),"test_rows":len(b),
                        "train_all_missing":len(evidence["train_all_missing_columns"])})
    if file_sha256(dataset)!=EXPECTED:
        raise ValueError("Dataset changed")
    output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(output/"fold_preprocessing.csv",index=False)
    pd.DataFrame(features).to_csv(output/"feature_availability.csv",index=False)
    (output/"fold_transform_details.json").write_text(json.dumps(details,indent=2,allow_nan=False),encoding="utf-8")
    report={"samples":len(p.y),"groups":p.groups.nunique(),"folds":len(splits),"variant_dimensions":{name:feature_variant(p,name).x.shape[1] for name in ABLATIONS},
            "preprocessor_fits":len(rows),"algorithms":config["model"]["algorithms"],"test_transform_mutation":False,
            "dataset_sha256":EXPECTED,"predictor_fit":False,"performance_measured":False}
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    run(args.dataset,args.output)
