"""Fold-local mean and median statistical references for regression."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.scientific_checks import assert_group_isolation, assert_partition_coverage
from lithiumscope.model_1.evaluation.metrics import regression_metrics
from lithiumscope.model_1.validation.spatial import materialize_regression_splits


def baseline_splits(prepared, config):
    validation=config["validation"]
    groups=getattr(prepared,"groups",None) if validation.get("prefer_spatial_groups",False) else None
    require=bool(validation.get("prefer_spatial_groups",False) and not validation.get("allow_random_fallback",False))
    splits=getattr(prepared,"outer_splits",None)
    if splits is None:
        splits,strategy=materialize_regression_splits(prepared.x,prepared.y,groups,
            int(validation["outer_folds"]),int(validation["random_seed"]),require_groups=require)
    else:
        strategy=getattr(prepared,"validation_strategy",None) or "unknown"
    splits=list(splits)
    assert_partition_coverage(splits,len(prepared.y))
    if require and (groups is None or strategy!="group_kfold_spatial"):
        raise ValueError("Baseline requires spatial validation")
    if groups is not None and strategy=="group_kfold_spatial":
        for train,test in splits:
            assert_group_isolation(train,test,groups)
    return splits,strategy


def evaluate_baseline(prepared, config, *, strategy="mean"):
    if strategy not in {"mean","median"}:
        raise ValueError("Unsupported baseline")
    splits,validation_strategy=baseline_splits(prepared,config)
    predictions=np.full(len(prepared.y),np.nan)
    rows=[]
    y=np.asarray(prepared.y,dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Baseline target must be finite")
    for fold,(train,test) in enumerate(splits,1):
        center=float(np.mean(y[train]) if strategy=="mean" else np.median(y[train]))
        predictions[test]=center
        metrics=regression_metrics(y[test],predictions[test]).to_dict()
        rows.append({"fold":fold,"algorithm":"baseline_"+strategy,
                     "n_train":len(train),"n_test":len(test),"train_reference_ppm":center,
                     **metrics,"validation_strategy":validation_strategy})
    table=pd.DataFrame(rows)
    overall=regression_metrics(y,predictions).to_dict()
    row={"algorithm":"baseline_"+strategy,"label":"Baseline — "+("media" if strategy=="mean" else "mediana")+" de TRAIN",
         "elapsed_seconds":0.0,"status":"ok","is_baseline":True,
         **{f"{m}_mean":float(table[m].mean(skipna=False)) for m in
            ("rmse","mae","r2","median_ae","explained_variance")},
         **{f"{m}_std":float(table[m].std(ddof=0,skipna=False)) for m in ("rmse","mae","r2")},
         **{f"oof_{m}":v for m,v in overall.items()}}
    return row,table,predictions


def evaluate_mean_baseline(prepared,config):
    """Stable historical entry point."""
    return evaluate_baseline(prepared,config,strategy="mean")


def evaluate_baselines(prepared,config):
    from lithiumscope.model_1.evaluation.oof_evidence import regression_oof_table
    splits,_=baseline_splits(prepared,config)
    results=[evaluate_baseline(prepared,config,strategy=name) for name in ("mean","median")]
    tables=pd.concat([r[1] for r in results],ignore_index=True)
    predictions=pd.concat([regression_oof_table(prepared,r[2],splits).assign(algorithm=r[0]["algorithm"])
                           for r in results],ignore_index=True)
    return [r[0] for r in results],tables,predictions
