"""Auxiliary high-Li diagnosis for a continuous regression, not a classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(a,b):
    return float(a/b) if b else float("nan")


def _confusion(y,p):
    tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum())
    tp=int(((y==1)&(p==1)).sum())
    sensitivity=_ratio(tp,tp+fn)
    specificity=_ratio(tn,tn+fp)
    return {"n":len(y),"tn":tn,"fp":fp,"fn":fn,"tp":tp,
            "positive_support":tp+fn,"negative_support":tn+fp,
            "sensitivity":sensitivity,"specificity":specificity,
            "balanced_accuracy":(sensitivity+specificity)/2,
            "precision":_ratio(tp,tp+fp),"f1":_ratio(2*tp,2*tp+fp+fn),
            "accuracy":_ratio(tp+tn,len(y))}


def high_lithium_diagnostics(oof):
    required={"row_position","outer_fold","fold_train_q75_ppm","Li_icpms_real",
              "Li_icpms_predicted","observed_high","predicted_high"}
    if not required<=set(oof) or oof.empty or oof.row_position.duplicated().any():
        raise ValueError("Auxiliary high-Li OOF table is incomplete or duplicated")
    y=oof.Li_icpms_real.to_numpy(float)
    p=oof.Li_icpms_predicted.to_numpy(float)
    cut=oof.fold_train_q75_ppm.to_numpy(float)
    if not np.isfinite(np.c_[y,p,cut]).all() or (y<0).any() or (cut<0).any():
        raise ValueError("Invalid high-Li regression inputs")
    truth=(y>=cut).astype(int);pred=(p>=cut).astype(int)
    if not np.array_equal(truth,oof.observed_high) or not np.array_equal(pred,oof.predicted_high):
        raise ValueError("High-Li flags disagree with fold TRAIN cutoff")
    rows=[];strata=[]
    for fold,part in oof.groupby("outer_fold",sort=True):
        limits=part.fold_train_q75_ppm.unique()
        if len(limits)!=1:
            raise ValueError("High-Li threshold must be constant within a fold")
        rows.append({"scope":"outer_fold","outer_fold":fold,"threshold_ppm":float(limits[0]),
                     **_confusion(part.observed_high,part.predicted_high)})
    rows.append({"scope":"pooled_fold_specific_endpoints","outer_fold":None,"threshold_ppm":None,**_confusion(truth,pred)})
    if "source_dataset" in oof:
        groups=oof.groupby(["source_dataset","observed_high"],dropna=False,sort=True)
    else:
        groups=oof.assign(source_dataset="unknown").groupby(["source_dataset","observed_high"],sort=True)
    for (source,high),part in groups:
        residual=part.Li_icpms_predicted-part.Li_icpms_real
        strata.append({"source":str(source),"observed_high":int(high),"n":len(part),
                       "bias_ppm":float(residual.mean()),"mae_ppm":float(residual.abs().mean()),
                       "rmse_ppm":float(np.sqrt(np.mean(residual**2)))})
    return pd.DataFrame(rows),pd.DataFrame(strata)


def save_high_lithium_diagnostics(oof,table_path,strata_path):
    confusion,strata=high_lithium_diagnostics(oof)
    confusion.to_csv(table_path,index=False)
    strata.to_csv(strata_path,index=False)
    return confusion,strata
