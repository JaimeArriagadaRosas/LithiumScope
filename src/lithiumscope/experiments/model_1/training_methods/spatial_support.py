"""TRAIN-only chemical support diagnostics and conservative RF fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from lithiumscope.experiments.data_lab.screening.candidates import CHEM
from lithiumscope.experiments.model_1.data_expansion.identity import multivariate_support

POLICIES=("ridge_disagreement","ridge_supported","ridge_high_support")
PRIMARY="ridge_high_support"


def support(train,y,groups,evaluation,prediction):
    """No evaluation targets. High support requires two nearby TRAIN groups."""
    if (not train.index.is_unique or not evaluation.index.is_unique
            or set(train.index)&set(evaluation.index)
            or not train.index.equals(y.index) or not train.index.equals(groups.index)
            or groups.isna().any() or not np.isfinite(y).all() or y.lt(0).any()):
        raise ValueError("Invalid support TRAIN")
    p=np.asarray(prediction,float)
    if p.shape!=(len(evaluation),) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError("Invalid support prediction")
    diag=multivariate_support(evaluation,train,list(CHEM))
    b=train[list(CHEM)].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    e=evaluation[list(CHEM)].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    median=b.median()
    spread=b.quantile(.75)-b.quantile(.25)
    cols=median.index[median.notna()&spread.gt(0)&b.notna().mean().ge(.5)].tolist()
    counts=np.zeros(len(e),dtype=int)
    if len(cols)>=5 and len(b)>=2:
        def matrix(f):
            v=f[cols]
            return np.column_stack([v.sub(median[cols]).div(spread[cols]).fillna(0.),v.isna()])
        nn=NearestNeighbors(n_neighbors=min(15,len(b))).fit(matrix(b))
        dist,ix=nn.kneighbors(matrix(e))
        radius=2.*float(diag.chemistry_nn_q95.iloc[0])
        # Only actual TRAIN labels and groups may establish concentration support.
        labels=y.to_numpy(float)
        gg=groups.to_numpy(str)
        for i,(d,j) in enumerate(zip(dist,ix,strict=True)):
            nearby=j[(d<=radius)&(labels[j]>=max(40.,.5*p[i]))]
            counts[i]=len(set(gg[nearby]))
    diag["high_support_groups"]=counts
    diag["high_supported"]=diag.chemistry_nn_supported&(counts>=2)
    return diag


def apply_policies(ridge,rf,diag):
    r,f=map(lambda v:np.asarray(v,float),(ridge,rf))
    if (r.ndim!=1 or r.shape!=f.shape or len(r)!=len(diag)
            or not np.isfinite(np.c_[r,f]).all() or (r<0).any() or (f<0).any()):
        raise ValueError("Invalid fallback inputs")
    disagreement=(r>40.)&(r>2.*f)&(r-f>20.)
    flags={
        "ridge_disagreement":disagreement,
        "ridge_supported":disagreement&~diag.chemistry_nn_supported.to_numpy(bool),
        "ridge_high_support":disagreement&~diag.high_supported.to_numpy(bool),
    }
    return {k:np.where(mask,f,r) for k,mask in flags.items()},flags
