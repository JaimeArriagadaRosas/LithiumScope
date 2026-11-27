"""Canonical, serializable HGB/XGB bases and bounded Ridge residual correction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lithiumscope.experiments.model_1.training_methods.losses import loss_params
from lithiumscope.experiments.model_1.training_methods.boosting import build_method_pipeline
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES, TRACE_ELEMENTS, SPATIAL_CANDIDATES, DERIVED, CATEGORICAL
from lithiumscope.resources.manager import estimator_thread_scope

POLICIES=("hgb_canonical","xgb_canonical","ridge_canonical")


def canonical_columns(schema):
    cols=list(schema.numeric)+list(schema.categorical)
    if cols!=OXIDES+TRACE_ELEMENTS+SPATIAL_CANDIDATES+DERIVED+CATEGORICAL or len(cols)!=33:
        raise ValueError("Expected frozen 33-column M1 contract")
    return cols


def fit_bases(x,y,schema,device):
    cols=canonical_columns(schema)
    if (list(x.columns)!=cols or not x.index.is_unique or not x.index.equals(y.index)
            or not len(y) or not np.isfinite(y).all() or y.lt(0).any()):
        raise ValueError("Invalid canonical TRAIN")
    h=build_pipeline("hist_gradient_boosting",device,42,schema,loss_params("squared_error"))
    h.set_params(preprocess__numeric="passthrough")
    z=build_method_pipeline(schema,device,"xgb_regularized")
    with estimator_thread_scope():
        h.fit(x,y,model__sample_weight=np.ones(len(y)))
        z.fit(x,y)
    return h,z


def predict_bases(bases,x,schema):
    if list(x.columns)!=canonical_columns(schema) or not x.index.is_unique:
        raise ValueError("Canonical inference schema differs")
    with estimator_thread_scope():
        pred=[np.maximum(np.asarray(m.predict(x),float),0.) for m in bases]
    if len(pred)!=2 or any(p.shape!=(len(x),) or not np.isfinite(p).all() for p in pred):
        raise ValueError("Invalid base prediction")
    return tuple(pred)


@dataclass
class RidgeCorrection:
    estimator: Pipeline
    median: float

    def predict(self,h,x):
        h,x=map(lambda v:np.asarray(v,float),(h,x))
        if h.ndim!=1 or h.shape!=x.shape or not np.isfinite(np.c_[h,x]).all() or (h<0).any() or (x<0).any():
            raise ValueError("Invalid base predictions")
        delta=.5*self.estimator.predict(np.column_stack([h,x]))
        cap=.5*np.maximum(h,self.median)
        result=np.maximum(h+np.clip(delta,-cap,cap),0.)
        if not np.isfinite(result).all():
            raise ValueError("Invalid corrected prediction")
        return result


def fit_ridge(y,h,x):
    """Fit only on aligned cross-fitted TRAIN predictions, never fitted residuals."""
    y,h,x=map(lambda v:np.asarray(v,float),(y,h,x))
    if (y.ndim!=1 or not len(y) or y.shape!=h.shape or y.shape!=x.shape
            or not np.isfinite(np.c_[y,h,x]).all() or min(y.min(),h.min(),x.min())<0):
        raise ValueError("Invalid ridge OOF")
    model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=100.))])
    model.fit(np.column_stack([h,x]),y-h)
    return RidgeCorrection(model,max(1.,float(np.median(h))))


def paired_crossfit(tr,splits,fit_predict):
    tr=np.asarray(tr,dtype=int)
    if tr.ndim!=1 or not len(tr) or len(np.unique(tr))!=len(tr):
        raise ValueError("Invalid crossfit TRAIN")
    lookup={p:i for i,p in enumerate(tr)}
    oof=np.full((len(tr),2),np.nan)
    for a,b in splits:
        a,b=np.asarray(a,dtype=int),np.asarray(b,dtype=int)
        if (not len(a) or not len(b) or len(set(a))!=len(a) or len(set(b))!=len(b)
                or set(a)&set(b) or set(a)|set(b)!=set(tr)):
            raise ValueError("Invalid crossfit partition")
        pos=[lookup[i] for i in b]
        if np.isfinite(oof[pos]).any():
            raise ValueError("Repeated OOF row")
        h,x=fit_predict(a,b)
        v=np.column_stack([h,x])
        if v.shape!=(len(b),2) or not np.isfinite(v).all() or (v<0).any():
            raise ValueError("Invalid paired OOF predictions")
        oof[pos]=v
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete paired OOF")
    return oof[:,0],oof[:,1]
