"""Bounded stacking and shallow residual correction using cross-fitted TRAIN predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from lithiumscope.experiments.model_1.integral.calibration import checked
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.resources.manager import estimator_thread_scope

BASE="identity"
POLICIES=(BASE,"equal_blend","convex_stack","ridge_stack","nonlinear_residual")
HP="__hgb_prediction_ppm"


def convex_weight(y,h,x):
    y,h,x=(np.asarray(a,float) for a in (y,h,x))
    if y.ndim!=1 or not len(y) or y.shape!=h.shape or y.shape!=x.shape or not np.isfinite(np.c_[y,h,x]).all():
        raise ValueError("Invalid stacking TRAIN")
    d=x-h
    # Ridge penalty on the XGB fraction, centered at HGB-only.
    return float(np.clip(np.dot(d,y-h)/(np.dot(d,d)+len(y)),0.,.75))


def fit_stack(frame,y,hgb,xgb,numeric,categorical,device):
    cols=list(numeric)+list(categorical)
    if (not frame.index.is_unique or not frame.index.equals(y.index)
            or not frame.index.equals(hgb.index) or not frame.index.equals(xgb.index)
            or list(frame.columns)!=cols or HP in cols or not len(y)
            or not np.isfinite(y).all() or y.lt(0).any()):
        raise ValueError("Invalid or unaligned stacking TRAIN")
    h=checked(hgb,frame.index)
    x=checked(xgb,frame.index)
    residual=y.to_numpy(float)-h
    weight=convex_weight(y,h,x)
    ridge=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=100.))])
    ridge.fit(np.column_stack([h,x]),residual)
    schema=FeatureSchema(list(numeric)+[HP],list(categorical),2,"stable_v1")
    model=build_pipeline("hist_gradient_boosting",device,42,schema,{"loss":"squared_error",
        "max_iter":100,"max_leaf_nodes":7,"min_samples_leaf":40,"learning_rate":.05,
        "l2_regularization":10.,"early_stopping":False})
    model.set_params(preprocess__numeric="passthrough")
    def augment(f,p):
        z=f.copy()
        z[HP]=p
        return z[schema.numeric+schema.categorical]
    with estimator_thread_scope():
        model.fit(augment(frame,h),residual)
    seen=set(frame.index)
    median=max(1.,float(np.median(h)))

    def predict(f,hp,xp):
        if not f.index.is_unique or seen.intersection(f.index) or list(f.columns)!=cols:
            raise ValueError("Stacking evaluation overlaps TRAIN or schema differs")
        a,b=checked(hp,f.index),checked(xp,f.index)
        cap=.5*np.maximum(a,median)
        with estimator_thread_scope():
            delta=.5*np.asarray(model.predict(augment(f,a)),float)
        corrections={"ridge_stack":.5*ridge.predict(np.column_stack([a,b])),
                     "nonlinear_residual":delta}
        out={BASE:a.copy(),"equal_blend":.5*(a+b),"convex_stack":(1-weight)*a+weight*b}
        for name,d in corrections.items():
            if not np.isfinite(d).all():
                raise ValueError("Invalid stacking correction")
            out[name]=np.maximum(a+np.clip(d,-cap,cap),0.)
        return out

    return predict,{"xgb_weight":weight,"ridge_alpha":100.,"residual_shrinkage":.5,
                    "correction_cap_fraction":.5,"base_prediction_median":median,
                    "fit_rows":len(y),"meta_estimators":2,"convex_weight_fits":1}
