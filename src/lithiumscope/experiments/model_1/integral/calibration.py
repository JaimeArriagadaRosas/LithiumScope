"""Point calibration on out-of-sample TRAIN predictions; no TEST labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE="identity"
POLICIES=(BASE,"additive","affine_shrunk","isotonic_shrunk","residual_ridge")
PRED="__base_prediction_ppm"


def checked(p,index):
    p=np.asarray(p,float)
    if p.shape!=(len(index),) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError("Invalid base predictions")
    return p


def fit_calibrators(x,y,oof,numeric,categorical):
    """Caller supplies cross-fitted OOF aligned to raw canonical TRAIN only."""
    cols=list(numeric)+list(categorical)
    if (not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(oof.index)
            or list(x.columns)!=cols or PRED in cols or not len(x)
            or not np.isfinite(y).all() or y.lt(0).any()):
        raise ValueError("Invalid or unaligned calibration TRAIN")
    p=checked(oof,x.index)
    residual=y.to_numpy(float)-p
    # Fit a scalar affine correction in original ppm, with slope regularized.
    affine=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.))])
    affine.fit(p[:,None],residual)
    iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=0.)
    iso.fit(p,y.to_numpy(float))
    # Use only canonical features that already exist at prediction time.
    numeric_pipe=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                           ("scale",StandardScaler())])
    categorical_pipe=Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="__missing__")),
                               ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    prep=ColumnTransformer([("numeric",numeric_pipe,list(numeric)+[PRED]),
                            ("categorical",categorical_pipe,list(categorical))],
                           remainder="drop",sparse_threshold=0.)
    model=Pipeline([("prep",prep),("ridge",Ridge(alpha=100.))])
    def augment(frame,pred):
        # Normalise missing categorical values consistently; no fitted state here.
        out=frame.copy()
        for name in categorical:
            out[name]=out[name].astype(object).where(out[name].notna(),np.nan)
        out[PRED]=pred
        return out
    model.fit(augment(x,p),residual)
    bias=float(residual.mean())
    median=max(1.,float(np.median(p)))
    seen=set(x.index)

    def predict(frame,pred):
        if (not frame.index.is_unique or seen.intersection(frame.index)
                or list(frame.columns)!=cols):
            raise ValueError("Calibration evaluation overlaps TRAIN or schema differs")
        z=checked(pred,frame.index)
        cap=.5*np.maximum(z,median)
        deltas={
            "additive":np.full(len(z),bias),
            "affine_shrunk":.5*affine.predict(z[:,None]),
            "isotonic_shrunk":.5*(iso.predict(z)-z),
            "residual_ridge":.5*model.predict(augment(frame,z)),
        }
        values={BASE:z.copy()}
        for name,delta in deltas.items():
            if not np.isfinite(delta).all():
                raise ValueError("Invalid calibration correction")
            values[name]=np.maximum(z+np.clip(delta,-cap,cap),0.)
        return values

    return predict,{"calibration_rows":len(x),"base_prediction_median":median,
                    "mean_residual":bias,"affine_alpha":10.,"residual_alpha":100.,
                    "shrinkage":.5,"correction_cap_fraction":.5,
                    "fitted_estimators":3,"fitted_bias_statistics":1}
