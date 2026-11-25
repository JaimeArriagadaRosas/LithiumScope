"""Train-only preprocessing contracts for M2 estimators and inner folds."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def scaled_estimator(estimator):
    """Keep scaling inside each fit, including nested calibration and CV."""
    return Pipeline([("scaler",StandardScaler()),("model",estimator)])


def inner_fold_labels(lithium,train,validation,*,quantile=.75,fixed_threshold=None):
    values=np.asarray(lithium,dtype=float)
    train,validation=np.asarray(train),np.asarray(validation)
    if (values.ndim!=1 or not np.isfinite(values).all() or (values<0).any()
            or train.ndim!=1 or validation.ndim!=1 or not len(train) or not len(validation)
            or train.dtype.kind not in "iu" or validation.dtype.kind not in "iu"
            or min(train.min(),validation.min())<0 or max(train.max(),validation.max())>=len(values)
            or np.intersect1d(train,validation).size or not 0<quantile<1):
        raise ValueError("Invalid train-only label inputs")
    threshold=float(np.quantile(values[train],quantile) if fixed_threshold is None else fixed_threshold)
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError("Invalid Li endpoint")
    return (values[train]>=threshold).astype(int),(values[validation]>=threshold).astype(int),threshold


def eligible_positions(mask,n):
    if mask is None:
        return np.arange(n,dtype=int)
    mask=np.asarray(mask)
    if mask.dtype.kind!="b" or mask.shape!=(n,) or not mask.any():
        raise ValueError("TRAIN quality mask must be aligned boolean data")
    return np.flatnonzero(mask)
