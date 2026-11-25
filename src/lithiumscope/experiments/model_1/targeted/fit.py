"""Isolate treatment effects while preserving archived fold hyperparameters."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.experiments.generalization.resampling import regression_smoter
from lithiumscope.experiments.model_1.targeted.features import FoldFeatures
from lithiumscope.experiments.model_1.targeted.policies import Target, adaptive_positions, weights
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.manager import estimator_thread_scope


def _base_fit(algorithm, arm, device, seed, params, x, y, groups, sources, numeric, categorical):
    if not (x.index.equals(y.index) and x.index.equals(groups.index) and x.index.equals(sources.index)):
        raise ValueError("Misaligned TRAIN data")
    schema = FeatureSchema(list(numeric), list(categorical), 2, "stable_v1")
    if arm.name == "control":
        model = build_pipeline(algorithm, device, seed, schema, params)
        fit_regression_estimator(model, x, y, groups)
        info = model.treatment_evidence_
        trace = [{"row": row, "action": "bootstrap_copy"} for row in info["copied_train_row_ids"]]
        return model.predict, {"fit_rows": info["fit_rows"], "treatment": info, "trace": trace}
    features = FoldFeatures(arm, numeric, categorical).fit(x, y, groups)
    target = Target(arm.transform).fit(y)
    trace = []
    fit_x, fit_y, fit_groups = x, y, groups
    if arm.resample == "quantile":
        extra = adaptive_positions(y, groups, seed)
        keep = np.r_[np.arange(len(x)), extra]
        fit_x, fit_y, fit_groups = (v.iloc[keep].reset_index(drop=True) for v in (x, y, groups))
        trace = [{"row": str(x.index[i]), "action": "bootstrap_copy"} for i in extra]
    elif arm.resample == "smoter":
        fit_x, fit_y, fit_groups, trace = regression_smoter(
            x, y, groups, sources, seed, numeric, categorical)
    allowed = set(x.index.astype(str))
    if any(str(item[key]) not in allowed for item in trace for key in ("row", "parent_a", "parent_b") if key in item):
        raise ValueError("Augmentation parent outside TRAIN")
    if arm.resample != "none" and arm.weight != "none":
        raise ValueError("Combined augmentation and weights require explicit source lineage")
    fit_weights = weights(fit_y, fit_groups, sources if arm.resample == "none" else fit_groups, arm.weight)
    schema = FeatureSchema(features.numeric, features.categorical, 2, "stable_v1")
    base = {key: value for key, value in params.items() if key != "treatment_policy"}
    model = build_pipeline(algorithm, device, seed, schema, base)
    with estimator_thread_scope():
        model.fit(features.transform(fit_x), target.encode(fit_y), model__sample_weight=fit_weights)

    def predict(frame):
        return target.decode(model.predict(features.transform(frame)))

    return predict, {"fit_rows": len(fit_x), "trace": trace, "features": features.numeric + features.categorical,
                     "weight_min": float(fit_weights.min()), "weight_max": float(fit_weights.max()),
                     "target_transform": arm.transform}


def fit_arm(algorithm, arm, device, seed, params, x, y, groups, sources, numeric, categorical, validation):
    """Fit a treatment; optional bias calibration uses grouped OOF within this TRAIN."""
    args = (algorithm, arm, device, seed, params)
    correction = 0.
    if arm.correction:
        if groups.nunique() < 3:
            raise ValueError("Cross-fit correction requires at least three TRAIN groups")
        residual = np.full(len(x), np.nan)
        for tr, va in GroupKFold(3).split(x, y, groups):
            predict, _ = _base_fit(*args, x.iloc[tr], y.iloc[tr], groups.iloc[tr],
                                   sources.iloc[tr], numeric, categorical)
            residual[va] = y.iloc[va].to_numpy(float) - predict(x.iloc[va])
        if not np.isfinite(residual).all():
            raise ValueError("Incomplete cross-fit calibration")
        # Original-ppm additive bias correction; not an unvalidated inverse multiplier.
        correction = float(residual.mean())
    predict, evidence = _base_fit(*args, x, y, groups, sources, numeric, categorical)
    evidence["crossfit_bias_ppm"] = correction
    evidence["calibration_fits"] = 3 if arm.correction else 0
    scores = np.maximum(np.asarray(predict(validation), float) + correction, 0.)
    train_scores = np.maximum(np.asarray(predict(x), float) + correction, 0.)
    if not np.isfinite(scores).all() or not np.isfinite(train_scores).all():
        raise ValueError("Nonfinite predictions")
    return scores, train_scores, evidence
