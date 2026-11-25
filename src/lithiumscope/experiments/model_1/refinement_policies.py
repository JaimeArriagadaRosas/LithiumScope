"""M1-only follow-up hypotheses; no changes to production model training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    resampling: str = "none"
    target: str = "identity"
    coordinates: bool = True
    robust: bool = False
    bounded: bool = False
    nonnegative: bool = True


REFERENCE = Policy("reference", nonnegative=False)

# Shortlist after the first 12-arm screen. Never interpret reused folds
# as independent confirmation of superiority.
POLICIES = {
    "random_forest": (
        REFERENCE, Policy("rare_group_no_coordinates", "rare_group_bootstrap", coordinates=False),
        Policy("group_no_coordinates", "group_bootstrap", coordinates=False),
        Policy("rare_group_sqrt", "rare_group_bootstrap", "sqrt"),
    ),
    "xgboost": (
        REFERENCE, Policy("bounded_log", target="log1p", bounded=True),
        Policy("bounded_log_no_coordinates", target="log1p", coordinates=False, bounded=True),
        Policy("rare_group_sqrt", "rare_group_bootstrap", "sqrt"),
    ),
    "svm": (
        REFERENCE, Policy("bounded_log", target="log1p", bounded=True),
        Policy("group_bounded_log", "group_bootstrap", "log1p", bounded=True),
        Policy("bounded_log_no_coordinates", target="log1p", coordinates=False, bounded=True),
    ),
    "tabnet": (
        REFERENCE, Policy("robust_identity", robust=True),
        Policy("robust_bounded_log", target="log1p", robust=True, bounded=True),
        Policy("robust_bounded_sqrt", target="sqrt", robust=True, bounded=True),
    ),
    "hist_gradient_boosting": (
        REFERENCE, Policy("bounded_log", target="log1p", bounded=True),
        Policy("rare_group_bounded_log", "rare_group_bootstrap", "log1p", bounded=True),
        Policy("bounded_log_no_coordinates", target="log1p", coordinates=False, bounded=True),
        Policy("rare_group_sqrt", "rare_group_bootstrap", "sqrt"),
    ),
    "catboost": (
        REFERENCE, Policy("rare_group_nonnegative", "rare_group_bootstrap"),
        Policy("rare_group_no_coordinates", "rare_group_bootstrap", coordinates=False),
        Policy("group_no_coordinates", "group_bootstrap", coordinates=False),
        Policy("rare_group_sqrt", "rare_group_bootstrap", "sqrt"),
    ),
}


def encode(y, transform):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid M1 target")
    if transform == "identity":
        return y
    if transform == "log1p":
        return np.log1p(y)
    if transform == "sqrt":
        return np.sqrt(y)
    raise ValueError("Unknown transform")


def decode(latent, policy, train_max):
    """Project negative outputs, or bound a transformed arm by TRAIN support.

    The upper bound is experimental, not a geological maximum. Counts retain
    evidence of attempted extrapolation and projection. Never use TEST labels.
    """
    z = np.asarray(latent, dtype=float).reshape(-1)
    if not np.isfinite(z).all() or not np.isfinite(train_max) or train_max < 0:
        raise ValueError("Invalid prediction or TRAIN support")
    negative = z < 0
    bound = float(encode([train_max], policy.target)[0])
    above = z > bound
    if policy.nonnegative:
        z = np.maximum(z, 0.)
    if policy.bounded:
        z = np.minimum(z, bound)
    with np.errstate(over="raise", invalid="raise"):
        if policy.target == "log1p":
            pred = np.expm1(z)
        elif policy.target == "sqrt":
            pred = z**2
        elif policy.target == "identity":
            pred = z
        else:
            raise ValueError("Unknown transform")
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite decoded M1 prediction")
    return pred, negative, above


def fit_policy(algorithm, policy, device, seed, x, y, groups, sources, numeric, categorical, test):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from lithiumscope.experiments.generalization.resampling import regression_resample
    from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
    from lithiumscope.model_1.training.factory import build_pipeline
    from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator

    columns = [c for c in x if policy.coordinates or c not in {"Latitude", "Longitude"}]
    schema = FeatureSchema([c for c in numeric if c in columns],
                           [c for c in categorical if c in columns], 2, "stable_v1")
    fit_x, fit_y, fit_groups, trace = regression_resample(
        x[columns], y, groups, sources, policy.resampling, seed)
    estimator = build_pipeline(algorithm, device, seed, schema, {}, target_transform="identity")
    if policy.robust:
        # Fit inside each TabNet spatial early-stopping split and final TRAIN.
        estimator.set_params(preprocess__numeric=Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
        ]))
    target = pd.Series(encode(fit_y, policy.target), index=fit_y.index)
    fit_regression_estimator(estimator, fit_x, target, fit_groups)
    train_max = float(y.max())
    scores, negative, above = decode(estimator.predict(test[columns]), policy, train_max)
    train_scores, _, _ = decode(estimator.predict(x[columns]), policy, train_max)
    return scores, train_scores, negative, above, trace, len(fit_x)
