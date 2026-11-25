"""Targeted M2 policies; preserve the original 170-feature signature identity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lithiumscope.experiments.generalization.arms import M2_REGULAR, prepare_classification_arm
from lithiumscope.experiments.generalization.signature_policy import signature_training


@dataclass(frozen=True)
class Policy:
    name: str
    regularized: bool = False
    signature: str | None = None
    features: str = "v4"


REFERENCE = Policy("reference")
REGULAR = Policy("regularized", True)
SOFT = Policy("regularized_signature_soft", True, "signature_soft")
WEIGHTED = Policy("regularized_signature_weighted", True, "signature_weighted")
LEGACY = Policy("regularized_126", True, features="v3")
KBEST = Policy("regularized_kbest48", True, features="kbest48")

# Five configurations for RF/SVM; four for the remaining families.
# These hypotheses reuse observed development folds, never an external holdout.
POLICIES = {
    "random_forest": (REFERENCE, REGULAR, SOFT, WEIGHTED, LEGACY),
    "extra_trees": (REFERENCE, REGULAR, SOFT, LEGACY),
    "hist_gradient_boosting": (REFERENCE, REGULAR, SOFT, LEGACY),
    "xgboost": (REFERENCE, REGULAR, SOFT, KBEST),
    "catboost": (REFERENCE, REGULAR, SOFT, KBEST),
    "svm_rbf": (REFERENCE, REGULAR, SOFT, WEIGHTED, LEGACY),
}


def feature_columns(x, policy):
    if len(x.columns) != 170 or not x.columns.is_unique:
        raise ValueError("Refinement requires the verified spectral V4 schema")
    # The upstream V4 loader proves that the first 126 are the original V3
    # features, with exactly 44 appended red-edge descriptors.
    return list(x.columns[:126] if policy.features == "v3" else x.columns)


def fit_policy(algorithm, policy, device, seed, x, y, groups, test):
    from sklearn.pipeline import Pipeline
    from lithiumscope.model_2.training.factory import create_model
    from lithiumscope.resources.manager import estimator_thread_scope

    columns = feature_columns(x, policy)
    prep = "kbest48" if policy.features == "kbest48" else "reference"
    fit_x, fit_y, train, reserved, names, trace = prepare_classification_arm(
        prep, seed, x[columns], y, groups, test[columns])
    weights = None
    if policy.signature is not None:
        # Signature identity is always computed on the full original data,
        # never after imputation, scaling or dimensionality reduction.
        fit_x, fit_y, weights, trace = signature_training(x, train, y, groups, policy.signature)
    params = dict(M2_REGULAR[algorithm]) if policy.regularized else {}
    if algorithm in {"random_forest", "extra_trees", "svm_rbf", "hist_gradient_boosting"}:
        params["class_weight"] = None
    model = create_model(algorithm, device, seed, params, calibrated=False)
    with estimator_thread_scope():
        kwargs = {} if weights is None else (
            {"model__sample_weight": weights} if isinstance(model, Pipeline) else {"sample_weight": weights})
        model.fit(fit_x, fit_y, **kwargs)
    probability = hasattr(model, "predict_proba")
    def score(matrix):
        values = model.predict_proba(matrix)[:,1] if probability else model.decision_function(matrix)
        values = np.asarray(values,float).reshape(-1)
        if len(values)!=len(matrix) or not np.isfinite(values).all():
            raise ValueError("Invalid M2 scores")
        if probability and ((values<0)|(values>1)).any():
            raise ValueError("Invalid probability range")
        return values
    return score(reserved), score(train), {
        "score_kind":"probability" if probability else "margin",
        "fit_rows":len(fit_y),"features":len(names),"feature_names":names,
        "effective_fit_weight":float(len(fit_y) if weights is None else weights.sum()),
        "effective_positive_weight":float(np.sum(fit_y) if weights is None else np.dot(fit_y,weights)),
    }, trace
