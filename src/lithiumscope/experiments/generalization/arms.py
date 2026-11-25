"""Fixed, predeclared treatments for nested spatial development screening."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lithiumscope.experiments.generalization.resampling import classification_resample, regression_resample, regression_smoter

M1_ARMS = ("reference", "nonnegative", "log1p", "deduplicate", "rare_bootstrap",
           "group_bootstrap", "rare_group_bootstrap", "smoter", "without_coordinates", "coverage_filter", "regularized", "knn_imputation")
M2_ARMS = ("reference", "regularized", "weighted", "oversample", "undersample", "smote_group",
           "pca48", "kbest48", "correlation_filter", "signature_weighted", "signature_soft")

M1_REGULAR = {
    "random_forest": {"max_depth": 10, "min_samples_leaf": 8},
    "xgboost": {"max_depth": 3, "min_child_weight": 8, "reg_lambda": 10, "subsample": .8},
    "hist_gradient_boosting": {"max_leaf_nodes": 12, "min_samples_leaf": 30, "l2_regularization": 5},
    "catboost": {"depth": 4, "l2_leaf_reg": 10},
    "svm": {"C": 10., "epsilon": .1, "gamma": "scale"},
    "tabnet": {"max_epochs": 60, "patience": 10},
}
M2_REGULAR = {
    "random_forest": {"max_depth": 8, "min_samples_leaf": 10},
    "extra_trees": {"max_depth": 8, "min_samples_leaf": 10},
    "xgboost": {"max_depth": 3, "min_child_weight": 8, "reg_lambda": 10},
    "hist_gradient_boosting": {"max_leaf_nodes": 10, "min_samples_leaf": 25, "l2_regularization": 5},
    "catboost": {"depth": 4, "l2_leaf_reg": 10},
    "svm_rbf": {"C": .5, "gamma": "scale"},
}


def correlation_columns(x, threshold=.98):
    corr = x.corr(method="spearman").abs()
    keep = []
    for name in x:
        if not any(pd.notna(corr.loc[name, prev]) and corr.loc[name, prev] >= threshold for prev in keep):
            keep.append(name)
    return keep


def prepare_regression_arm(arm, seed, x, y, groups, sources, numeric, categorical):
    from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
    method = arm if arm in {"deduplicate", "rare_bootstrap", "group_bootstrap", "rare_group_bootstrap"} else "none"
    cols = list(x.columns)
    if arm == "without_coordinates":
        cols = [c for c in cols if c not in {"Latitude", "Longitude"}]
    if arm == "coverage_filter":
        cols = [c for c in cols if x[c].notna().mean() >= .5 and x[c].nunique(dropna=True) > 1]
    if not cols:
        raise ValueError("No TRAIN features remain")
    schema = FeatureSchema([c for c in numeric if c in cols], [c for c in categorical if c in cols], 2, "knn_v1" if arm == "knn_imputation" else "stable_v1")
    if arm == "smoter":
        fit_x, fit_y, fit_groups, provenance = regression_smoter(x[cols], y, groups, sources, seed, schema.numeric, schema.categorical)
    else:
        fit_x, fit_y, fit_groups, provenance = regression_resample(x[cols], y, groups, sources, method, seed)
    return fit_x, fit_y, fit_groups, schema, provenance


def regression_fit(algorithm, arm, device, seed, x, y, groups, sources, numeric, categorical, validation):
    from lithiumscope.model_1.training.factory import build_pipeline
    from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
    fit_x, fit_y, fit_groups, schema, provenance = prepare_regression_arm(arm, seed, x, y, groups, sources, numeric, categorical)
    cols = schema.numeric + schema.categorical
    params = M1_REGULAR.get(algorithm, {}) if arm == "regularized" else {}
    if arm == "regularized" and algorithm not in M1_REGULAR:
        raise ValueError("Regularized arm not yet configured for this family")
    estimator = build_pipeline(algorithm, device, seed, schema, params,
                               target_transform="log1p" if arm == "log1p" else "identity")
    fit_regression_estimator(estimator, fit_x, fit_y, fit_groups)
    pred = np.asarray(estimator.predict(validation[cols]), float)
    train_pred = np.asarray(estimator.predict(x[cols]), float)
    if arm in {"nonnegative", "log1p"}:
        pred, train_pred = np.maximum(pred, 0), np.maximum(train_pred, 0)
    return pred, train_pred, {"features": cols, "fit_rows": len(fit_x), "generated": provenance}


def prepare_classification_arm(arm, seed, x, y, groups, validation=None, *, return_weights=False):
    columns = correlation_columns(x) if arm == "correlation_filter" else list(x.columns)
    pre = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("variance", VarianceThreshold()), ("scale", StandardScaler())])
    fit = pre.fit_transform(x[columns])
    test = pre.transform(validation[columns]) if validation is not None else None
    selector = None
    k = min(48, fit.shape[1], len(fit)-1)
    if arm == "pca48":
        selector = PCA(n_components=k, random_state=seed)
    if arm == "kbest48":
        selector = SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)
    if selector is not None:
        fit = selector.fit_transform(fit, y)
        if test is not None:
            test = selector.transform(test)
    method = arm if arm in {"oversample", "undersample", "smote_group"} else "none"
    xx, yy, provenance = classification_resample(pd.DataFrame(fit, index=x.index), y, groups, method, seed)
    weights = None
    if arm in {"signature_weighted", "signature_soft"}:
        from lithiumscope.experiments.generalization.signature_policy import signature_training
        xx, yy, weights, provenance = signature_training(x, fit, y, groups, arm)
    names = np.asarray(columns)[pre.named_steps["variance"].get_support()].tolist()
    if arm == "pca48":
        names = [f"PC{i+1}" for i in range(fit.shape[1])]
    elif arm == "kbest48":
        names = np.asarray(names)[selector.get_support()].tolist()
    result = (xx, yy, fit, test, names, provenance)
    return (*result, weights) if return_weights else result


def classification_fit(algorithm, arm, device, seed, x, y, groups, validation):
    from lithiumscope.model_2.training.factory import create_model
    from lithiumscope.resources.manager import estimator_thread_scope
    xx, yy, fit, test, names, provenance, weights = prepare_classification_arm(arm, seed, x, y, groups, validation, return_weights=True)
    params = dict(M2_REGULAR.get(algorithm, {})) if arm == "regularized" else {}
    if algorithm in {"random_forest", "extra_trees", "svm_rbf", "hist_gradient_boosting"}:
        params["class_weight"] = None
    if arm == "weighted":
        if algorithm in {"random_forest", "extra_trees", "svm_rbf", "hist_gradient_boosting"}:
            params["class_weight"] = "balanced"
        elif algorithm == "catboost":
            params["auto_class_weights"] = "Balanced"
        elif algorithm == "xgboost":
            params["scale_pos_weight"] = float((y == 0).sum() / (y == 1).sum())
    if arm in {"oversample", "undersample", "smote_group"} and algorithm in {"random_forest", "extra_trees", "svm_rbf"}:
        params["class_weight"] = None
    model = create_model(algorithm, device, seed, params, calibrated=False)
    with estimator_thread_scope():
        kwargs = {} if weights is None else ({"model__sample_weight": weights} if isinstance(model, Pipeline) else {"sample_weight": weights})
        model.fit(xx, yy, **kwargs)
    def scores(matrix):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(matrix)[:, 1]
        return model.decision_function(matrix)
    # Raw SVM margins are ranked, never interpreted as probabilities.
    return scores(test), scores(fit), {"features": int(fit.shape[1]), "feature_names": names, "fit_rows": len(xx),
            "generated": provenance, "effective_fit_weight": float(len(yy) if weights is None else weights.sum()), "score_kind": "probability" if hasattr(model, "predict_proba") else "margin"}
