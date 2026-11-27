"""Operating cut fitted only on grouped cross-fitted development scores.

The binary endpoint is fixed at the enclosing development partition's Li cut.
Hyperparameters may already have been selected on that development partition;
only its outer TEST can evaluate the resulting operating policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.core.scientific_checks import assert_partition_coverage, assert_group_isolation
from lithiumscope.core.scientific.classification_metrics import validate_binary_inputs, classification_metrics
from lithiumscope.model_2.methodology.evaluation.decision import freeze_inner_oof_threshold


def learn_operating_policy(x, y, groups, *, fit, row_indices, reserved_indices,
                           n_splits: int, objective: str = "balanced_accuracy"):
    """Cross-fit the actual calibrated estimator policy; never receive outer TEST X/y."""
    labels, _ = validate_binary_inputs(y, np.full(len(y), .5))
    groups = pd.Series(groups).reset_index(drop=True)
    rows = np.asarray(row_indices)
    reserved = np.asarray(reserved_indices)
    if (len(x) != len(labels) or len(groups) != len(x) or rows.shape != (len(x),)
            or len(set(rows)) != len(rows) or set(rows) & set(reserved)
            or groups.isna().any() or groups.nunique() < n_splits or n_splits < 2):
        raise ValueError("Invalid grouped development partition for operating threshold")
    splits = list(GroupKFold(n_splits=n_splits).split(x, groups=groups))
    assert_partition_coverage(splits, len(x))
    # Preflight all partitions before any expensive fit.
    for train, valid in splits:
        assert_group_isolation(train, valid, groups)
        if np.unique(labels[train]).size != 2 or np.unique(labels[valid]).size != 2:
            raise ValueError("Operating threshold inner fold lacks both classes")
    records = []
    for number, (train, valid) in enumerate(splits, 1):
        estimator = fit(x.iloc[train], pd.Series(labels[train]), groups.iloc[train].reset_index(drop=True))
        scores = np.asarray(estimator.predict_proba(x.iloc[valid]))[:, 1]
        validate_binary_inputs(labels[valid], scores)
        records.extend({"row_index": int(rows[i]), "group": str(groups.iloc[i]),
                        "inner_fold": number, "y_true": int(labels[i]), "score": float(score)}
                       for i, score in zip(valid, scores, strict=True))
    table = pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)
    frozen = freeze_inner_oof_threshold(table, permitted_outer_train_indices=rows,
                                        reserved_outer_test_indices=reserved, objective=objective)
    return frozen, table


def operating_evidence(frozen, table, y_test, scores_test):
    """Evaluate a previously selected cut; preserve development selection evidence."""
    selection = frozen.selection
    evaluated = classification_metrics(y_test, scores_test, threshold=selection.threshold)
    return {"operating_threshold": selection.threshold,
            "operating_objective": selection.objective,
            "operating_inner_objective": selection.objective_value,
            "operating_inner_count": frozen.inner_count,
            "operating_inner_rows_sha256": frozen.inner_rows_sha256,
            "operating_inner_predictions": table.to_dict("records"),
            "operating_outer_metrics": evaluated}


def fit_policy(algorithm, device, seed, params, x, y, groups, inner_folds, mode):
    """Shared estimator construction for inner, outer, and final training."""
    if mode == "spatial_sigmoid":
        from lithiumscope.model_2.training.calibration.policy import fit_spatially_calibrated
        return fit_spatially_calibrated(algorithm, device, seed, params, x, y, groups, inner_folds)
    from lithiumscope.model_2.training.execution.fold_utilities import _fit_estimator
    return _fit_estimator(algorithm, device, seed, params, x, y, groups, inner_folds, strict_spatial=True)
