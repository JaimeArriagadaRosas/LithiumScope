"""Versioned, train-only spatial calibration for ALL six M2 competitors.

The outer TEST never appears in the fit or in calibration. The same policy
applies to every algorithm; it cannot bake in Random Forest as the winner.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

from lithiumscope.model_2.training.factory import create_model

POLICIES = {"legacy", "spatial_sigmoid"}


def calibration_policy(optimization: dict) -> str:
    policy = str(optimization.get("calibration_policy", "legacy"))
    if policy not in POLICIES:
        raise ValueError(f"Unknown M2 calibration policy: {policy}")
    return policy


def fit_spatially_calibrated(
    algorithm,
    device,
    seed,
    params,
    x_train,
    y_train,
    groups_train,
    inner_folds,
):
    """Fit calibrated probabilities with grouped TRAIN-only CV.

    All algorithms use an uncalibrated estimator as the base; this avoids
    nested double SVM calibration. Every calibration fold requires both
    classes in train and validation.
    """
    if groups_train is None or groups_train.nunique() < inner_folds:
        raise ValueError("Calibrator needs enough training spatial groups")
    y = np.asarray(y_train, int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibrator needs two training classes")
    splits = list(
        GroupKFold(n_splits=inner_folds).split(x_train, y, groups_train)
    )
    training_groups = np.asarray(groups_train)
    for train, valid in splits:
        if (len(set(training_groups[train]) & set(training_groups[valid])) != 0
            or len(np.unique(y[train])) != 2
            or len(np.unique(y[valid])) != 2):
            raise ValueError("Invalid grouped train-only calibration fold")
    base = create_model(algorithm, device, seed, params, calibrated=False)
    model = CalibratedClassifierCV(
        estimator=base, method="sigmoid", cv=splits, ensemble=False,
    )
    model.fit(x_train, y)
    return model
