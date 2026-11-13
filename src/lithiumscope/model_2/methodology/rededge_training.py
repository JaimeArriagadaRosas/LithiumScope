"""Fixed-training M2 v4 factory preserving inner-group calibration and six arms."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from lithiumscope.model_2.training.factory import create_model


def fit_frozen_model(
    algorithm,
    params,
    train_x,
    train_y,
    groups,
    device,
    seed,
):
    if algorithm == "svm_rbf":
        splits = list(
            GroupKFold(n_splits=3).split(
                train_x,
                train_y,
                groups=groups,
            )
        )
        for inner_train, inner_test in splits:
            if len(np.unique(train_y[inner_train])) < 2 or len(np.unique(train_y[inner_test])) < 2:
                raise ValueError("SVM calibrator lacks a class in an inner fold")
        # A sigmoid is fitted entirely inside outer TRAIN.
        model = create_model(
            algorithm,
            device,
            seed,
            params,
            calibrated=True,
            calibration_cv=splits,
        )
    else:
        model = create_model(
            algorithm,
            device,
            seed,
            params,
            calibrated=False,
        )
    model.fit(train_x, train_y)
    return model


def combine_quality_masks(base_mask, quality_csv, manifest):
    """Exclude low-coverage NEW bands from TRAIN; keep every outer TEST."""
    import pandas as pd

    summary = pd.read_csv(quality_csv)
    if len(summary) != len(manifest) or not np.array_equal(
        summary.sample_id.astype(str),
        manifest.sample_id.astype(str),
    ):
        raise ValueError("M2 v4 red-edge coverage audit is not aligned")
    fractions = summary.rededge_joint_valid_fraction.to_numpy(float)
    if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("Invalid red-edge coverage fraction")
    base = np.asarray(base_mask, dtype=bool)
    if base.shape != fractions.shape:
        raise ValueError("M2 v4 original quality mask drift")
    combined = base & (fractions >= 0.95)
    if not combined.any():
        raise ValueError("No usable M2 v4 training data after QA95")
    return combined, {
        "original_qa95_train_eligible": int(base.sum()),
        "rededge_qa95_train_eligible": int((fractions >= 0.95).sum()),
        "joint_qa95_train_eligible": int(combined.sum()),
        "all_outer_test_retained": True,
    }
