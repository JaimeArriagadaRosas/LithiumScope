import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from lithiumscope.model_2.training.svm_rbf import create_model


def test_model_2_svm_uses_explicit_calibration():
    estimator = create_model(
        random_seed=42,
        calibrated=True,
        calibration_cv=3,
    )

    assert isinstance(
        estimator,
        CalibratedClassifierCV,
    )
    assert isinstance(
        estimator.estimator,
        Pipeline,
    )


def test_model_2_svm_fit_has_no_probability_futurewarning():
    estimator = create_model(
        random_seed=42,
        calibrated=False,
    )
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.2, 0.9],
            [0.2, 0.1],
            [0.9, 1.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            FutureWarning,
        )
        estimator.fit(x, y)
