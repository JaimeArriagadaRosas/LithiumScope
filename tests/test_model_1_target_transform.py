import numpy as np
from sklearn.dummy import DummyRegressor

from lithiumscope.model_1.training.target_transform import (
    unwrap_fitted_regressor,
    validate_target_transform,
    wrap_target_regressor,
)


def test_log1p_target_transform_predicts_in_original_scale():
    estimator = wrap_target_regressor(
        DummyRegressor(strategy="mean"),
        "log1p",
    )
    x = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.asarray(
        [1.0, 3.0, 7.0, 15.0],
        dtype=float,
    )

    estimator.fit(x, y)
    prediction = estimator.predict(
        np.asarray([[10.0]])
    )[0]

    expected = np.expm1(
        np.log1p(y).mean()
    )
    assert np.isclose(
        prediction,
        expected,
    )
    assert (
        unwrap_fitted_regressor(
            estimator
        )
        is estimator.regressor_
    )


def test_log1p_target_transform_rejects_negative_targets():
    with np.testing.assert_raises(
        ValueError
    ):
        validate_target_transform(
            "log1p",
            np.asarray(
                [1.0, -0.1, 2.0]
            ),
        )
