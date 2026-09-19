from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from lithiumscope.model_2.training.svm_rbf import create_model


def test_model_2_svm_uses_explicit_calibration_not_svc_probability():
    estimator = create_model(
        random_seed=42,
        calibrated=True,
        calibration_cv=3,
    )

    assert isinstance(estimator, CalibratedClassifierCV)
    base = estimator.estimator
    assert isinstance(base, Pipeline)
    assert base.named_steps["model"].probability is False
