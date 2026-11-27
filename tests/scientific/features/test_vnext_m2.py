import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_2.representation.augmentation import augment_image_training_only
from lithiumscope.experiments.model_2.evaluation.metrics import expected_calibration_error, score_metrics
from lithiumscope.experiments.model_2.calibration.partitions import calibration_splits


def test_image_augmentation_is_bounded_deterministic_and_non_destructive():
    cube=np.full((6,32,32),.25,dtype=np.float32)
    original=cube.copy()
    a=augment_image_training_only(cube,seed=21)
    b=augment_image_training_only(cube,seed=21)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(cube,original)
    assert a.shape==cube.shape
    assert np.isnan(a).any()
    assert np.nanmin(a)>=0
    assert np.nanmax(a)<=1.6


def test_calibration_error_math():
    labels=[0,0,1,1]
    probs=[.1,.2,.8,.9]
    score=expected_calibration_error(labels,probs,bins=2)
    assert abs(score-.15)<.00001
    with pytest.raises(ValueError):
        expected_calibration_error(labels,[.1,.2,.3])


def test_no_random_calibration_fallback_on_bad_groups():
    x=pd.DataFrame({"x":range(9)})
    y=pd.Series([0,0,0,1,1,1,0,0,0])
    groups=pd.Series([0,0,0,1,1,1,2,2,2])
    assert calibration_splits(x,y,groups,3) is None


def test_scores_reject_one_class_holdout():
    with pytest.raises(ValueError):
        score_metrics([1,1],[.6,.8])
