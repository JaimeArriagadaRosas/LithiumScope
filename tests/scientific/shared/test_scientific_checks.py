import numpy as np
import pandas as pd
import pytest

from lithiumscope.core.scientific_checks import (
    ScientificValidationError,
    assert_group_isolation,
    assert_no_target_leakage,
    assert_oof_complete,
)


def test_target_leakage_is_rejected():
    with pytest.raises(ScientificValidationError):
        assert_no_target_leakage("Li_icpms", ["SiO2", "Li_icpms"])


def test_oof_requires_one_finite_prediction_per_sample():
    assert_oof_complete(np.array([1.0, 2.0, 3.0]), 3)
    with pytest.raises(ScientificValidationError):
        assert_oof_complete(np.array([1.0, np.nan, 3.0]), 3)


def test_spatial_groups_cannot_cross_train_test():
    groups = pd.Series(["A", "A", "B", "B"])
    assert_group_isolation([0, 1], [2, 3], groups)
    with pytest.raises(ScientificValidationError):
        assert_group_isolation([0, 2], [1, 3], groups)
