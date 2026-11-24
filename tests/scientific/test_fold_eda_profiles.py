import numpy as np
import pandas as pd
import pytest

from lithiumscope.core.scientific.fold_profiles import fold_profiles


def test_fold_profiles_uses_only_eligible_training_rows():
    x = pd.DataFrame({"a": [1.0, None, 999.0, 500.0]})
    y = pd.Series([5, 10, 8000, 7000])
    groups = pd.Series(["g1", "g1", "g2", "g2"])
    splits = [(np.array([0, 1]), np.array([2, 3]))]
    summary, profile = fold_profiles(x, y, groups, splits, np.array([True, False, True, True]))
    assert summary.loc[0, "train_li_max"] == 5
    assert summary.loc[0, "test_rows_reserved"] == 2
    assert profile.loc[0, "train_missing"] == 0
    with pytest.raises(ValueError, match="overlap"):
        fold_profiles(x, y, groups, [(np.array([0]), np.array([1, 2]))])
