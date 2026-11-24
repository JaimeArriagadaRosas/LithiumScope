import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.preparation.target_filtering import filter_target
from lithiumscope.model_1.preparation.missing_values import should_include_age
from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples


def test_m1_physical_target_keeps_extreme_nonnegative_values():
    raw = pd.DataFrame({"Li_icpms": [-1, np.inf, -np.inf, None, "bad", 0, 1, 9999]})
    result, target = filter_target(raw, ["Li_icpms"])
    assert result[target].tolist() == [0, 1, 9999]
    assert result.attrs["target_validation"]["excluded_negative"] == 1
    assert result.attrs["target_validation"]["excluded_nonfinite"] == 2
    assert result.attrs["target_validation"]["excluded_missing_or_unparseable"] == 2
    assert result.attrs["target_validation"]["zero_policy"] == "retain"


def test_m1_zero_policy_explicit():
    raw = pd.DataFrame({"Li_icpms": [0, 1, 9999]})
    result, _ = filter_target(raw, ["Li_icpms"], zero_policy="exclude")
    assert result["Li_icpms"].tolist() == [1, 9999]
    with pytest.raises(ValueError):
        filter_target(raw, ["Li_icpms"], zero_policy="guess")


def test_age_coverage_counts_only_physical_numeric_values():
    assert not should_include_age(pd.DataFrame({"Age": ["unknown", -1, np.inf, 5]}))
    assert should_include_age(pd.DataFrame({"Age": ["10", None, 0, np.inf]}))


def test_m2_rejects_negative_and_retains_zero(tmp_path):
    path = tmp_path / "samples.csv"
    pd.DataFrame({
        "Sample": ["neg", "zero", "high", "inf"],
        "Li_icpms": [-1, 0, 9000, np.inf],
        "Longitude": [-70, -71, -72, -73],
        "Latitude": [-20, -21, -22, -23],
    }).to_csv(path, index=False)
    result = load_georeferenced_li_samples(path)
    assert result["Li_icpms"].tolist() == [0, 9000]
