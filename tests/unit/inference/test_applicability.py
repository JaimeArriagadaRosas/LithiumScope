import pandas as pd

from lithiumscope.model_1.schema import applicability_fraction, applicability_profile
from lithiumscope.model_2.schema import feature_range_profile, out_of_range_fraction


def test_model_1_applicability_flags_values_outside_training_profile():
    training = pd.DataFrame({"SiO2": [45, 50, 55, 60, 65]})
    profile = applicability_profile(training, ["SiO2"])
    candidate = pd.DataFrame({"SiO2": [1000.0]})

    fraction = applicability_fraction(candidate, profile)

    assert fraction.iloc[0] == 1.0


def test_model_2_applicability_profile_detects_out_of_range_feature():
    training = pd.DataFrame({"band_01_mean": [0.0, 1.0, 2.0, 3.0]})
    profile = feature_range_profile(training)
    candidate = pd.DataFrame({"band_01_mean": [100.0]})

    assert out_of_range_fraction(candidate, profile) == 1.0
