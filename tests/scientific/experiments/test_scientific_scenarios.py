import numpy as np
import pandas as pd
from lithiumscope.experiments.scenarios.scenarios import Scenario, catalog, perturb, GROUPS


def sample():
    return pd.DataFrame({"case_id":["a","b","c"],"Li_icpms":[1.,2.,3.],
        "SiO2":[50.,60.,70.],"MgO":[1.,2.,3.],"Longitude":[-70.,-69.,-68.],
        "Latitude":[-23.,-22.,-21.],"Rock_type":["a","b","c"]})


def test_catalog_deterministic():
    assert catalog() == catalog()
    assert len(catalog()) == 18


def test_baseline_identity_and_no_mutation():
    original = sample()
    result = perturb(original,Scenario("baseline","baseline",0))
    pd.testing.assert_frame_equal(result,original)
    assert result is not original


def test_missing_preserves_labels_identifiers():
    frame = sample()
    result = perturb(frame,Scenario("missing_1","missing",1))
    assert result.drop(columns=["case_id","Li_icpms"]).isna().all().all()
    pd.testing.assert_series_equal(result["Li_icpms"],frame["Li_icpms"])
    pd.testing.assert_series_equal(result["case_id"],frame["case_id"])


def test_group_drops_only_selected_columns():
    frame = sample()
    result = perturb(frame,Scenario("group_oxides","group",1))
    assert result["SiO2"].isna().all()
    assert result["MgO"].isna().all()
    assert result["Longitude"].notna().all()


def test_noise_reproducible_without_touching_coordinates():
    frame = sample()
    scenario = Scenario("noise","noise",0.1,seed=13)
    pd.testing.assert_frame_equal(perturb(frame,scenario),perturb(frame,scenario))
    pd.testing.assert_series_equal(perturb(frame,scenario)["Longitude"],frame["Longitude"])


def test_extreme_never_changes_ground_truth():
    frame = sample()
    result = perturb(frame,Scenario("extreme","extreme",3))
    assert np.allclose(result["SiO2"],frame["SiO2"]*3)
    pd.testing.assert_series_equal(result["Li_icpms"],frame["Li_icpms"])
