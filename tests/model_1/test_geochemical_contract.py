import numpy as np
import pandas as pd
import pytest
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.preparation.quality_control import apply_quality_control
from lithiumscope.model_1.preparation.target_filtering import filter_target


def test_versioned_ratios_do_not_manufacture_values():
    frame = pd.DataFrame({"Na2O": [0,3,-1], "K2O": [0,4,2], "MgO": [0,2,1e-9],
                          "Fe2O3": [0,6,2], "Al2O3": [1,14,4], "CaO": [0,1,np.inf]})
    result = add_geochemical_features(frame, version=2)
    assert result.loc[0, ["Mg_Number","A_CNK_proxy","K_Mg_ratio"]].isna().all()
    assert result.loc[1,"Mg_Number"] == pytest.approx(.25)
    assert result.loc[2, ["Alkali_Sum","A_CNK_proxy","K_Mg_ratio"]].isna().all()
    assert add_geochemical_features(frame, version=1).loc[0,"A_CNK_proxy"] == pytest.approx(1e6)


def test_qc_and_target_report_exact_exclusion_reasons():
    frame = pd.DataFrame({"Li_icpms": [1, -1, np.inf, None, 0, 99], "sum": [94, 100, 100, 100, 102, 103]})
    valid, _ = filter_target(frame, ["Li_icpms"], zero_policy="exclude")
    assert valid.index.tolist() == [0,5]
    assert {r["reason"] for r in valid.attrs["target_exclusions"]} == {"negative", "nonfinite", "missing_or_unparseable", "zero_policy"}
    filtered = apply_quality_control(frame, ["sum"])
    assert 0 in filtered.index and 4 in filtered.index
    assert filtered.attrs["qc_exclusions"][0]["reason"] == "above_maximum"
