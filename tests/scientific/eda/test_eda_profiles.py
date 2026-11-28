import numpy as np
import pandas as pd

from lithiumscope.model_1.eda.statistics import target_tails
from lithiumscope.model_2.eda.statistics import redundancy_table,period_summary


def test_m1_target_tail_counts_invalid_without_removing_extremes():
    frame=pd.DataFrame({"Li_icpms":[0,10,-1,4000,np.nan],"source_dataset":["A","A","A","A","A"]})
    row=target_tails(frame,"Li_icpms",stage="raw").iloc[0]
    assert row.finite_nonnegative==3 and row.negative==1 and row.nonfinite_or_missing==1
    assert row.maximum==4000


def test_m2_redundancy_requires_identical_numeric_values_and_missingness():
    frame=pd.DataFrame({"a":[1.,2.,np.nan],"b":[1.,2.,np.nan],"c":[1.,np.nan,2.],"d":[0.,-0.,0.]})
    data=redundancy_table(frame,list(frame))
    assert data.loc[data.feature=="b","reference"].item()=="a"
    assert "c" not in set(data.feature)
    assert data.loc[data.feature=="d","state"].item()=="constant_or_all_missing"


def test_period_profiles_preserve_unknown_dates_and_local_qa():
    frame=pd.DataFrame({"sentinel_datetime":["2024-01-01T12:00:00Z","bad","2024-01-03"],
                        "Li_icpms":[1.,2.,3.],"sixband_joint_valid_fraction":[.9,.5,1.]})
    data=period_summary(frame,"Li_icpms").set_index("period_utc")
    assert data.loc["2024-01","samples"]==2
    assert np.isclose(data.loc["2024-01","mean_local_valid_fraction"],.95)
    assert data.loc["unknown","samples"]==1


def test_m1_coverage_reports_source_dependence_and_missing_composition():
    from types import SimpleNamespace
    from lithiumscope.model_1.eda.representativeness import coverage_tables
    x=pd.DataFrame({"SiO2":[50.,51.,52.,53.],"Rock_type":["unknown","unknown","granite","granite"]})
    frame=x.assign(source_dataset=["A","A","B","B"])
    prepared=SimpleNamespace(x=x, y=pd.Series([0.,10.,50.,700.]), frame=frame,
        groups=pd.Series(["g1","g1","g2","g3"]),
        schema=SimpleNamespace(numeric=["SiO2"],categorical=["Rock_type"]))
    report=coverage_tables(prepared)
    assert report["target_bands_by_source"].rows.sum()==4
    assert report["categorical_source_association"].cramers_v_source.iloc[0]==1.
    assert report["compositional_readiness"].incomplete.sum()==4
    assert not report["compositional_readiness"].ilr_applied.any()
    assert report["missingness_patterns"].rows.sum()==4


def test_feature_profiles_support_boolean_review_flags_and_nullable_values():
    from lithiumscope.core.scientific.eda_statistics import feature_profiles
    frame = pd.DataFrame({"source_dataset": ["candidate"] * 4,
        "qc": [True, False, True, False],
        "nullable": pd.Series([True, pd.NA, False, True], dtype="boolean"),
        "Li_icpms": [10., 20., 30., 40.]})
    result = feature_profiles(frame, ["qc", "nullable", "Li_icpms"], stage="candidate").set_index("feature")
    assert result.loc["qc", "median"] == .5
    assert result.loc["qc", "zero"] == 2
    assert result.loc["nullable", "missing"] == 1
    assert result.loc["nullable", "numeric_finite"] == 3
    assert np.isclose(result.loc["nullable", "mean"], 2/3)
    assert result.loc["Li_icpms", "median"] == 25.
    assert str(frame.nullable.dtype) == "boolean"
