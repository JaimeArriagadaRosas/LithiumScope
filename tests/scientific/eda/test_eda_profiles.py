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
