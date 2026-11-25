import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.evaluation.domain import fit_domain_profile, evaluate_domain
from lithiumscope.model_1.evaluation.group_intervals import group_calibration_radius, run_group_interval_study


def test_joint_domain_detects_incompatible_combination_and_missingness():
    frame = pd.DataFrame({"a": np.arange(20.), "b": np.arange(20.), "rock": ["known"]*20})
    profile = fit_domain_profile(frame, ["a","b"], ["rock"])
    query = pd.DataFrame({"a":[9.,0.,np.nan,9.], "b":[9.,19.,np.nan,9.],
                          "rock":["known","known","known","novel"]})
    table = evaluate_domain(query,profile)
    assert table.multivariate_warning.tolist() == ["OK","OUT_OF_DOMAIN","INSUFFICIENT_INPUT","OUT_OF_DOMAIN"]


def test_group_radius_uses_group_maxima_not_row_count():
    radius,n = group_calibration_radius([1,2,4,9],[1,0,1,2],["a","a","b","c"],alpha=.25)
    assert (radius,n)==(7.,3)
    with pytest.raises(ValueError,match="Too few"):
        group_calibration_radius([1,2],[1,1],["a","b"],alpha=.1)


def test_group_study_separates_fit_calibration_and_evaluation():
    groups=pd.Series(np.repeat(np.arange(50),2))
    x=pd.DataFrame({"x":np.arange(100.)})
    y=x.x.copy()
    called=[]
    def fit(x,y,g):
        called.extend(g.unique())
        return object()
    report,table=run_group_interval_study(x,y,groups,fit=fit,predict=lambda m,x:x.x.to_numpy(),execute=False)
    assert report["state"]=="preflight" and table is None and not called
    report,table=run_group_interval_study(x,y,groups,fit=fit,predict=lambda m,x:x.x.to_numpy(),execute=True)
    assert report["state"]=="complete" and report["row_coverage"]==1.
    assert not set(called)&set(table.group)
    assert report["radius_ppm"]==0.
