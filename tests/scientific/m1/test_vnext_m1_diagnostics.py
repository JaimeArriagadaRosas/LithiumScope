import numpy as np
import pytest

from lithiumscope.experiments.analysis.run_vnext_m1_diagnostics import (
    cluster_bootstrap_improvement,
)


def test_grouped_paired_bootstrap_detects_improvement_and_is_deterministic():
    y=np.array([1,2,3,4,5,6],dtype=float)
    old=y+2
    new=y+0.5
    groups=["A","A","B","B","C","C"]
    a=cluster_bootstrap_improvement(y,old,new,groups,repeats=70,seed=23)
    b=cluster_bootstrap_improvement(y,old,new,groups,repeats=70,seed=23)
    assert a==b
    assert a["mae_gain"]["low_95"] > 1
    assert a["rmse_gain"]["positive_fraction"]==1


def test_invalid_grouped_bootstrap_raises():
    with pytest.raises(ValueError):
        cluster_bootstrap_improvement([1],[2],[3],["A"])
