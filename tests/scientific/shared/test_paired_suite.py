import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

from lithiumscope.experiments.shared import paired_suite
from lithiumscope.experiments.shared.paired_comparison import paired_group_intervals
from lithiumscope.experiments.shared.paired_suite import Arm, run_paired_suite


def cohort():
    groups = pd.Series(np.repeat(np.arange(6), 4))
    y = pd.Series(np.tile([1., 2., 3., 4.], 6))
    x = pd.DataFrame({"a": y, "b": y+1})
    return x, y, groups, list(GroupKFold(3).split(x, y, groups))


def test_preflight_preserves_all_test_rows_and_never_fits(tmp_path, monkeypatch):
    x,y,g,splits = cohort()
    mask = np.ones(len(y),dtype=bool)
    mask[::6] = False
    monkeypatch.setattr(paired_suite, "evaluation_identity", lambda *a,**k:"synthetic")
    monkeypatch.setattr(paired_suite, "runtime_fingerprint", lambda:{})
    result = run_paired_suite(arms=[Arm("ref","dummy",None,x,{}),Arm("qa","dummy","ref",x,{},mask)],
        y=y,groups=g,splits=splits,kind="regression",output=tmp_path/"run",
        fit=lambda *a: pytest.fail("fit in preflight"),predict=None)
    assert result["state"] == "preflight"
    for part in result["partitions"]:
        assert part["test"] == splits[part["fold"]-1][1].tolist()
        if part["arm"] == "qa":
            assert mask[part["train"]].all()


def test_failed_arm_never_compares_a_successful_subset(tmp_path,monkeypatch):
    x,y,g,splits = cohort()
    monkeypatch.setattr(paired_suite,"evaluation_identity",lambda *a,**k:"synthetic")
    monkeypatch.setattr(paired_suite,"runtime_fingerprint",lambda:{})
    def fit(arm,fold,a,b,groups,raw):
        assert not set(groups)&set(g.iloc[splits[fold-1][1]])
        if arm.name=="broken" and fold==2:
            raise RuntimeError("synthetic")
        return DummyRegressor().fit(a,b)
    out=tmp_path/"run"
    report=run_paired_suite(arms=[Arm("ref","dummy",None,x,{}),Arm("broken","dummy","ref",x,{}),
                                 Arm("same","dummy","ref",x[["a"]],{})],
        y=y,groups=g,splits=splits,kind="regression",output=out,fit=fit,
        predict=lambda m,x:m.predict(x),execute=True,bootstrap_repeats=10)
    assert report["state"]=="incomplete"
    intervals=pd.read_csv(out/"paired_intervals.csv")
    assert intervals.loc[intervals.arm=="broken","state"].item()=="incomplete_oof"
    assert (intervals.loc[intervals.arm=="same","delta_alternative_minus_reference"]==0).all()
    assert not list(out.glob("*.joblib"))


def test_bootstrap_rejects_identity_drift_and_uses_group_resampling():
    x,y,g,splits=cohort()
    rows=pd.DataFrame({"position":np.arange(len(y)),"fold":np.repeat([1,2,3],8),
                       "group":g.astype(str),"truth":y,"prediction":y+2})
    alt=rows.assign(prediction=y+1)
    result=paired_group_intervals(rows,alt,kind="regression",repeats=20)
    assert all(r["delta_alternative_minus_reference"]==-1 for r in result)
    assert all(r["ci95_low"]==r["ci95_high"]==-1 for r in result)
    with pytest.raises(ValueError,match="identities"):
        paired_group_intervals(rows,alt.assign(fold=99),kind="regression",repeats=20)
