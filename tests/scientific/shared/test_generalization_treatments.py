import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.generalization.prepared import load_snapshot, save_snapshot
from lithiumscope.experiments.generalization.resampling import classification_resample, regression_resample


def test_prepared_snapshot_preserves_rows_and_detects_tampering(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("original")
    x = pd.DataFrame({"a": [1.0, np.nan, 3.5], "cat": ["a", None, "b"]}, index=["r0","r1","r2"])
    y = pd.Series([1., 2., 3.], index=x.index)
    groups = pd.Series(["a","b","c"], index=x.index)
    path = save_snapshot(tmp_path/"versions", "key", "model_1", source, x, y, groups, groups,
                         np.array([True, True, True]), ["a"], ["cat"])
    loaded = load_snapshot(path, "key")
    assert list(loaded[0].index) == list(x.index)
    assert np.isnan(loaded[0].loc["r1","a"])
    assert loaded[1].tolist() == y.tolist()
    (path/"prepared.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_snapshot(path, "key")
    assert source.read_text() == "original"


def test_group_smote_parent_trace_and_no_test_access():
    x = pd.DataFrame({"a": [0.,1.,2.,3.,4.,5.,6.,7.], "b": [1.,2.,3.,4.,5.,6.,7.,8.]},
                     index=[f"r{i}" for i in range(8)])
    y = pd.Series([1,1,0,0,0,0,0,0], index=x.index)
    groups = pd.Series(["A","A","A","B","B","C","C","D"], index=x.index)
    xx, yy, trace = classification_resample(x,y,groups,"smote_group",42)
    assert len(xx) == 11 and sum(yy) == 5
    assert all({p["parent_a"],p["parent_b"]} == {"r0","r1"} and p["group"] == "A" for p in trace)
    assert np.array_equal(xx[:len(x)],x.to_numpy())
    assert np.all((xx[8:,0] > 0) & (xx[8:,0] < 1))
    with pytest.raises(ValueError, match="no distinct"):
        classification_resample(x,y,pd.Series(list("ABCDEFGH"),index=x.index),"smote_group",42)


def test_regression_bootstrap_copies_real_values_and_groups():
    x = pd.DataFrame({"a":range(8)})
    y = pd.Series(range(8),dtype=float)
    groups = pd.Series(list("AABBCCDD"))
    xx, yy, gg, trace = regression_resample(x,y,groups,groups,"rare_bootstrap",42)
    assert len(xx) == 10 and len(trace) == 2
    assert yy.iloc[-2:].tolist() == [7.,7.]
    assert gg.iloc[-2:].tolist() == ["D","D"]

def test_smoter_uses_same_source_and_recomputes_ratios():
    from lithiumscope.experiments.generalization.resampling import regression_smoter
    n = 30
    x = pd.DataFrame({"Latitude": [-20.]*n,"Longitude": [-70.]*n,
        "MgO": np.linspace(1,3,n),"Fe2O3": np.linspace(2,4,n),
        "Mg_Number": [-999.]*n,"Rock_type": ["A"]*n})
    y = pd.Series(np.arange(n,dtype=float))
    groups = pd.Series(["g"]*n)
    sources = pd.Series(["source"]*n)
    xx, yy, gg, trace = regression_smoter(x,y,groups,sources,42,
        ["Latitude","Longitude","MgO","Fe2O3","Mg_Number"],["Rock_type"])
    assert len(trace) == n//4
    assert len(xx) == n+len(trace)
    assert (yy.iloc[n:] >= y.quantile(.90)).all()
    assert (yy.iloc[n:] <= y.max()).all()
    assert gg.iloc[n:].eq("g").all()
    assert np.allclose(xx.iloc[n:].Mg_Number,xx.iloc[n:].MgO/(xx.iloc[n:].MgO+xx.iloc[n:].Fe2O3))
    assert xx.iloc[:n].Mg_Number.eq(-999).all()
    with pytest.raises(ValueError,match="no compatible"):
        regression_smoter(x,y,groups,pd.Series([str(i) for i in range(n)]),42,
            ["Latitude","Longitude","MgO","Fe2O3","Mg_Number"],["Rock_type"])


def test_rare_group_bootstrap_parent_limit():
    from collections import Counter
    n=100
    x=pd.DataFrame({"a":np.arange(n,dtype=float)})
    y=pd.Series(np.arange(n,dtype=float))
    g=pd.Series(["A"]*95+["B"]*5)
    xx, yy, gg, trace=regression_resample(x,y,g,g,"rare_group_bootstrap",42)
    assert len(xx)==125 and len(yy)==len(gg)==len(xx)
    assert max(Counter(t["row"] for t in trace).values())<=4
    assert (yy.iloc[n:]>=y.quantile(.9)).all()


def test_smoter_stops_at_pair_cap():
    from lithiumscope.experiments.generalization.resampling import regression_smoter
    n=100
    x=pd.DataFrame({"Latitude":[-20.]*n,"Longitude":[-70.]*n,
        "MgO":np.linspace(1,3,n),"Fe2O3":np.linspace(2,4,n),
        "Rock_type":[str(i) for i in range(n-2)]+["compatible","compatible"]})
    y=pd.Series(np.arange(n,dtype=float))
    g=pd.Series(["g"]*n)
    xx, yy, gg, trace=regression_smoter(x,y,g,g,42,list(x.columns[:-1]),["Rock_type"])
    assert len(xx)==104 and len(trace)==4
    assert len(yy)==len(gg)==len(xx)
    assert all({t["parent_a"],t["parent_b"]}=={"98","99"} for t in trace)
