"""HGB loss changes are isolated; selection sees only the inner scores."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.integral import training_losses as tl
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device


def table():
    rows=[]
    for policy in tl.LOSSES:
        for fold in range(1,4):
            rows.append({"policy":policy,"inner_fold":fold,"rmse":7.,"r2":.63,
                         "mae":4.,"tail_n":2,"tail_sse":1200.})
    return pd.DataFrame(rows)


def test_loss_params_only_change_registered_loss():
    base=tl.loss_params(tl.BASE)
    for policy,options in tl.LOSSES.items():
        params=tl.loss_params(policy)
        assert {k:v for k,v in params.items() if k not in ("loss","quantile")}=={
            k:v for k,v in base.items() if k not in ("loss","quantile")}
        assert all(params[k]==v for k,v in options.items())
        assert params["early_stopping"] is False
    with pytest.raises(ValueError):
        tl.loss_params("unknown")


def test_select_loss_requires_tail_mae_and_r2_guards():
    rows=table()
    rows.loc[rows.policy.eq("gamma"),["rmse","tail_sse"]]=[6.,1300.]
    rows.loc[rows.policy.eq("poisson"),["rmse","mae"]]=[6.,4.2]
    rows.loc[rows.policy.eq("absolute_error"),["rmse","r2"]]=[6.,.49]
    rows.loc[rows.policy.eq("quantile_060"),["rmse","tail_sse"]]=[6.5,1100.]
    assert tl.select_loss(rows)=="quantile_060"


def test_select_loss_fallback_and_missing_inner_fold():
    rows=table()
    assert tl.select_loss(rows)==tl.BASE
    with pytest.raises(ValueError):
        tl.select_loss(rows.iloc[:-1])
    with pytest.raises(ValueError):
        tl.select_loss(pd.concat([rows,rows.iloc[[0]]],ignore_index=True))


@pytest.mark.parametrize("policy",list(tl.LOSSES))
def test_native_loss_fits_without_inventing_labels(monkeypatch,policy):
    original=tl.loss_params
    monkeypatch.setattr(tl,"loss_params",lambda p:{**original(p),"max_iter":8})
    rng=np.random.default_rng(42)
    x=pd.DataFrame({"SiO2":rng.uniform(45,75,48),"Rock_type":["a","b"]*24})
    x.loc[3,"SiO2"]=np.nan
    y=pd.Series(np.linspace(5,90,48),index=x.index)
    schema=FeatureSchema(["SiO2"],["Rock_type"],2,"stable_v1")
    pred=tl.fit_loss(x.iloc[:40],y.iloc[:40],x.iloc[40:],schema,detect_device(prefer_gpu=False),policy)
    assert pred.shape==(8,) and np.isfinite(pred).all() and (pred>=0).all()


def test_fit_rejects_overlap_and_invalid_positive_support():
    x=pd.DataFrame({"SiO2":[50.,60.,70.]})
    y=pd.Series([5.,10.,20.])
    schema=FeatureSchema(["SiO2"],[],2,"stable_v1")
    with pytest.raises(ValueError,match="overlapping"):
        tl.fit_loss(x.iloc[:2],y.iloc[:2],x.iloc[[1]],schema,None,tl.BASE)
    with pytest.raises(ValueError,match="Invalid"):
        tl.fit_loss(x.iloc[:2],pd.Series([0.,10.]),x.iloc[[2]],schema,None,"gamma")
