"""Contracts and smoke tests for bounded boosting training-method comparison."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.training_methods import boosting as bm
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device


def scores():
    return pd.DataFrame([
        {"policy":p,"inner_fold":i,"rmse":7.,"r2":.63,"mae":4.,"tail_n":2,"tail_sse":1200.}
        for p in (bm.BASE,*bm.PROFILES_NEW) for i in range(1,4)
    ])


def test_fixed_profiles_and_fit_budget():
    assert bm.MAX_FITS==len(bm.PROFILES_NEW)*5*4==100
    assert bm.CAT not in bm.PROFILES_NEW and bm.BASE not in bm.PROFILES_NEW
    assert bm.PROFILES_NEW["xgb_squared"][1]["objective"]=="reg:squarederror"
    assert bm.PROFILES_NEW["xgb_pseudohuber"][1]["huber_slope"]==10
    assert bm.PROFILES_NEW["catboost_expectile"][1]["loss_function"]=="Expectile:alpha=0.65"
    for family,params in bm.PROFILES_NEW.values():
        assert "treatment_policy" not in params
        assert (params["device"]=="cpu" if family=="xgboost" else params["task_type"]=="CPU")


def test_selection_falls_back_and_checks_guards():
    s=scores()
    assert bm.choose(s)==bm.BASE
    s.loc[s.policy.eq("xgb_squared"),["rmse","tail_sse"]]=[6.,1300.]
    s.loc[s.policy.eq("xgb_regularized"),["rmse","mae"]]=[6.,4.2]
    s.loc[s.policy.eq("catboost_huber"),["rmse","r2"]]=[6.,.49]
    s.loc[s.policy.eq("catboost_expectile"),["rmse","tail_sse"]]=[6.5,1100.]
    assert bm.choose(s)=="catboost_expectile"
    with pytest.raises(ValueError):
        bm.choose(s.iloc[:-1])
    with pytest.raises(ValueError):
        bm.choose(pd.concat([s,s.iloc[[0]]],ignore_index=True))


def test_archived_partition_matches_ids_and_columns():
    x=pd.DataFrame({"a":[1,2,3]},index=["r1","r2","r3"])
    part={"fold":1,"train_ids":["r1","r2"],"test_ids":["r3"],"features":["a"]}
    bm.check_partition(part,1,x,np.array([0,1]),np.array([2]),["a"])
    with pytest.raises(ValueError):
        bm.check_partition(part,1,x,np.array([0,2]),np.array([1]),["a"])
    with pytest.raises(ValueError):
        bm.check_partition(part,1,x,np.array([0,1]),np.array([2]),["b"])


@pytest.mark.parametrize("policy",list(bm.PROFILES_NEW))
def test_real_cpu_smoke_fit(monkeypatch,policy):
    family,params=bm.PROFILES_NEW[policy]
    pytest.importorskip(family)
    params=dict(params)
    params["n_estimators" if family=="xgboost" else "iterations"]=8
    monkeypatch.setitem(bm.PROFILES_NEW,policy,(family,params))
    rng=np.random.default_rng(42)
    x=pd.DataFrame({"SiO2":rng.uniform(45,75,48),"Rock_type":["a","b"]*24})
    x.loc[3,"SiO2"]=np.nan
    y=pd.Series(np.linspace(5,90,48),index=x.index)
    schema=FeatureSchema(["SiO2"],["Rock_type"],2,"stable_v1")
    pred=bm.fit_method(x.iloc[:40],y.iloc[:40],x.iloc[40:],schema,detect_device(prefer_gpu=False),policy)
    assert pred.shape==(8,) and np.isfinite(pred).all() and (pred>=0).all()


def test_fit_rejects_overlapping_train_and_invalid_labels():
    x=pd.DataFrame({"SiO2":[50.,60.,70.]})
    y=pd.Series([5.,10.,20.])
    schema=FeatureSchema(["SiO2"],[],2,"stable_v1")
    with pytest.raises(ValueError,match="overlapping"):
        bm.fit_method(x.iloc[:2],y.iloc[:2],x.iloc[[1]],schema,None,"xgb_squared")
    with pytest.raises(ValueError,match="Invalid"):
        bm.fit_method(x.iloc[:2],pd.Series([0.,10.]),x.iloc[[2]],schema,None,"xgb_squared")


def test_xgb_backend_is_separate_from_factory_device():
    pytest.importorskip("xgboost")
    from dataclasses import replace
    # Even if the host advertises CUDA, the experimental estimator is CPU.
    device=replace(detect_device(prefer_gpu=False),xgboost_cuda_available=True)
    schema=FeatureSchema(["SiO2"],["Rock_type"],2,"stable_v1")
    before=dict(bm.PROFILES_NEW["xgb_squared"][1])
    pipeline=bm.build_method_pipeline(schema,device,"xgb_squared")
    assert pipeline.named_steps["model"].get_params()["device"]=="cpu"
    assert pipeline.named_steps["model"].get_params()["tree_method"]=="hist"
    assert pipeline.get_params()["preprocess__numeric"]=="passthrough"
    assert bm.PROFILES_NEW["xgb_squared"][1]==before
