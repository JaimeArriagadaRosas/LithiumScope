"""Contracts for the bounded academic family and local analog comparison."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.data_lab.candidate_screening import CHEM
from lithiumscope.experiments.model_1.expansion.academic_analog import fit_analog
from lithiumscope.experiments.model_1.expansion.academic_final import BASE, POLICIES, combine, choose, run
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import DeviceInfo
from lithiumscope.resources.manager import estimator_thread_scope


def samples():
    x=pd.DataFrame({c:np.arange(1.,7.)*(i+1) for i,c in enumerate(CHEM[:10])},
                   index=[f"train{i}" for i in range(6)])
    x["Latitude"]=-30.+np.arange(6)*.01
    x["Longitude"]=-70.
    y=pd.Series([1000.,10.,20.,30.,40.,50.],index=x.index)
    return x,y


def test_analog_uses_only_train_and_excludes_near_identical_site_chemistry():
    x,y=samples()
    predictor,info=fit_analog(x,y)
    query=x.iloc[:1].set_axis(["test"])
    values,trace=predictor(query,[9.])
    assert 10.<=values[0]<=50.
    assert info["min_shared"]==8
    assert trace[0]["near_duplicate_exclusions"]==1
    assert not trace[0]["fallback"]
    assert len(trace[0]["neighbors"])==5
    assert {n["row_id"] for n in trace[0]["neighbors"]}==set(x.index[1:])


def test_analog_fallback_and_identity_protection():
    x,y=samples()
    predictor,_=fit_analog(x,y)
    query=x.iloc[:1].set_axis(["test"])
    query.loc[:,list(CHEM[:10])]=np.nan
    values,trace=predictor(query,[12.])
    assert values.tolist()==[12.]
    assert trace[0]["fallback"] and not trace[0]["neighbors"]
    with pytest.raises(ValueError,match="overlaps TRAIN"):
        predictor(x.iloc[:1],[12.])
    with pytest.raises(ValueError,match="identities"):
        fit_analog(x,y.iloc[::-1])


def test_fixed_convex_mixtures_and_invalid_predictions():
    values=combine([1.,2.],[3.,4.],[5.,6.])
    assert set(values)==set(POLICIES)
    assert values[BASE].tolist()==[1.,2.]
    assert values["hgb_catboost"].tolist()==[2.,3.]
    assert np.allclose(values["three_way"],[3.,4.])
    with pytest.raises(ValueError):
        combine([1.],[2.],[np.nan])
    with pytest.raises(ValueError):
        combine([1.],[2.,3.],[4.])


def test_inner_selection_and_research_opt_in():
    scores=pd.DataFrame([{"policy":p,"inner_fold":f,"rmse":4. if p=="three_way" else 6.,
                          "r2":.4 if p=="three_way" else .6}
                         for p in POLICIES for f in range(1,4)])
    assert choose(scores)!="three_way"
    assert choose(scores.assign(r2=.6))=="three_way"
    assert choose(scores.assign(r2=.4))==BASE
    with pytest.raises(ValueError):
        choose(scores.iloc[:-1])
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run()


def test_catboost_pipeline_handles_native_missing_numeric():
    pytest.importorskip("catboost")
    x=pd.DataFrame({"SiO2":[50.,np.nan,60.,65.,np.nan,72.],
                    "Rock_type":["a","b","a","b",np.nan,"a"]})
    y=pd.Series([5.,7.,12.,15.,20.,25.])
    schema=FeatureSchema(["SiO2"],["Rock_type"],2,"stable_v1")
    device=DeviceInfo("cpu","test",1,False,False,False)
    model=build_pipeline("catboost",device,42,schema,{"iterations":8,"depth":3})
    model.set_params(preprocess__numeric="passthrough")
    with estimator_thread_scope():
        model.fit(x,y)
    assert np.isfinite(model.predict(x)).all()
