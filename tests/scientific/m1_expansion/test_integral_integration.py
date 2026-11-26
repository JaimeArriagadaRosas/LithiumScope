"""Tests for the canonical inference contract and fixed integration evaluation."""
from types import SimpleNamespace
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from lithiumscope.experiments.model_1.integral import inference_contract as ic
from lithiumscope.experiments.model_1.integral import integration as it
from lithiumscope.experiments.model_1.integral import boosting_methods as bm
from lithiumscope.model_1.training.preprocessing.tabular import (
    FeatureSchema, OXIDES, TRACE_ELEMENTS, SPATIAL_CANDIDATES, DERIVED, CATEGORICAL,
)
from lithiumscope.model_1.pipeline import prepare_prediction_data
from lithiumscope.resources.detection.device import detect_device


def schema():
    return FeatureSchema(OXIDES+TRACE_ELEMENTS+SPATIAL_CANDIDATES+DERIVED,
                         CATEGORICAL,2,"stable_v1")


def sample(n=48):
    s=schema()
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.uniform(1.,20.,(n,len(s.numeric))),columns=s.numeric)
    x["Longitude"]=-70.
    x["Latitude"]=-25.
    for c in s.categorical:
        x[c]="unknown"
    y=pd.Series(np.linspace(0.,80.,n),index=x.index)
    return x,y,s


def test_exact_canonical_contract():
    s=schema()
    assert len(ic.canonical_columns(s))==33
    with pytest.raises(ValueError,match="33-column"):
        ic.canonical_columns(FeatureSchema(s.numeric[:-1]+["extra_ppm"],s.categorical,2,"stable_v1"))


def test_inference_parity_without_target(monkeypatch):
    x,y,s=sample()
    raw=x.drop(columns=DERIVED).assign(Li_icpms=y)
    expected=prepare_prediction_data(raw.drop(columns="Li_icpms"),"common",s)
    actual=it.prepare_prediction_data
    def verify(frame,*args):
        assert "Li_icpms" not in frame
        return actual(frame,*args)
    monkeypatch.setattr(it,"prepare_prediction_data",verify)
    p=SimpleNamespace(schema=s,target="Li_icpms",raw_frame=raw,x=expected)
    assert it.check_inference_parity(p)["parity"]
    p.x=expected.copy()
    p.x.loc[0,"SiO2"]+=1
    with pytest.raises(AssertionError):
        it.check_inference_parity(p)


def test_crossfit_exact_coverage_and_order():
    tr=np.array([30,2,19,8,41,4,15,6,21])
    folds=[(tr[a],tr[b]) for a,b in KFold(3).split(tr)]
    def fit(a,b):
        assert not set(a)&set(b)
        return b.astype(float),b.astype(float)+1
    h,x=ic.paired_crossfit(tr,folds,fit)
    np.testing.assert_array_equal(h,tr)
    np.testing.assert_array_equal(x,tr+1)
    with pytest.raises(ValueError,match="Incomplete"):
        ic.paired_crossfit(tr,folds[:-1],fit)
    with pytest.raises(ValueError,match="Repeated"):
        ic.paired_crossfit(tr,[folds[0],folds[0]],fit)


def test_ridge_correction_bounds_and_serialization():
    h=np.linspace(5.,60.,40)
    x=h+3.
    y=h+2.
    r=ic.fit_ridge(y,h,x)
    v=r.predict(h,x)
    assert np.isfinite(v).all() and (v>=0).all()
    assert np.all(np.abs(v-h)<=.5*np.maximum(h,np.median(h))+1e-10)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(r)).predict(h,x),v)
    with pytest.raises(ValueError):
        ic.fit_ridge(y,h[:-1],x)


def test_canonical_bases_real_smoke(monkeypatch):
    pytest.importorskip("xgboost")
    x,y,s=sample()
    params=ic.loss_params("squared_error")
    monkeypatch.setattr(ic,"loss_params",lambda p:{**params,"max_iter":8})
    family,profile=bm.PROFILES_NEW["xgb_regularized"]
    monkeypatch.setitem(bm.PROFILES_NEW,"xgb_regularized",(family,{**profile,"n_estimators":8}))
    models=ic.fit_bases(x.iloc[:40],y.iloc[:40],s,detect_device(prefer_gpu=False))
    h,z=ic.predict_bases(models,x.iloc[40:],s)
    assert h.shape==z.shape==(8,)
    assert np.isfinite(h).all() and np.isfinite(z).all()
    assert (h>=0).all() and (z>=0).all()


def test_reference_guard_retains_hgb_if_stack_worsens_tail():
    def row(protocol,policy,rmse,tail):
        return {"protocol":protocol,"policy":policy,"rmse_mean":rmse,"rmse_global":rmse,
                "r2_mean":.6,"tail_rmse":tail,"mae_global":4.,"extreme_n":9 if protocol=="spatial" else 0,
                "extreme_rmse":tail if protocol=="spatial" else None}
    report=[row("academic","hgb_canonical",7.,25.),row("academic","ridge_canonical",6.9,26.),
            row("spatial","hgb_canonical",100.,500.),row("spatial","ridge_canonical",99.,490.),
            row("spatial","archived_svm",150.,600.),row("spatial","archived_rf",140.,550.)]
    result=it.decision(report)
    assert result[0]["reference_review_eligible"]
    assert not result[1]["reference_review_eligible"]


def legacy_reference_fixture():
    x=pd.DataFrame({"Longitude":[-70.,-71.,-72.],"Latitude":[-25.,-26.,-27.]},index=[10,40,90])
    y=pd.Series([20.,40.,60.],index=x.index)
    groups=pd.Series(["g1","g2","g3"],index=x.index)
    folds=np.array([1,2,3])
    manifest=pd.DataFrame({"row_position":[0,1,2],"row_index":["10","40","90"],
                           "outer_fold":folds,"spatial_group":groups.to_numpy(),
                           "Longitude":x.Longitude.to_numpy(),"Latitude":x.Latitude.to_numpy()})
    pred=pd.DataFrame({"row_index":["10","40","90"],"Li_icpms_real":y.to_numpy(),
                       "Li_icpms_predicted":[19.,38.,57.]})
    return pred,manifest,x,y,groups,folds


def test_legacy_reference_recovers_identity_without_row_position():
    p,m,x,y,g,f=legacy_reference_fixture()
    np.testing.assert_array_equal(it.align_spatial_reference(p.iloc[::-1],m.iloc[::-1],x,y,g,f),
                                  [19.,38.,57.])


def test_modern_reference_verifies_redundant_identity():
    p,m,x,y,g,f=legacy_reference_fixture()
    modern=p.merge(m,on="row_index",validate="one_to_one")
    np.testing.assert_array_equal(it.align_spatial_reference(modern,m,x,y,g,f),[19.,38.,57.])
    modern.loc[0,"outer_fold"]=3
    with pytest.raises(ValueError,match="redundant identity"):
        it.align_spatial_reference(modern,m,x,y,g,f)


def test_reference_rejects_invalid_ids_truth_and_manifest():
    p,m,x,y,g,f=legacy_reference_fixture()
    for key,value in (("row_index","missing"),("row_index","40"),("Li_icpms_real",21.)):
        invalid=p.copy()
        invalid.loc[0,key]=value
        with pytest.raises(ValueError):
            it.align_spatial_reference(invalid,m,x,y,g,f)
    invalid=m.copy()
    invalid.loc[0,"Longitude"]=-71.
    with pytest.raises(ValueError,match="manifest identity"):
        it.align_spatial_reference(p,invalid,x,y,g,f)
