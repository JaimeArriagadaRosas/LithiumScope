"""Fixed spatial recovery contracts, including TRAIN-only support."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.integral import spatial_support as ss
from lithiumscope.experiments.model_1.integral import spatial_recovery as sr
from lithiumscope.experiments.data_lab.candidate_screening import CHEM
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device


def test_gate_respects_direction_and_support():
    d=pd.DataFrame({"chemistry_nn_supported":[True,False,True,False,False],
                    "high_supported":[False,False,True,False,False]})
    r=np.array([100.,100.,100.,30.,30.])
    f=np.array([20.,20.,20.,1.,100.])
    p,m=ss.apply_policies(r,f,d)
    np.testing.assert_array_equal(p["ridge_disagreement"],[20,20,20,30,30])
    np.testing.assert_array_equal(p["ridge_supported"],[100,20,100,30,30])
    np.testing.assert_array_equal(p[ss.PRIMARY],[20,20,100,30,30])
    assert m[ss.PRIMARY].sum()==2


def test_support_uses_distinct_train_groups_and_no_eval_truth():
    rng=np.random.default_rng(42)
    train=pd.DataFrame(np.repeat(rng.uniform(1,10,(20,len(CHEM))),2,axis=0),columns=CHEM)
    y=pd.Series(np.full(40,120.),index=train.index)
    groups=pd.Series(["g1","g2"]*20,index=train.index)
    ev=train.iloc[[0]].set_axis([100])
    d=ss.support(train,y,groups,ev,[100.])
    assert d.high_supported.iloc[0]
    one=ss.support(train,y,groups.replace("g2","g1"),ev,[100.])
    assert not one.high_supported.iloc[0]
    low=ss.support(train,y*0,groups,ev,[100.])
    assert not low.high_supported.iloc[0]
    with pytest.raises(ValueError,match="Invalid support TRAIN"):
        ss.support(train,y,groups,train.iloc[[0]],[100.])


def test_support_fails_closed_when_chemistry_unavailable():
    train=pd.DataFrame(np.nan,index=range(10),columns=CHEM)
    y=pd.Series(np.ones(10),index=train.index)
    g=pd.Series(["g"]*10,index=train.index)
    d=ss.support(train,y,g,train.iloc[[0]].set_axis([100]),[100.])
    assert not d.high_supported.iloc[0]
    assert d.high_support_groups.iloc[0]==0


def test_archived_alignment_rejects_changed_truth():
    x=pd.DataFrame({"a":[1,2]},index=[10,20])
    y=pd.Series([5.,8.],index=x.index)
    g=pd.Series(["g1","g2"],index=x.index)
    p=pd.DataFrame({"protocol":"academic","policy":"ridge_canonical","fold":1,
                    "row_id":["20","10"],"truth":[8.,5.],"group":["g2","g1"],
                    "prediction":[9.,6.]})
    np.testing.assert_array_equal(sr.aligned(p,"academic","ridge_canonical",1,x,y,g,[0,1]),[6,9])
    p.loc[0,"truth"]=10.
    with pytest.raises(ValueError):
        sr.aligned(p,"academic","ridge_canonical",1,x,y,g,[0,1])


def test_fixed_rf_smoke(monkeypatch):
    monkeypatch.setattr(sr,"RF_PARAMS",{**sr.RF_PARAMS,"n_estimators":5})
    x=pd.DataFrame({"SiO2":np.linspace(45,75,40),"Rock_type":["a","b"]*20})
    y=pd.Series(np.linspace(5,50,40),index=x.index)
    model=sr.fit_rf(x.iloc[:30],y.iloc[:30],FeatureSchema(["SiO2"],["Rock_type"],2,"stable_v1"),
                    detect_device(prefer_gpu=False))
    assert np.isfinite(model.predict(x.iloc[30:])).all()


def test_primary_cannot_be_replaced_by_diagnostic_winner():
    def row(protocol,policy,rmse,tail):
        return dict(protocol=protocol,policy=policy,rmse_mean=rmse,rmse_global=rmse,
                    r2_mean=.6,mae_global=4.,tail_rmse=tail,extreme_n=9,
                    extreme_rmse=tail)
    r=[row("academic","hgb_canonical",7.,25.),row("academic",ss.PRIMARY,6.9,24.),
       row("spatial","hgb_canonical",100.,500.),row("spatial","archived_rf",95.,550.),
       row("spatial","archived_svm",110.,600.),row("spatial",ss.PRIMARY,90.,560.),
       row("spatial","ridge_disagreement",80.,400.)]
    assert not sr.decisions(r)["reference_review_eligible"]
