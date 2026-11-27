"""Academic HGB treatment contracts."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.treatments.academic import (
    BASE, PROFILES_HGB, training_weights, fit_profile, select_profile)
from lithiumscope.experiments.model_1.studies.expansion.academic_refinement import run
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import DeviceInfo


def test_weights_preserve_historical_q90_and_attenuate_candidate_tail():
    y=pd.Series([1.,2.,3.,4.,5.,100.])
    candidate=pd.Series([False]*5+[True])
    w,q=training_weights(y,candidate,2.,.1)
    assert q==pytest.approx(4.6)
    assert w[:5].mean()==pytest.approx(1.)
    assert w[-1]==pytest.approx(.1*w[4])
    with pytest.raises(ValueError):
        training_weights(y,candidate,2.,0.)


def test_native_hgb_accepts_missing_predictors():
    idx=pd.Index([str(i) for i in range(40)])
    x=pd.DataFrame({"SiO2":np.linspace(40,75,40),"MgO":[np.nan,1.]*20,
                    "Rock_type":["granite","rhyolite"]*20},index=idx)
    y=pd.Series(np.linspace(5,100,40),index=idx)
    mask=pd.Series([False]*35+[True]*5,index=idx)
    schema=FeatureSchema(["SiO2","MgO"],["Rock_type"],2,"stable_v1")
    device=DeviceInfo("cpu","test",1,False,False,False)
    pred,info=fit_profile(x,y,mask,schema,device,42,"native_tail_2_transfer_010")
    assert info["candidate_rows"]==5 and info["native_missing"]
    assert np.isfinite(pred(x)).all() and (pred(x)>=0).all()


def test_inner_selection_requires_r2_and_complete_scores():
    rows=[]
    for p in PROFILES_HGB:
        for fold in range(1,4):
            rows.append({"profile":p,"inner_fold":fold,"rmse":5. if p=="native_tail_2" else 7.,
                         "r2":.6 if p=="native_tail_2" else .4})
    scores=pd.DataFrame(rows)
    assert select_profile(scores)[0]=="native_tail_2"
    assert select_profile(scores.assign(r2=.4))[0]==BASE
    with pytest.raises(ValueError):
        select_profile(scores.iloc[:-1])


def test_fixed_budget_and_explicit_opt_in():
    assert 5*(3*len(PROFILES_HGB)+1)==110
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run("missing")
