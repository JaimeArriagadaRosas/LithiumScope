"""Tests for calibration and nested TRAIN-only cross-fitting."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.integral.calibration import BASE,POLICIES,fit_calibrators
from lithiumscope.experiments.model_1.integral import posttraining as pt


def dataset():
    p=np.linspace(5.,65.,40)
    x=pd.DataFrame({"SiO2":np.linspace(45.,75.,40),"Rock_type":["a","b"]*20})
    x.loc[2,"SiO2"]=np.nan
    x.loc[4,"Rock_type"]=None
    y=pd.Series(1.15*p+2,index=x.index)
    return x,y,pd.Series(p,index=x.index)


def test_calibrators_keep_identity_and_bound_corrections():
    x,y,p=dataset()
    fn,info=fit_calibrators(x,y,p,["SiO2"],["Rock_type"])
    ev=pd.DataFrame({"SiO2":[np.nan,55.,80.],"Rock_type":["unseen",None,"a"]},index=[100,101,102])
    base=np.array([0.,25.,100.])
    pred=fn(ev,base)
    assert set(pred)==set(POLICIES)
    np.testing.assert_array_equal(pred[BASE],base)
    for v in pred.values():
        assert np.isfinite(v).all() and (v>=0).all()
        assert np.all(np.abs(v-base)<=.5*np.maximum(base,np.median(p))+1e-10)
    assert info["fitted_estimators"]==3 and info["fitted_bias_statistics"]==1


def test_additive_learns_train_bias():
    x,y,p=dataset()
    fn,_=fit_calibrators(x,y,p,["SiO2"],["Rock_type"])
    ev=x.iloc[[0]].set_axis([100])
    np.testing.assert_allclose(fn(ev,[30.])["additive"],[30.+float((y-p).mean())])


def test_calibration_rejects_misalignment_and_overlap():
    x,y,p=dataset()
    with pytest.raises(ValueError,match="unaligned"):
        fit_calibrators(x,y,p.iloc[::-1],["SiO2"],["Rock_type"])
    fn,_=fit_calibrators(x,y,p,["SiO2"],["Rock_type"])
    with pytest.raises(ValueError,match="overlaps"):
        fn(x.iloc[:2],[10.,20.])
    with pytest.raises(ValueError,match="Invalid base"):
        fn(x.iloc[:2].set_axis([100,101]),[10.,np.nan])


def test_crossfit_predicts_each_train_identity_once():
    tr=np.array([90,7,25,11,6,40,5,3,50])
    seen=[]
    def fit(a,b,j):
        assert not set(a)&set(b)
        assert set(a)|set(b)==set(tr)
        seen.extend(b.tolist())
        return b.astype(float)
    pred,parts=pt.crossfit(tr,fit,42)
    np.testing.assert_array_equal(pred,tr)
    assert len(parts)==3 and sorted(seen)==sorted(tr)


def test_crossfit_rejects_duplicate_or_nonfinite():
    with pytest.raises(ValueError):
        pt.crossfit([1,1,2],lambda a,b,j: b,42)
    with pytest.raises(ValueError):
        pt.crossfit([1,2,3],lambda a,b,j: np.full(len(b),np.nan),42)


def test_selection_guards_and_fallback():
    rows=pd.DataFrame([{"policy":p,"inner_fold":i,"rmse":7.,"r2":.63,"mae":4.,
                       "tail_n":2,"tail_sse":1200.}
                       for p in POLICIES for i in range(1,4)])
    assert pt.choose(rows)==BASE
    rows.loc[rows.policy.eq("additive"),["rmse","tail_sse"]]=[6.,1300.]
    rows.loc[rows.policy.eq("affine_shrunk"),["rmse","mae"]]=[6.,4.2]
    rows.loc[rows.policy.eq("isotonic_shrunk"),["rmse","r2"]]=[6.,.49]
    rows.loc[rows.policy.eq("residual_ridge"),["rmse","tail_sse"]]=[6.5,1100.]
    assert pt.choose(rows)=="residual_ridge"
    with pytest.raises(ValueError):
        pt.choose(rows.iloc[:-1])
