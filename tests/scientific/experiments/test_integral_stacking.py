"""Contracts for learned stacking, residual correction, and archived OOF reuse."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.training_methods import stacking as st
from lithiumscope.experiments.model_1.studies.integration import stacking as ss
from lithiumscope.resources.detection.device import detect_device


def test_convex_weight_regularized_and_bounded():
    h=np.array([10.,20.,30.])
    x=h+5.
    assert st.convex_weight(h,h,x)==0.
    assert st.convex_weight(x,h,x)==.75
    assert 0.<st.convex_weight(h+1.,h,x)<.75
    assert st.convex_weight(h,h,h)==0.


def test_stack_outputs_bounded_and_handles_missing():
    n=120
    h=pd.Series(np.linspace(5.,65.,n))
    x=h+3.
    y=h+2.
    frame=pd.DataFrame({"SiO2":np.linspace(45.,75.,n),"Rock_type":["a","b"]*(n//2)})
    frame.loc[2,"SiO2"]=np.nan
    fn,info=st.fit_stack(frame,y,h,x,["SiO2"],["Rock_type"],detect_device(prefer_gpu=False))
    ev=pd.DataFrame({"SiO2":[np.nan,50.],"Rock_type":["unseen","a"]},index=[200,201])
    hp=np.array([10.,40.])
    xp=np.array([14.,50.])
    out=fn(ev,hp,xp)
    assert set(out)==set(st.POLICIES)
    np.testing.assert_array_equal(out[st.BASE],hp)
    np.testing.assert_allclose(out["equal_blend"],(hp+xp)/2)
    assert 0<=info["xgb_weight"]<=.75
    for p in out.values():
        assert p.shape==(2,) and np.isfinite(p).all() and (p>=0).all()
    for name in ("ridge_stack","nonlinear_residual"):
        assert np.all(np.abs(out[name]-hp)<=.5*np.maximum(hp,np.median(h))+1e-10)
    with pytest.raises(ValueError,match="overlaps"):
        fn(frame.iloc[:2],hp,xp)


def test_stack_rejects_misaligned_oof():
    h=pd.Series([5.,10.,20.])
    frame=pd.DataFrame({"SiO2":[45.,55.,60.]})
    with pytest.raises(ValueError,match="unaligned"):
        st.fit_stack(frame,h,h.iloc[::-1],h,["SiO2"],[],None)


def test_archived_oof_checks_ids_and_truth():
    frame=pd.DataFrame({"a":[1,2,3]},index=["r1","r2","r3"])
    y=pd.Series([10.,20.,30.],index=frame.index)
    table=pd.DataFrame({"fold":[1,1],"stage":["outer","outer"],"row_id":["r3","r1"],
                        "truth":[30.,10.],"oof_prediction":[27.,9.]})
    np.testing.assert_array_equal(ss.archived_oof(table,1,"outer",frame,y,np.array([0,2])),[9.,27.])
    with pytest.raises(ValueError,match="identities"):
        ss.archived_oof(table,1,"outer",frame,y,np.array([0,1]))
    table.loc[0,"truth"]=31.
    with pytest.raises(ValueError,match="values"):
        ss.archived_oof(table,1,"outer",frame,y,np.array([0,2]))


def test_inner_selection_uses_guards_and_complete_folds():
    rows=pd.DataFrame([{"policy":p,"inner_fold":i,"rmse":7.,"r2":.63,"mae":4.,
                       "tail_n":2,"tail_sse":1200.} for p in st.POLICIES for i in range(1,4)])
    assert ss.choose(rows)==st.BASE
    rows.loc[rows.policy.eq("equal_blend"),["rmse","tail_sse"]]=[6.,1300.]
    rows.loc[rows.policy.eq("convex_stack"),["rmse","mae"]]=[6.,4.2]
    rows.loc[rows.policy.eq("ridge_stack"),["rmse","r2"]]=[6.,.49]
    rows.loc[rows.policy.eq("nonlinear_residual"),["rmse","tail_sse"]]=[6.5,1100.]
    assert ss.choose(rows)=="nonlinear_residual"
    with pytest.raises(ValueError):
        ss.choose(rows.iloc[:-1])
