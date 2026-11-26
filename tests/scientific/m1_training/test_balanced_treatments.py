"""Real-row reduction and regime routing contracts."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.balanced.selection import ARMS,balanced_positions,MIN_COHORT_ROWS


def test_balance_preserves_tail_and_group_source_strata():
    n=3420
    x=pd.DataFrame({"Rock_type":["rhyolite"]*n,"Rb_icpms":np.arange(n,dtype=float)})
    y=pd.Series(np.linspace(1,99,n))
    y.iloc[-15:]=np.linspace(100,4160,15)
    groups=pd.Series(np.arange(n)//40)
    sources=pd.Series(np.where(np.arange(n)%2==0,"a","b"))
    keep=balanced_positions(x,y,groups,sources,MIN_COHORT_ROWS,42)
    assert len(keep)==2043 and len(np.unique(keep))==2043
    assert set(np.flatnonzero(y>=100))<=set(keep)
    assert set(zip(groups,sources))==set(zip(groups.iloc[keep],sources.iloc[keep]))
    np.testing.assert_array_equal(keep,balanced_positions(x,y,groups,sources,MIN_COHORT_ROWS,42))
    with pytest.raises(ValueError):
        balanced_positions(x,y,groups,sources,681,42)


def test_balance_campaign_count():
    assert len(ARMS)==7 and len({a.name for a in ARMS})==7


def test_regime_experts_predict_without_target_column():
    from lithiumscope.experiments.model_1.balanced.runner import fit_balanced
    from lithiumscope.resources.detection.device import detect_device
    rng=np.random.default_rng(42)
    x=pd.DataFrame({"Rb_icpms":rng.uniform(0,300,300),"Sr_icpms":rng.uniform(0,100,300)})
    y=pd.Series(x.Rb_icpms*.7,index=x.index)
    groups=pd.Series(np.arange(300)//5,index=x.index)
    sources=pd.Series("synthetic",index=x.index)
    params={"n_estimators":5,"max_depth":3,"treatment_policy":"rare_group_sqrt"}
    arm=next(a for a in ARMS if a.name=="experts_fixed")
    prediction,train,info=fit_balanced("random_forest",arm,detect_device(prefer_gpu=False),42,
        params,x,y,groups,sources,list(x.columns),[],x.iloc[:5])
    assert prediction.shape==(5,) and train.shape==(300,)
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    assert info["router"]=="soft_RF_X_only"
