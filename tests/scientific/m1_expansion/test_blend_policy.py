"""Guard transfer policy choice on aligned inner OOF, never on outer labels."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.treatments.blend_policy import BASE, POLICIES, combine, select_inner


def test_prediction_limits_use_only_predictions_and_train_scale():
    base=np.array([10.,100.,200.])
    expanded=np.array([100.,300.,0.])
    assert np.allclose(combine(base,expanded,"H_weight_010_blend_25",20.),[32.5,150.,150.])
    assert np.allclose(combine(base,expanded,"I_weight_025_cap_0.5",20.),[20.,150.,100.])
    assert np.array_equal(combine(base,None,BASE,20.),base)
    with pytest.raises(ValueError):
        combine(base,[1.],"H_weight_010_blend_25",20.)


def inner_table():
    truth=np.array([20.]*5+[300.]*5+[1500.]*2)
    rows=[]
    for policy in POLICIES:
        err=np.full(len(truth),10.)
        if policy=="H_weight_010_blend_50":
            err[:]=5.
        if policy=="I_weight_025_blend_100":
            err[:]=1.
            err[5:10]=12.
        rows.extend({"position":i,"inner_fold":i%3+1,"truth":v,"policy":policy,
                     "prediction":v+err[i]} for i,v in enumerate(truth))
    return pd.DataFrame(rows)


def test_inner_selection_rejects_middle_band_deterioration():
    selected,detail=select_inner(inner_table())
    assert selected=="H_weight_010_blend_50"
    assert not next(v for v in detail if v["policy"]=="I_weight_025_blend_100")["guard_pass"]


def test_inner_control_is_fallback_without_middle_band_support():
    data=inner_table()
    assert select_inner(data.loc[data.truth.ne(300.)])[0]==BASE


def test_inner_rejects_missing_or_misaligned_oof():
    data=inner_table()
    with pytest.raises(ValueError,match="identities"):
        select_inner(data.drop(index=data.index[-1]))
    with pytest.raises(ValueError,match="Incomplete"):
        select_inner(data.loc[data.policy.ne(BASE)])


def test_nested_requires_explicit_research_metadata_opt_in():
    from lithiumscope.experiments.model_1.studies.expansion.nested import run
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run("nonexistent")
