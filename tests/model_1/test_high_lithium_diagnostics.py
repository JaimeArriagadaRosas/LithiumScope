import numpy as np
import pandas as pd
import pytest
from lithiumscope.model_1.evaluation.high_lithium import high_lithium_diagnostics


def example():
    return pd.DataFrame({"row_position":[0,1,2,3,4],"outer_fold":[1,1,2,2,2],
        "fold_train_q75_ppm":[10.,10.,20.,20.,20.],"Li_icpms_real":[10.,5.,25.,15.,20.],
        "Li_icpms_predicted":[9.,12.,21.,10.,20.],"observed_high":[1,0,1,0,1],
        "predicted_high":[0,1,1,0,1],"source_dataset":["a","a","b","b","b"]})


def test_fold_specific_confusion_and_strata():
    table,strata=high_lithium_diagnostics(example())
    pooled=table.iloc[-1]
    assert pooled.scope=="pooled_fold_specific_endpoints"
    assert (pooled.tn,pooled.fp,pooled.fn,pooled.tp)==(1,1,1,2)
    assert np.isclose(pooled.sensitivity,2/3)
    assert table.iloc[0].threshold_ppm==10
    assert table.iloc[1].threshold_ppm==20
    assert strata.n.sum()==5


def test_single_class_is_undefined_not_perfect():
    table,_=high_lithium_diagnostics(example().iloc[[0]])
    assert np.isnan(table.iloc[-1].specificity)
    assert np.isnan(table.iloc[-1].balanced_accuracy)


def test_rejects_duplicate_flags_and_changed_fold_threshold():
    f=example()
    with pytest.raises(ValueError,match="duplicated"):
        high_lithium_diagnostics(pd.concat([f,f]))
    f.loc[0,"predicted_high"]=1
    with pytest.raises(ValueError,match="flags"):
        high_lithium_diagnostics(f)
    f=example()
    f.loc[1,"fold_train_q75_ppm"]=11
    with pytest.raises(ValueError,match="constant"):
        high_lithium_diagnostics(f)
