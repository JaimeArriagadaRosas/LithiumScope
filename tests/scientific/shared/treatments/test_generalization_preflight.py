import numpy as np
import pandas as pd

from lithiumscope.experiments.generalization.partitions import partitions, training_target
from lithiumscope.experiments.generalization.preflight import _prepare_one


def test_partition_preparation_does_not_read_test_targets_or_predictors():
    n=36
    x=pd.DataFrame({"a":np.arange(n,dtype=float),"b":np.sin(np.arange(n))})
    y=pd.Series(np.tile([1.,2.,3.,4.,5.,6.],6))
    groups=pd.Series(np.repeat(list("ABCDEF"),6))
    eligible=np.ones(n,dtype=bool)
    eligible[0]=False
    data=(x,y,groups,groups,eligible,list(x),[])
    cfg={"validation":{"folds":3,"inner_folds":2},"model":{"high_lithium_quantile":.75}}
    tr,te,inner=partitions("model_2",data,cfg)[0]
    fit,labels,threshold=training_target("model_2",data,cfg,tr)
    assert not set(groups.iloc[tr]) & set(groups.iloc[te])
    for a,b in inner:
        assert set(a)|set(b)==set(tr)
        assert not set(a)&set(te)
    result,trace=_prepare_one("model_2","oversample",data,cfg,tr,te,42)
    altered_x=x.copy()
    altered_y=y.copy()
    altered_x.iloc[te]=10000
    altered_y.iloc[te]=99999
    changed=(altered_x,altered_y,groups,groups,eligible,list(x),[])
    assert _prepare_one("model_2","oversample",changed,cfg,tr,te,42)==(result,trace)
    assert result["threshold_ppm"]==threshold
    assert result["test_reserved"]==len(te)
    assert all(p["parent_a"] in set(x.index[fit].astype(str)) for p in trace)
