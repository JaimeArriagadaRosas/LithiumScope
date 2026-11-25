import numpy as np
import pandas as pd

from lithiumscope.experiments.generalization.signature_policy import (
    signature_weights, signature_training, signature_summary, signature_overlap,
)


def test_soft_signature_preserves_empirical_class_mass():
    x=pd.DataFrame({"a":[1.,1.,1.,2.,3.]})
    y=pd.Series([0,1,1,0,1])
    g=pd.Series(["a","a","a","b","c"])
    weights=signature_weights(x)
    assert np.allclose(weights,[1/3,1/3,1/3,1,1])
    xx,yy,w,trace=signature_training(x,x.to_numpy(),y,g,"signature_soft")
    assert len(xx)==4 and np.isclose(w.sum(),3)
    assert np.isclose(np.dot(w,yy),np.dot(weights,y))
    assert len(trace)==len(x)
    assert signature_summary(x,y)["unavoidable_errors_same_signature"]==1
    assert signature_overlap(x.iloc[:3],x.iloc[3:])==0
