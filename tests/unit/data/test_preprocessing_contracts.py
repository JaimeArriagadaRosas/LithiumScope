import numpy as np
import pytest
from sklearn.svm import SVC
from lithiumscope.model_2.training.preprocessing.contracts import inner_fold_labels, eligible_positions, scaled_estimator

def test_label_endpoint_is_train_only():
    a,b,t=inner_fold_labels([1.,2.,3.,10000.],[0,1,2],[3],quantile=.5)
    assert t==2 and a.tolist()==[0,1,1] and b.tolist()==[1]
    with pytest.raises(ValueError):
        inner_fold_labels([1.,2.,3.],[0,1],[1,2])

def test_mask_rejects_implicit_integer_conversion():
    assert eligible_positions([True,False,True],3).tolist()==[0,2]
    with pytest.raises(ValueError):
        eligible_positions([1,0,1],3)

def test_scaler_is_fitted_inside_pipeline():
    pipeline=scaled_estimator(SVC())
    assert not hasattr(pipeline.named_steps["scaler"],"mean_")
    pipeline.fit([[0.],[1.],[3.],[4.]],[0,0,1,1])
    assert pipeline.named_steps["scaler"].mean_[0]==2
