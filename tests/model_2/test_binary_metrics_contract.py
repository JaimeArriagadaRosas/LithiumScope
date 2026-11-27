import numpy as np
import pytest
from lithiumscope.core.scientific.classification_metrics import classification_metrics,reliability_table
from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold


def test_exact_confusion_counts_and_metrics():
    m=classification_metrics([0,0,0,1,1,1],[.1,.3,.7,.4,.8,.9])
    assert (m["tn"],m["fp"],m["fn"],m["tp"])==(2,1,1,2)
    assert m["accuracy"]==pytest.approx(2/3)
    assert m["balanced_accuracy"]==pytest.approx(2/3)
    assert m["mcc"]==pytest.approx(1/3)
    assert m["positive_support"]==m["negative_support"]==3


def test_single_class_not_validated_as_perfect():
    with pytest.raises(ValueError,match="Both classes"):
        classification_metrics([0,0],[.1,.2])
    m=classification_metrics([0,0],[.1,.2],allow_single_class=True)
    assert m["accuracy"]==1 and m["tn"]==2
    assert np.isnan(m["balanced_accuracy"]) and np.isnan(m["roc_auc"])
    assert np.isnan(m["average_precision"]) and np.isnan(m["sensitivity"])
    assert select_operating_threshold([0,0],[.1,.2]).state=="unavailable_single_class"


@pytest.mark.parametrize("y,p", [([0,2],[.1,.2]),([0,.5],[.1,.2]),([0,1],[.1,np.inf]),
                                  ([0,1],[.1,1.1]),([0,1],[.1]),([[0],[1]],[.1,.2])])
def test_invalid_arrays_fail_metrics_and_threshold(y,p):
    with pytest.raises(ValueError):
        classification_metrics(y,p)
    with pytest.raises(ValueError):
        select_operating_threshold(y,p)


def test_reliability_includes_endpoints_and_empty_bins():
    r=reliability_table([0,1],[0.,1.])
    assert r["count"].sum()==2 and r.iloc[0]["count"]==r.iloc[-1]["count"]==1
    assert r.iloc[1]["count"]==0
    assert classification_metrics([0,1],[0.,1.])["ece_10"]==0.
