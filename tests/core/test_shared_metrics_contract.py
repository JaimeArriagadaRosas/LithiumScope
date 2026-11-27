from importlib.util import find_spec

import numpy as np


def test_shared_metrics_have_no_model_facades():
    assert find_spec("lithiumscope.model_1.evaluation." + "metrics") is None
    assert find_spec("lithiumscope.model_2.evaluation." + "metrics") is None


def test_metrics_preserve_strict_input_contract():
    from lithiumscope.core.scientific.regression_metrics import regression_metrics
    from lithiumscope.core.scientific.classification_metrics import classification_metrics
    assert regression_metrics([1,2],[1,2]).rmse==0
    assert classification_metrics([0,1],[.1,.9])["roc_auc"]==1
    assert np.isnan(regression_metrics([1,1],[1,1]).r2)
