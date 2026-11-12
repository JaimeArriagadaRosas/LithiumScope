import numpy as np
import pytest

from lithiumscope.experiments.analysis.run_vnext_m2_threshold import (
    threshold_by_training_prevalence,
)


def test_threshold_uses_train_prevalence_and_only_train_scores():
    p=np.array([.05,.12,.15,.3,.6,.8])
    labels=np.array([0,0,0,0,1,1])
    cut=threshold_by_training_prevalence(p,labels)
    assert .3<=cut<=.6
    assert cut==threshold_by_training_prevalence(p,labels)


def test_invalid_threshold_inputs():
    with pytest.raises(ValueError):
        threshold_by_training_prevalence([.1,.2],[1,1])
    with pytest.raises(ValueError):
        threshold_by_training_prevalence([.1],[1,0])
