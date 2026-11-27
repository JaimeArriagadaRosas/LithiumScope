"""Academic benchmark is separate from full-domain spatial release evidence."""
import numpy as np
import pytest

from lithiumscope.experiments.model_1.studies.expansion.academic_bridge import (
    benchmark_status, range_pool, run, PROFILES, ARMS)


def test_both_published_mean_metrics_must_strictly_improve():
    assert benchmark_status(5.360,.500)["both_better"]
    assert not benchmark_status(5.361,.500)["both_better"]
    assert not benchmark_status(5.360,.499)["both_better"]
    assert not benchmark_status(7.7237,.5614)["both_better"]
    assert not benchmark_status(np.nan,.9)["both_better"]


def test_frozen_academic_domain_filters_only_candidate_training_pool():
    result=range_pool([True,False,True,True,True],[5.,8.,126.,127.,1.],
                      {"q025":4.2439,"q975":126.12})
    assert result.tolist()==[True,False,True,False,False]
    with pytest.raises(ValueError):
        range_pool([True],[np.nan],{"q025":4.2439,"q975":126.12})


def test_fixed_campaign_budget_and_explicit_opt_in():
    assert set(PROFILES)=={"random_forest","catboost","hist_gradient_boosting"}
    assert len(PROFILES)*len(ARMS)*5==30
    with pytest.raises(ValueError,match="allow-unverified-metadata"):
        run("missing")
