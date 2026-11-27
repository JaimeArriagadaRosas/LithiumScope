import numpy as np
import pytest

from lithiumscope.experiments.model_1.studies.refinement.policies import Policy, POLICIES, decode, encode


def test_bounded_log_avoids_exponential_overflow_and_records_projection():
    p=Policy("bounded",target="log1p",bounded=True)
    pred,negative,above=decode([-1.,np.log1p(50.),1e6],p,696.)
    assert np.allclose(pred,[0.,50.,696.])
    assert negative.tolist()==[True,False,False]
    assert above.tolist()==[False,False,True]


def test_sqrt_round_trip_and_no_negative_predictions():
    p=Policy("sqrt",target="sqrt")
    pred,negative,above=decode(np.r_[-2.,encode([0.,25.,400.],"sqrt")],p,100.)
    assert np.allclose(pred,[0.,0.,25.,400.])
    assert negative[0] and above[-1]


def test_reference_does_not_hide_negative_predictions():
    p=POLICIES["catboost"][0]
    pred,negative,_=decode([-2.,5.],p,100.)
    assert pred.tolist()==[-2.,5.]
    assert negative.tolist()==[True,False]


def test_invalid_latent_values_fail_even_when_bounded():
    with pytest.raises(ValueError):
        decode([np.inf],Policy("bounded",target="log1p",bounded=True),696.)


def test_shortlist_size_and_uniqueness():
    assert sum(map(len,POLICIES.values()))==26
    for arms in POLICIES.values():
        assert arms[0].name=="reference"
        assert len({p.name for p in arms})==len(arms)
