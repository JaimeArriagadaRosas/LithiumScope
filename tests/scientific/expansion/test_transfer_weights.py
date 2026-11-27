"""Protect attenuation of real high-domain rows and rare bootstrap copies."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_1.treatments.transfer_weights import weighted_positions
from lithiumscope.model_1.training.preprocessing.treatment_policy import POLICIES, rare_group_positions


@pytest.mark.parametrize("weight", [.1,.25,.5,1.])
def test_high_domain_weight_applies_to_all_parent_copies(weight):
    idx=pd.Index(["h"+str(i) for i in range(20)])
    y=pd.Series(np.arange(20,dtype=float),index=idx)
    groups=pd.Series(["g"+str(i//2) for i in range(20)],index=idx)
    high=y.ge(18)
    policy=POLICIES["rare_group_sqrt"]
    pos,w=weighted_positions(y,groups,high,policy,42,weight)
    extra=rare_group_positions(y,groups,42)
    assert np.array_equal(pos,np.r_[np.arange(20),extra])
    assert high.iloc[extra].all()
    assert np.all(w[high.iloc[pos].to_numpy()]==weight)
    assert np.all(w[~high.iloc[pos].to_numpy()]==1.)


def test_weights_reject_misaligned_or_unregistered_policy():
    y=pd.Series([1.,2.],index=["a","b"])
    g=pd.Series(["g1","g2"],index=y.index)
    mask=pd.Series([False,True],index=y.index)
    policy=POLICIES["nonnegative"]
    with pytest.raises(ValueError,match="identities"):
        weighted_positions(y,g,mask.iloc[::-1],policy,42,.25)
    with pytest.raises(ValueError,match="Unregistered"):
        weighted_positions(y,g,mask,policy,42,.75)
    pos,w=weighted_positions(y,g,mask,policy,42,.25)
    assert pos.tolist()==[0,1]
    assert w.tolist()==[1.,.25]
