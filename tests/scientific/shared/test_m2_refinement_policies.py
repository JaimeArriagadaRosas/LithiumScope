import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_2.refinement_policies import POLICIES, LEGACY, REFERENCE, feature_columns
from lithiumscope.experiments.model_2.refinement_metrics import ranking


def test_shortlist_and_legacy_feature_contract():
    x=pd.DataFrame(np.zeros((2,170)),columns=[f"f{i}" for i in range(170)])
    assert sum(map(len,POLICIES.values()))==26
    assert feature_columns(x,LEGACY)==list(x.columns[:126])
    assert feature_columns(x,REFERENCE)==list(x.columns)
    with pytest.raises(ValueError):
        feature_columns(x.iloc[:,:169],LEGACY)


def test_rankings_accept_margins_without_probability_claim():
    x=pd.DataFrame({"f":[0.,0.,1.,2.]})
    groups=pd.Series(["a","a","b","c"])
    m=ranking([0,1,0,1],[-2.,-2.,-1.,3.],x,groups)
    assert 0<=m["roc_auc"]<=1
    assert 0<=m["signature_auc"]<=1
    assert m["roc_auc"]!=m["signature_auc"]
