"""Tests for source identity, spatial subset isolation and fixed test targets."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.analysis.verify_vnext_data_adequacy import (
    sample_keys, group_train_indices,
)


def test_sample_keys_identify_same_rock_and_location():
    data=pd.DataFrame({
        "source_dataset":["GEOROC"]*3,
        "source_sample":["r-1","r-1","r-2"],
        "Longitude":[-70.123456,-70.123456,-71.12],
        "Latitude":[-23.121001,-23.121001,-24.1],
    })
    keys=sample_keys(data)
    assert keys.iloc[0]==keys.iloc[1]
    assert keys.iloc[0]!=keys.iloc[2]


def test_group_subsampling_never_includes_external_test_groups():
    groups=pd.Series(["A","A","B","B","C","C","D","D","E","E"])
    full_train=np.asarray([0,1,2,3,4,5,6,7],dtype=int)
    heldout=np.asarray([8,9],dtype=int)
    for fraction in (.35,.6,.8,1.0):
        chosen=group_train_indices(full_train,groups,fraction,42)
        assert len(chosen)>0
        assert not set(chosen).intersection(set(heldout))
        assert not set(groups.iloc[chosen]).intersection(set(groups.iloc[heldout]))


def test_subset_determinism_and_invalid_fraction():
    groups=pd.Series(["A","B","C","D","E","F"])
    ids=np.arange(6)
    assert np.array_equal(
        group_train_indices(ids,groups,.5,123),
        group_train_indices(ids,groups,.5,123),
    )
    with pytest.raises(ValueError):
        group_train_indices(ids,groups,0,42)
    with pytest.raises(ValueError):
        group_train_indices(ids,groups,1.01,42)
