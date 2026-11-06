import pandas as pd
import pytest
from lithiumscope.experiments.analysis.validate_feot_expansion_m1 import (
    heldout_new_geography,
)


def test_new_groups_do_not_overlap_baseline():
    a=pd.DataFrame({"Longitude":[-70,-69,-68,-67,-66],
                    "Latitude":[-23]*5})
    b=pd.DataFrame({"Longitude":[-65,-64,-63,-62,-61,-60,-70],
                    "Latitude":[-23]*7})
    data,groups=heldout_new_geography(a,b)
    assert len(data)==6
    assert groups.nunique()==6


def test_no_extra_groups_prevents_invalid_evaluation():
    a=pd.DataFrame({"Longitude":[-70,-69,-68,-67,-66],
                    "Latitude":[-23]*5})
    with pytest.raises(ValueError):
        heldout_new_geography(a,a)
