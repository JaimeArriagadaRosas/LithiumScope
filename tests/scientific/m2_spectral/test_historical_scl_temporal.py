from types import SimpleNamespace

import numpy as np
import pytest

from lithiumscope.model_2.methodology.historical_scl_recovery import identity
from lithiumscope.experiments.model_2.temporal_comparison import compare_cubes


def test_scl_identity_requires_scene_coordinates_and_metadata():
    a=SimpleNamespace(sentinel_scene_id="scene",latitude=-20.,longitude=-70.)
    b=SimpleNamespace(sentinel_scene_id="other",latitude=-20.,longitude=-70.)
    c=SimpleNamespace(sentinel_scene_id="scene",latitude=-21.,longitude=-70.)
    assert len({identity(a,"image","meta"),identity(b,"image","meta"),
                identity(c,"image","meta"),identity(a,"image","changed")})==4


def test_temporal_median_uses_only_joint_valid_dates_and_reference_order():
    one=np.full((6,32,32),1.)
    two=np.full_like(one,2.)
    three=np.full_like(one,3.)
    two[0,0,0]=np.nan
    three[:,0,0]=np.nan
    ref,composite=compare_cubes([three,one,two],["2024-03-01","2024-01-01","2024-02-01"])
    assert np.array_equal(ref,one)
    assert np.isnan(composite[:,0,0]).all()
    assert np.all(composite[:,1,1]==2)
    with pytest.raises(ValueError,match="distinct"):
        compare_cubes([one,two,three],["2024-01-01"]*3)
