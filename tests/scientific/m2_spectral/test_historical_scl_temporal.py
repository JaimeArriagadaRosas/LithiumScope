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


def test_temporal_windows_require_three_nonoverlapping_intervals():
    from lithiumscope.experiments.model_2.temporal_acquisition import validate_windows
    valid=["2024-03-01/2024-03-15","2024-01-01/2024-01-15","2024-02-01/2024-02-15"]
    assert validate_windows(valid)==sorted(valid)
    with pytest.raises(ValueError,match="disjoint"):
        validate_windows([valid[0],valid[1],"2024-01-15/2024-01-31"])


def test_historical_recovery_persists_mask_without_mutating_source(tmp_path):
    import json
    import pandas as pd
    from lithiumscope.core.hashing import file_sha256
    from lithiumscope.model_2.methodology.historical_scl_recovery import recover

    patch=tmp_path/"patch.npy"
    np.save(patch,np.ones((6,32,32)))
    side=patch.with_suffix(".json")
    side.write_text(json.dumps({"sample_id":"0001","scene_id":"scene","latitude":-20.,"longitude":-70.}))
    manifest=tmp_path/"manifest.csv"
    pd.DataFrame([{"sample_id":"0001","image_path":str(patch),"sentinel_scene_id":"scene",
                   "latitude":-20.,"longitude":-70.}]).to_csv(manifest,index=False)
    cache=tmp_path/"metadata"
    cache.mkdir()
    scl_asset=tmp_path/"scl.tif"
    scl_asset.write_bytes(b"synthetic")
    (cache/"scene.json").write_text(json.dumps({"id":"scene","collection":"sentinel-2-l2a",
                                    "assets":{"scl":{"href":str(scl_asset)}}}))
    hashes=[file_sha256(p) for p in (patch,side,manifest)]
    calls=[]
    def reader(*args,**kwargs):
        calls.append(1)
        assert kwargs["categorical"]
        return np.full((32,32),5.)
    out=tmp_path/"results"/"audit"
    first=recover(tmp_path,manifest,hashes[2],out,metadata_cache=cache,reader=reader)
    second=recover(tmp_path,manifest,hashes[2],out,metadata_cache=cache,reader=reader)
    assert first["recovered"]==second["recovered"]==1
    assert first["new_attempts"]==1 and second["new_attempts"]==0 and len(calls)==1
    assert [file_sha256(p) for p in (patch,side,manifest)]==hashes
