import numpy as np
import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.prediction import frozen_rededge as registry


def catalog(root,monkeypatch,patch,*,duplicate=False):
    m=pd.DataFrame({"sample_id":[f"s{i:03}" for i in range(706)],
                    "sentinel_scene_id":["scene-a"]*706,
                    "latitude":np.arange(706)/100-20,"longitude":-70.,
                    "image_path":"unused.npy"})
    q=pd.DataFrame({"sample_id":m.sample_id,"scene_id":m.sentinel_scene_id,
                    "original_patch_sha256":["x"+str(i) for i in range(706)],
                    "rededge_patch_sha256":"red-sha","scene_band_hash":"scene-sha",
                    "rededge_joint_valid_fraction":1.})
    q.loc[0,"original_patch_sha256"]=file_sha256(patch)
    if duplicate:
        q.loc[1,"original_patch_sha256"]=q.loc[0,"original_patch_sha256"]
        m.loc[1,"latitude"]=m.loc[0,"latitude"]
        q.loc[1,"rededge_patch_sha256"]="different"
    source=root/"data/processed/model_2/training_manifest.csv"
    qa=root/registry.V4_DIR/"cohort_706/rededge_quality.csv"
    source.parent.mkdir(parents=True)
    qa.parent.mkdir(parents=True)
    m.to_csv(source,index=False)
    q.to_csv(qa,index=False)
    monkeypatch.setattr(registry,"MANIFEST_SHA",file_sha256(source))
    monkeypatch.setattr(registry,"QUALITY_SHA",file_sha256(qa))
    return source,qa,registry.register_frozen_catalog(root)


def test_lookup_uses_bytes_scene_coordinates_and_not_supplied_sample_id(tmp_path,monkeypatch):
    patch=tmp_path/"external-copy.npy"
    np.save(patch,np.ones((6,32,32),np.float32),allow_pickle=False)
    source,qa,directory=catalog(tmp_path,monkeypatch,patch)
    q=pd.read_csv(qa)
    calls=[]
    def fake(sample,**kwargs):
        calls.append((sample,kwargs))
        return {"f":1.},q.iloc[0].to_dict()
    monkeypatch.setattr(registry,"cached_rededge_sample",fake)
    metadata={"scene_id":"scene-a","latitude":-20.,"longitude":-70.,"sample_id":"malicious-id"}
    features,evidence=registry.find_frozen_rededge(patch,metadata,root=tmp_path)
    assert features=={"f":1.} and evidence["rededge_cache_source"]=="verified_frozen_v4_706"
    assert calls[0][0].sample_id=="s000"
    assert calls[0][1]["allow_acquisition"] is False
    source.write_text("different future training cohort",encoding="utf-8")
    assert registry.find_frozen_rededge(patch,metadata,root=tmp_path)[0]==features
    with pytest.raises(ValueError,match="declared scene"):
        registry.find_frozen_rededge(patch,{**metadata,"longitude":-71.},root=tmp_path)
    novel=tmp_path/"novel.npy"
    np.save(novel,np.zeros((6,32,32)),allow_pickle=False)
    assert registry.find_frozen_rededge(novel,metadata,root=tmp_path) is None
    (directory/"rededge_quality.csv").write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="hash mismatch"):
        registry.find_frozen_rededge(patch,metadata,root=tmp_path)


def test_ambiguous_same_bytes_scene_coordinate_is_rejected(tmp_path,monkeypatch):
    patch=tmp_path/"patch.npy"
    np.save(patch,np.ones((6,32,32)),allow_pickle=False)
    catalog(tmp_path,monkeypatch,patch,duplicate=True)
    def never(*a,**kw):
        raise AssertionError("Do not score ambiguous patches")
    monkeypatch.setattr(registry,"cached_rededge_sample",never)
    with pytest.raises(ValueError,match="Ambiguous"):
        registry.find_frozen_rededge(patch,{"scene_id":"scene-a","latitude":-20.,"longitude":-70.},root=tmp_path)


def test_missing_catalog_returns_none_and_partial_catalog_fails(tmp_path):
    path=tmp_path/"never.npy"
    metadata={}
    assert registry.find_frozen_rededge(path,metadata,root=tmp_path) is None
    directory=tmp_path/registry.CATALOG
    directory.mkdir(parents=True)
    (directory/"training_manifest.csv").write_text("partial",encoding="utf-8")
    with pytest.raises(ValueError,match="incomplete"):
        registry.find_frozen_rededge(path,metadata,root=tmp_path)


def test_cache_provenance_drift_is_blocked(tmp_path,monkeypatch):
    patch=tmp_path/"patch.npy"
    np.save(patch,np.ones((6,32,32)),allow_pickle=False)
    _,qa,_=catalog(tmp_path,monkeypatch,patch)
    evidence=pd.read_csv(qa).iloc[0].to_dict()
    evidence["rededge_patch_sha256"]="wrong"
    monkeypatch.setattr(registry,"cached_rededge_sample",lambda *a,**kw:({"f":1.},evidence))
    with pytest.raises(ValueError,match="evidence drift"):
        registry.find_frozen_rededge(patch,{"scene_id":"scene-a","latitude":-20.,"longitude":-70.},root=tmp_path)
