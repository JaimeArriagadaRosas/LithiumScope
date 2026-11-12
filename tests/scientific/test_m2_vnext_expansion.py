"""M2 vNext expansion: data leakage, source screening, quality, isolation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.data import expansion_vnext as ex
from lithiumscope.model_2.data.sentinel2 import SentinelAssetError


def _legacy(n=12):
    return pd.DataFrame({
        "sample_id": [f"old_{i}" for i in range(n)],
        "Li_icpms": [8, 10, 11, 14, 16, 18, 21, 22, 23, 27, 33, 40][:n],
        "latitude": [-24.0]*n, "longitude": [-70.0]*n,
        "spatial_group": ["g_132_220"]*n,
        "image_path": ["a.npy"]*n,
    })


def _source():
    return pd.DataFrame([
        {
            "sample_id": f"GEOROC_{i}", "CITATIONS": f"[{i % 6}]",
            "Latitude": -25.0-i*0.5, "Longitude": -70.0,
            "Li_icpms": float(10+i*3), "LOCATION": f"/X/CHILE/R{i}",
            "screening_tier": "PROVISIONAL_SITE_REVIEW"
            if i != 3 else "EXCLUDE_SUBSURFACE_LABEL",
        } for i in range(12)
    ])


def test_strict_source_filter_excludes_subsurface_and_existing_groups():
    baseline = _legacy()
    source = _source()
    eligible=ex.candidates(source,baseline,max_li_ppm=30, degrees=.5)
    assert not eligible.sample_id.eq("GEOROC_3").any()
    assert eligible.Li_icpms.max()<=30
    assert not set(eligible.spatial_group).intersection(baseline.spatial_group)
    assert eligible.high_lithium.any()
    assert not eligible.high_lithium.all()


def test_selection_keeps_geographic_uniqueness_and_paper_limit():
    frame=ex.candidates(_source(),_legacy(),max_li_ppm=100, degrees=.5)
    queue=ex.balanced_queue(frame,target=8,max_per_paper=2)
    assert queue.spatial_group.is_unique
    assert queue.CITATIONS.value_counts().max()<=2
    assert len(queue)>0
    assert ex.balanced_queue(frame,target=8,max_per_paper=2).sample_id.tolist()==queue.sample_id.tolist()


def test_m2_frozen_integrity_protects_old_manifest(monkeypatch,tmp_path):
    legacy=_legacy()
    p=tmp_path/"legacy.csv"
    (tmp_path/"a.npy").write_bytes(b"stub")
    legacy["image_path"] = str(tmp_path/"a.npy")
    legacy.to_csv(p,index=False)
    monkeypatch.setattr(ex,"file_sha256",lambda path:"correct")
    config={"training":{
        "manifest_path":str(p),
        "frozen_expected_rows":len(legacy),
        "frozen_manifest_sha256":"correct",
    }}
    assert len(ex.frozen_m2(tmp_path,config))==len(legacy)
    config["training"]["frozen_manifest_sha256"]="incorrect"
    with pytest.raises(RuntimeError,match="changed"):
        ex.frozen_m2(tmp_path,config)


def test_scl_cloud_filter_uses_local_pixels_not_scene_summary(monkeypatch):
    from lithiumscope.model_2.data.sentinel2 import SentinelConfig
    cfg=SentinelConfig(
        stac_url="",collection="",datetime="",cloud_cover_max=30,
        search_limit=3,request_timeout_seconds=2,max_retries=1,
        scene_cache_decimals=4,patch_size_m=640,patch_pixels=32,
        bands=("red","green","blue","nir","swir16","swir22"),
    )
    class Stub:
        config=cfg
    scene={"assets":{"scl":{"href":"example_scl"}}}
    monkeypatch.setattr(ex,"_read_patch",lambda *args,**kw:
                        np.full((32,32),9.,dtype=float))
    with pytest.raises(SentinelAssetError,match="Cloud"):
        ex._cloud_mask(scene,-22.,-69.,Stub(),max_cloud=.25)
    monkeypatch.setattr(ex,"_read_patch",lambda *args,**kw:
                        np.full((32,32),4.,dtype=float))
    assert ex._cloud_mask(scene,-22.,-69.,Stub(),max_cloud=.25)[
        "patch_cloud_fraction_scl"]==0


def test_spectral_contract_126_features_for_valid_6_band_patch(tmp_path: Path):
    path=tmp_path/"s2.npy"
    rng=np.random.default_rng(42)
    np.save(path,rng.uniform(.01,.55,(6,32,32)).astype("float32"))
    assert ex._validate_bands(path,{"imagery":{"bands":[
        "red","green","blue","nir","swir16","swir22",
    ]}})==126
    np.save(path,rng.random((5,32,32)).astype("float32"))
    with pytest.raises(ValueError,match="shape"):
        ex._validate_bands(path,{"imagery":{"bands":[
            "red","green","blue","nir","swir16","swir22",
        ]}})


def test_expansion_plan_is_download_free_and_leaves_legacy_unchanged(monkeypatch,tmp_path):
    # CI has no untracked historical Li imagery; reproduce the frozen
    # 706-row integrity contract without relying on local private datasets.
    from copy import deepcopy

    config=deepcopy(ex.load_config("model_2"))
    assert config["training"]["expansion"]["enabled"] is True
    image=tmp_path/"reference_patch.npy"
    np.save(image,np.ones((6,32,32),dtype="float32"))
    baseline=pd.DataFrame({
        "sample_id":[f"historical_{i}" for i in range(706)],
        "Li_icpms":np.resize(_legacy().Li_icpms.to_numpy(float),706),
        "latitude":[-24.0]*706,
        "longitude":[-70.0]*706,
        "spatial_group":["historical_group"]*706,
        "image_path":[str(image)]*706,
    })
    oldfile=tmp_path/"historical_706.csv"
    baseline.to_csv(oldfile,index=False)
    audit=tmp_path/"source_audit.csv"
    _source().to_csv(audit,index=False)
    config["training"].update({
        "manifest_path":str(oldfile),
        "frozen_expected_rows":706,
        "frozen_manifest_sha256":ex.file_sha256(oldfile),
    })
    config["training"]["expansion"].update({
        "source_audit_path":str(audit),
        "state_dir":str(tmp_path/"selection_state"),
        "patch_dir":str(tmp_path/"new_patches"),
        "target_new_samples":2,
        "minimum_new_samples":1,
        "max_li_ppm":100,
    })
    monkeypatch.setattr(ex,"load_config",lambda name:config)
    monkeypatch.setattr(ex,"Sentinel2Provider",
                        lambda cfg: pytest.fail("Plan must not use satellite provider"))
    base=ex.frozen_m2(tmp_path,config)
    original=oldfile.read_bytes()
    assert ex.ensure_expanded_m2_manifest(
        root=tmp_path,acquire=False,max_new_downloads=1
    ) is None
    assert oldfile.read_bytes()==original
    assert len(base)==706
    assert not (tmp_path/"new_patches").exists()



def test_anchored_scientific_folds_preserve_all_legacy_test_rows():
    from lithiumscope.model_2.training.scientific_validation import (
        make_anchored_scientific_folds,
    )
    lithium=pd.Series(([10.,30.]*50)+([9.,32.]*5))
    groups=pd.Series(
        [f"original_{i}" for i in range(100)]
        + [f"new_{i}" for i in range(10)]
    )
    ref=[]
    for fold in range(5):
        te=[i for i in range(100) if i%5==fold]
        tr=[i for i in range(100) if i%5!=fold]
        ref.append({"train_indices":tr,"test_indices":te})
    result=make_anchored_scientific_folds(
        lithium,groups,ref,baseline_count=100,
    )
    assert len(result)==5
    for idx,fold in enumerate(result):
        old_test=fold.test_idx[fold.test_idx<100]
        assert np.array_equal(old_test,np.array(ref[idx]["test_indices"]))
        assert fold.label_threshold_ppm==30
        assert not set(groups.iloc[fold.train_idx]).intersection(
            set(groups.iloc[fold.test_idx])
        )
    assert sum(len(f.test_idx) for f in result)==110
    with pytest.raises(ValueError,match="overlaps"):
        collision=groups.copy()
        collision.iloc[100]=groups.iloc[0]
        make_anchored_scientific_folds(
            lithium,collision,ref,baseline_count=100,
        )


def test_isolated_end_to_end_manifest_preparation_with_mocked_download(
    monkeypatch,tmp_path,
):
    from copy import deepcopy
    from lithiumscope.model_2.training import scientific_validation
    base=_legacy()
    images=tmp_path/"existing"
    images.mkdir()
    for i in range(len(base)):
        path=images/f"old_{i}.npy"
        np.save(path,np.ones((6,32,32),dtype="float32"))
        base.loc[i,"image_path"]=str(path)
        base.loc[i,"spatial_group"]=f"oldgroup_{i}"
    oldfile=tmp_path/"old.csv"
    base.to_csv(oldfile,index=False)
    import hashlib
    sha=hashlib.sha256(oldfile.read_bytes()).hexdigest()
    audit=pd.DataFrame([
      {"sample_id":f"GEOROC_{i}","CITATIONS":f"[{i}]",
       "Latitude":-20.-i, "Longitude":-69., "Li_icpms":float(10+i*20),
       "LOCATION":f"/A/CHILE/{i}","screening_tier":"PROVISIONAL_SITE_REVIEW"}
      for i in range(2)
    ])
    source=tmp_path/"source.csv"
    audit.to_csv(source,index=False)
    cfg=deepcopy(ex.load_config("model_2"))
    cfg["training"].update({
      "manifest_path":str(oldfile),
      "frozen_expected_rows":len(base),
      "frozen_manifest_sha256":sha,
    })
    cfg["training"]["expansion"].update({
      "source_audit_path":str(source),
      "state_dir":str(tmp_path/"state"),
      "patch_dir":str(tmp_path/"patches"),
      "output_manifest_path":str(tmp_path/"expanded.csv"),
      "target_new_samples":2,"minimum_new_samples":2,
      "max_per_publication":1,
      "max_consecutive_failures":3,
    })
    monkeypatch.setattr(ex,"load_config",lambda _:cfg)
    class Provider:
        def __init__(self,config):
            self.config=config
        def validate(self):
            pass
        def search_best_scene(self,lat,lon):
            return {"id":f"scene_{lat}"}
    def download(sample_id,latitude,longitude,destination,provider):
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.save(destination,np.ones((6,32,32),dtype="float32"))
        return destination,{"scene_id":"test_scene","cloud_cover":1}
    monkeypatch.setattr(ex,"Sentinel2Provider",Provider)
    monkeypatch.setattr(ex,"download_patch",download)
    monkeypatch.setattr(ex,"_cloud_mask",lambda *a,**k:{
        "patch_cloud_fraction_scl":0.0
    })
    monkeypatch.setattr(scientific_validation,"make_scientific_folds",
                        lambda *a,**k:[1,2,3,4,5])
    original=oldfile.read_bytes()
    result=ex.ensure_expanded_m2_manifest(
        tmp_path,acquire=True
    )
    assert result.is_file()
    frame=pd.read_csv(result)
    assert len(frame)==len(base)+2
    assert frame.iloc[:len(base)].sample_id.tolist()==base.sample_id.tolist()
    assert set(frame.iloc[-2:].source_dataset)=={
        "GEOROC_B_SURFACE_PROVISIONAL"
    }
    assert oldfile.read_bytes()==original
    assert frame.spatial_group.nunique()==len(base)+2
    assert ex.ensure_expanded_m2_manifest(tmp_path,acquire=False) is None
