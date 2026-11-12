"""Pure offline guardrails for local STAC QA, SCL, and radiometric features."""
from __future__ import annotations

import json

from pathlib import Path

import numpy as np

import pytest

from lithiumscope.model_2.methodology.satellite_quality import (
    ScientificScenePolicy, scene_metadata, analyze_arrays, choose_scene,
    inspect_scene, scene_day, SPECTRAL_BANDS
)

from lithiumscope.model_2.methodology.radiometric_features import (
    apply_radiometry, extract_valid_spectral_features, build_feature_variants,
)

from lithiumscope.model_2.data.sentinel2 import (
    SentinelConfig, Sentinel2Provider, download_patch, SentinelAssetError,
)

from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features


@pytest.fixture
def scene():
    return {
        "id":"S2B_19KCB_20261030_0_L2A",
        "collection":"sentinel-2-l2a",
        "properties":{"datetime":"2026-10-30T15:00:00Z",
                      "eo:cloud_cover":0.01,
                      "s2:nodata_pixel_percentage":70.6,
                      "s2:processing_baseline":"05.13"},
        "assets":{
            **{band:{"href":f"asset_{band}",
                "raster:bands":[{"scale":.0001,"offset":-.1,"nodata":0}]}
               for band in SPECTRAL_BANDS},
            "scl":{"href":"asset_scl"}
        },
    }


def test_scene_day_and_policy_validation(scene):
    assert scene_day(scene)=="2026-10-30"
    assert scene_day({"id":"S2B_19KCB_20261030_0_L2A"})=="2026-10-30"
    with pytest.raises(ValueError):
        ScientificScenePolicy(min_joint_valid=0)
    with pytest.raises(ValueError):
        ScientificScenePolicy(max_candidates=101)


def test_scientific_selector_requires_frozen_temporal_range():
    class Client:
        def search(self,**kwargs):
            raise AssertionError("No STAC search when time window is not defined")
    provider=Sentinel2Provider(SentinelConfig(
        stac_url="",collection="sentinel-2-l2a",datetime="2020-01-01/..",
        cloud_cover_max=35,search_limit=20,request_timeout_seconds=5,
        max_retries=1,scene_cache_decimals=4,patch_size_m=640,
        patch_pixels=32,bands=SPECTRAL_BANDS),client=Client())
    with pytest.raises(ValueError,match="datetime interval"):
        provider.search_validated_scene(-16,-70)


def test_local_manifest_inventory_is_exact_and_read_only(tmp_path):
    from lithiumscope.model_2.methodology.satellite_inventory import local_inventory,sha
    arr=np.ones((6,32,32),dtype=np.float32)
    arr[:,:4,:]=np.nan
    file=tmp_path/"sample.npy"
    np.save(file,arr,allow_pickle=False)
    meta={"sample_id":"A","scene_id":"S2_X","latitude":-16.,
          "longitude":-70.}
    file.with_suffix(".json").write_text(json.dumps(meta),encoding="utf-8")
    original_sha=sha(file)
    import pandas as pd
    m=pd.DataFrame([{"sample_id":"A","latitude":-16,"longitude":-70,
                     "image_path":str(file),"sentinel_scene_id":"S2_X",
                     "sentinel_cloud_cover":.01,"Li_icpms":25}])
    row=local_inventory(m).iloc[0]
    assert row.image_sha256==original_sha
    assert row.under_95pct_coverage
    assert row.sixband_joint_valid_fraction==.875
    assert file.read_bytes()==(tmp_path/"sample.npy").read_bytes()
    meta["scene_id"]="wrong"
    file.with_suffix(".json").write_text(json.dumps(meta),encoding="utf-8")
    with pytest.raises(ValueError,match="scene differs"):
        local_inventory(m)


def test_site_decision_does_not_confuse_scl_class5_with_field_confirmation():
    from lithiumscope.model_2.methodology.satellite_quality import analyze_arrays
    arr=np.ones((6,32,32))
    scl=np.full((32,32),5.)
    result=analyze_arrays(arr,scl)
    assert result["scientifically_accepted_proxy"] is True
    assert result["field_rock_exposure_certified"] is False


def test_radiometry_handles_pre_2022_zero_offset_and_post_2022_shift():
    raw=np.full((6,32,32),2000.,dtype=np.float32)
    pre={name:{"scale":.0001,"offset":0.0} for name in SPECTRAL_BANDS}
    post={name:{"scale":.0001,"offset":-.1} for name in SPECTRAL_BANDS}
    a=apply_radiometry(raw,pre)
    b=apply_radiometry(raw,post)
    assert a[0,3,3]==pytest.approx(.2,abs=1e-6)
    assert b[0,3,3]==pytest.approx(.1,abs=1e-6)
    assert b[3,3,3]<a[3,3,3]


def test_cached_scientific_features_use_stac_per_scene_and_not_original_model(
    monkeypatch,tmp_path
):
    from lithiumscope.model_2.methodology import radiometric_comparison as c
    import pandas as pd
    monkeypatch.setattr(c,"ROOT_CACHE","cache")
    cache=tmp_path/"cache";cache.mkdir()
    im=tmp_path/"patch.npy"
    arr=np.full((6,32,32),1800.,dtype="float32")
    arr[3]=3800.
    np.save(im,arr,allow_pickle=False)
    scene={
        "id":"SCENE_A",
        "assets":{
            band:{"href":f"test_{band}","raster:bands":[
                  {"scale":.0001,"offset":-.1,"nodata":0}]}
            for band in SPECTRAL_BANDS},
    }
    (cache/"SCENE_A.json").write_text(json.dumps(scene),encoding="utf-8")
    contract=pd.DataFrame({
        "sample_id":[f"s{i}" for i in range(783)],
        "sentinel_scene_id":["SCENE_A"]*783,
        "image_path":[str(im)]*783,
    })
    old=im.read_bytes()
    columns=list(extract_spectral_features(np.ones((6,32,32))))
    frames,meta=c.build_feature_caches(
        tmp_path,tmp_path,contract=contract,frozen_columns=columns)
    assert meta["unique_image_scene_pairs"]==1
    assert meta["scenes_with_stac_radiometry"]==1
    assert all(x.shape==(783,126) for x in frames.values())
    assert frames["reflectance_valid_only"].iloc[0].equals(
        frames["reflectance_valid_only"].iloc[-1])
    assert im.read_bytes()==old


def test_explicit_scientific_candidate_acquisition_does_not_overwrite(
    tmp_path,monkeypatch,scene
):
    import pandas as pd
    from lithiumscope.model_2.methodology import acquire_validated_candidate as q
    from lithiumscope.model_2.methodology.satellite_inventory import sha
    folder=tmp_path/"data";folder.mkdir()
    old=folder/"old.npy"
    np.save(old,np.full((6,32,32),1200.,dtype=np.float32),allow_pickle=False)
    manifest=folder/"manifest.csv"
    pd.DataFrame([{
        "sample_id":"S1","image_path":str(old),
        "sentinel_scene_id":"S2B_19KCB_20261030_0_L2A",
        "latitude":-16.,"longitude":-70.,"Li_icpms":35,
    }]).to_csv(manifest,index=False)
    monkeypatch.setattr(q,"MANIFEST","data/manifest.csv")
    monkeypatch.setattr(q,"EXPECTED_SHA",sha(manifest))
    monkeypatch.setattr(q,"_sentinel_config",lambda cfg:object())
    monkeypatch.setattr(q,"load_config",lambda name:{})
    valid=dict(scene)
    valid["id"]="S2B_19KBB_20261030_0_L2A"
    evidence={"scene_id":valid["id"],"scientifically_accepted_proxy":True}
    class Provider:
        def search_validated_scene(self,lat,lon,*,datetime):
            assert "2026-10-30" in datetime
            return valid,evidence,[evidence]
    calls=[]
    def download(sample_id,latitude,longitude,destination,provider,**kwargs):
        calls.append(str(destination))
        np.save(destination,np.ones((6,32,32),dtype=np.float32))
        destination.with_suffix(".json").write_text(
            json.dumps({"sample_id":sample_id}),encoding="utf-8")
        return destination,{"sample_id":sample_id}
    monkeypatch.setattr(q,"download_patch",download)
    original_sha=sha(old)
    out=tmp_path/"review"
    before=q.proposal_for_sample(tmp_path,"S1",out=out,provider_factory=lambda _:Provider())
    assert before["candidate_saved"] is False
    assert calls==[]
    assert len(list(out.glob("*.npy")))==0
    saved=q.proposal_for_sample(tmp_path,"S1",out=out,
          save_candidate=True,provider_factory=lambda _:Provider())
    assert saved["candidate_saved"]
    assert len(calls)==1
    assert sha(old)==original_sha
    assert len(list(out.glob("*.npy")))==1
    with pytest.raises(FileExistsError,match="overwrite"):
        q.proposal_for_sample(tmp_path,"S1",out=out,
          save_candidate=True,provider_factory=lambda _:Provider())
    with pytest.raises(ValueError,match="Unsafe"):
        q.proposal_for_sample(tmp_path,"../escape",out=out,
          provider_factory=lambda _:Provider())
