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


def test_stac_radiometry_fails_closed(scene):
    data=scene_metadata(scene)
    assert data["red"]["scale"]==.0001
    assert data["red"]["offset"]==-.1
    scene["assets"]["nir"].pop("raster:bands")
    with pytest.raises(ValueError,match="scale/offset"):
        scene_metadata(scene)


def test_corrected_ndvi_uses_offset_not_digital_number(scene):
    raw=np.full((6,32,32),2000.,dtype="float32")
    raw[3]=4000.
    raw[0,0,0]=np.nan
    source=raw.copy()
    meta=scene_metadata(scene)
    reflectance=apply_radiometry(raw,meta)
    assert np.isnan(reflectance[0,0,0])
    assert reflectance[0,4,4]==pytest.approx(.1,abs=1e-6)
    assert reflectance[3,4,4]==pytest.approx(.3,abs=1e-6)
    clean=extract_valid_spectral_features(reflectance)
    original=extract_valid_spectral_features(raw)
    assert clean["ndvi_mean"]==pytest.approx(.5,abs=1e-6)
    assert original["ndvi_mean"]==pytest.approx(1./3,abs=1e-6)
    assert np.array_equal(raw,source,equal_nan=True)
    assert list(clean)==list(extract_spectral_features(np.ones_like(raw)))
    assert len(clean)==126


def test_no_data_and_zero_denominator_never_create_fake_ratios(scene):
    cube=np.full((6,32,32),1000.,dtype="float32")
    cube[:,0,0]=np.nan
    cube[3]=2000.
    x=apply_radiometry(cube,scene_metadata(scene))
    features=extract_valid_spectral_features(x)
    assert features["ndvi_mean"]==pytest.approx(1.0,abs=1e-6)
    assert features["red_blue_ratio_mean"]==0.
    with pytest.raises(ValueError,match="coverage"):
        extract_valid_spectral_features(x,min_finite_fraction=1)
    with pytest.raises(ValueError,match="valid"):
        extract_valid_spectral_features(np.full_like(cube,np.nan))


def test_all_three_isolated_variants_keep_schema(scene):
    cube=np.full((6,32,32),2000.,dtype="float32")
    cube[3]=3200.
    cube[:,4,4]=np.nan
    result=build_feature_variants(cube,scene_metadata(scene))
    assert list(result)==[
        "raw_valid_only","reflectance_valid_only","reflectance_median_filled"]
    for x in result.values():
        assert len(x)==126
        assert np.isfinite(list(x.values())).all()


def test_scl_science_qa_uses_patch_not_global_percentage():
    data=np.ones((6,32,32),dtype=float)
    not_vegetated=np.full((32,32),5.,dtype=float)
    allowed=analyze_arrays(data,not_vegetated)
    assert allowed["scientifically_accepted_proxy"]
    assert allowed["field_rock_exposure_certified"] is False
    assert allowed["joint_spectral_coverage"]==1.
    # Snow may be 100% spectral coverage: reject nevertheless.
    snow=analyze_arrays(data,np.full((32,32),11))
    assert "SNOW_ICE" in snow["rejection_reasons"]
    assert snow["scientifically_accepted_proxy"] is False
    # Dark pixels/water have complete coverage but cannot be called exposed rock.
    dark=analyze_arrays(data,np.full((32,32),2))
    assert "INSUFFICIENT_NONVEGETATED_PROXY" in dark["rejection_reasons"]
    missing=data.copy();missing[:,:,:16]=np.nan
    bad=analyze_arrays(missing,not_vegetated)
    assert bad["joint_spectral_coverage"]==.5
    assert "SPECTRAL_COVERAGE" in bad["rejection_reasons"]


def test_categorical_scl_unknown_classes_fail():
    data=np.ones((6,32,32),dtype=float)
    scl=np.full((32,32),15.)
    with pytest.raises(ValueError,match="Invalid SCL"):
        analyze_arrays(data,scl)


def test_candidate_selection_prefers_valid_local_surface_over_low_global_cloud(scene):
    low=dict(scene)
    low["id"]="low_cloud_bad_spectral"
    low["properties"]=dict(scene["properties"],**{"eo:cloud_cover":.01})
    high=dict(scene)
    high["id"]="higher_cloud_good_patch"
    high["properties"]=dict(scene["properties"],**{"eo:cloud_cover":5.})
    low["assets"]={**scene["assets"],
        **{k:{**v,"href":f"low_{k}"} for k,v in scene["assets"].items()}}
    high["assets"]={**scene["assets"],
        **{k:{**v,"href":f"good_{k}"} for k,v in scene["assets"].items()}}
    def read(href,*a,**kw):
        if href.endswith("scl"):
            return np.full((32,32),5.)
        result=np.ones((32,32),dtype=float)
        if href.startswith("low_"):
            result[:16,:]=np.nan
        return result
    win,score,rows=choose_scene([low,high],-16.,-70.,reader=read)
    assert win["id"]=="higher_cloud_good_patch"
    assert score["joint_spectral_coverage"]==1.
    assert len(rows)==2
    assert rows[0]["scientifically_accepted_proxy"] is False


def test_candidate_selection_rejects_snow_and_preserves_diagnostics(scene):
    def read(href,*args,**kwargs):
        return (np.full((32,32),11.) if href=="asset_scl"
                else np.ones((32,32),dtype=float))
    chosen,detail,audits=choose_scene([scene],-16.,-70.,reader=read)
    assert chosen is None
    assert detail is None
    assert "SNOW_ICE" in audits[0]["rejection_reasons"]


def test_existing_provider_best_scene_still_unchanged_and_local_search_bounded():
    class Search:
        def items(self):
            return iter([
                {"id":"cloudier","properties":{"eo:cloud_cover":9}},
                {"id":"clearer","properties":{"eo:cloud_cover":1}},
            ])
    class Client:
        def __init__(self):self.calls=[]
        def search(self,**kw):
            self.calls.append(kw)
            return Search()
    client=Client()
    cfg=SentinelConfig(
        stac_url="",collection="sentinel-2-l2a",datetime="2020-01-01/..",
        cloud_cover_max=35,search_limit=20,request_timeout_seconds=5,
        max_retries=1,scene_cache_decimals=4,patch_size_m=640,
        patch_pixels=32,bands=SPECTRAL_BANDS)
    p=Sentinel2Provider(cfg,client=client)
    assert p.search_best_scene(-16.,-70.)["id"]=="clearer"
    items=p.search_candidate_scenes(-16.,-70.,datetime="2026-10-30/2026-10-30")
    assert [i["id"] for i in items]==["clearer","cloudier"]
    assert "2026-10-30" in client.calls[-1]["datetime"]
    assert client.calls[-1]["bbox"][2]-client.calls[-1]["bbox"][0]>.002


def test_download_can_bind_validated_scene_without_changing_existing_cache(
    tmp_path,scene,monkeypatch
):
    from lithiumscope.model_2.data import sentinel2 as src
    class Provider:
        config=SentinelConfig(
            stac_url="",collection="",datetime="",cloud_cover_max=35,
            search_limit=10,request_timeout_seconds=5,max_retries=1,
            scene_cache_decimals=4,patch_size_m=640,patch_pixels=32,
            bands=SPECTRAL_BANDS)
        def search_best_scene(self,*a):
            raise AssertionError("Do not search again for validated scene")
    monkeypatch.setattr(src,"_read_patch",lambda *a,**kw:np.full((32,32),2000.))
    location=tmp_path/"new.npy"
    bad={"scene_id":scene["id"],"scientifically_accepted_proxy":False}
    with pytest.raises(SentinelAssetError):
        download_patch("A",-16.,-70.,location,Provider(),
                       selected_scene=scene,scientific_quality=bad)
    assert not location.exists()
    good={"scene_id":scene["id"],"scientifically_accepted_proxy":True,
          "field_rock_exposure_certified":False}
    path,meta=download_patch("A",-16.,-70.,location,Provider(),
                       selected_scene=scene,scientific_quality=good)
    assert path.exists() and meta["scene_id"]==scene["id"]
    assert meta["scientific_patch_qa"]["field_rock_exposure_certified"] is False
    old=path.read_bytes()
    again,_=download_patch("A",-16.,-70.,location,Provider(),
                          selected_scene=scene,scientific_quality=bad)
    assert again.read_bytes()==old
