import json

import numpy as np
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.preparation.patch_contract import (
    make_patch_contract, load_verified_patch,
)
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES, extract_spectral_features
from lithiumscope.model_2.prediction.spectral_contract import prepare_spectral_input, SpectralAbstention


def sidecar(path, *, latitude=-20., longitude=-70., **overrides):
    meta={**make_patch_contract(),"scene_id":"local_test_scene",
          "latitude":latitude,"longitude":longitude,"image_sha256":file_sha256(path)}
    meta.update(overrides)
    path.with_suffix(path.suffix+".json").write_text(json.dumps(meta),encoding="utf-8")


def bundle():
    return {"patch_input_contract":make_patch_contract(),
            "features":list(extract_spectral_features(np.ones((6,32,32)))),
            "quality_min_joint_valid":.95,"feature_extractor_version":3}


def test_missing_metadata_and_wrong_shape_cannot_score(tmp_path):
    path=tmp_path/"small.npy"
    np.save(path,np.ones((6,2,2),dtype=np.float32))
    with pytest.raises(SpectralAbstention,match="sidecar"):
        prepare_spectral_input(path,bundle())
    sidecar(path)
    with pytest.raises(SpectralAbstention,match="32x32"):
        prepare_spectral_input(path,bundle())
    # A legacy bundle retains its historical contract.
    old=bundle()
    old.pop("patch_input_contract")
    frame,detail=prepare_spectral_input(path,old)
    assert frame.shape==(1,126)
    assert detail["input_contract_status"]=="legacy_unverified_geometry"


def test_permuted_declared_bands_or_tampered_bytes_fail(tmp_path):
    path=tmp_path/"patch.npy"
    np.save(path,np.ones((6,32,32),dtype=np.float32))
    sidecar(path,bands=list(DEFAULT_BAND_NAMES[::-1]))
    with pytest.raises(ValueError,match="bands"):
        load_verified_patch(path,make_patch_contract())
    sidecar(path)
    np.save(path,np.zeros((6,32,32),dtype=np.float32))
    with pytest.raises(ValueError,match="SHA256"):
        load_verified_patch(path,make_patch_contract())


def test_corrupt_npy_abstains_even_with_correct_hash(tmp_path):
    path=tmp_path/"broken.npy"
    path.write_bytes(b"not a numpy array")
    sidecar(path)
    with pytest.raises(SpectralAbstention,match="Invalid patch input"):
        prepare_spectral_input(path,bundle())


def test_geotiff_and_npy_equivalence_with_mask(tmp_path):
    rasterio=pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    pixels=np.broadcast_to(np.arange(1,7,dtype=np.float32)[:,None,None],(6,32,32)).copy()
    pixels[:,0,0]=np.nan
    x,y=500000.,7800000.
    lon,lat=transform("EPSG:32719","EPSG:4326",[x],[y])
    npy=tmp_path/"patch.npy"
    tif=tmp_path/"patch.tif"
    np.save(npy,pixels)
    with rasterio.open(tif,"w",driver="GTiff",width=32,height=32,count=6,
                       dtype="float32",crs="EPSG:32719",transform=from_origin(x-320,y+320,20,20),
                       nodata=-9999) as dst:
        dst.write(np.nan_to_num(pixels,nan=-9999))
        for i,name in enumerate(DEFAULT_BAND_NAMES,1):
            dst.set_band_description(i,name)
    sidecar(npy,latitude=lat[0],longitude=lon[0])
    sidecar(tif,latitude=lat[0],longitude=lon[0])
    a,_,qa=load_verified_patch(npy,make_patch_contract())
    b,_,qb=load_verified_patch(tif,make_patch_contract())
    np.testing.assert_array_equal(a,b)
    assert qa["joint_valid_fraction"]==qb["joint_valid_fraction"]==1023/1024
    assert qb["geometry_verification"]=="metric_crs_resolution_and_center_verified"
    fa,_=prepare_spectral_input(npy,bundle())
    fb,_=prepare_spectral_input(tif,bundle())
    np.testing.assert_array_equal(fa,fb)
    with rasterio.open(tif,"r+") as dst:
        dst.set_band_description(1,"blue")
    sidecar(tif,latitude=lat[0],longitude=lon[0])
    with pytest.raises(ValueError,match="band descriptions"):
        load_verified_patch(tif,make_patch_contract())


@pytest.mark.parametrize("crs,resolution,error",[
    ("EPSG:4326",.0002,"projected metric"),
    ("EPSG:32719",10.,"resolution"),
])
def test_raster_geometry_mismatch(tmp_path,crs,resolution,error):
    rasterio=pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    tif=tmp_path/"bad.tif"
    with rasterio.open(tif,"w",driver="GTiff",width=32,height=32,count=6,
                       dtype="float32",crs=crs,transform=from_origin(500000,7800000,resolution,resolution)) as dst:
        dst.write(np.ones((6,32,32),dtype=np.float32))
        for i,name in enumerate(DEFAULT_BAND_NAMES,1):
            dst.set_band_description(i,name)
    sidecar(tif)
    with pytest.raises(ValueError,match=error):
        load_verified_patch(tif,make_patch_contract())
