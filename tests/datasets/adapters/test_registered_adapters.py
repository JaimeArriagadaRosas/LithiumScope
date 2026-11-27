import json
import numpy as np
import pytest
from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.adapters.registry import get_adapter
from lithiumscope.datasets.management.registry import DATASETS
from lithiumscope.datasets.adapters.spectral.wavelengths import response_weighted_reflectance

def test_all_registered_sources_have_adapter():
    assert all(callable(get_adapter(key)) for key in DATASETS)

def test_fregeneda_documented_instrument_units(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("header\n"*28 + "400 50\n500 60\n", encoding="utf-8")
    result = get_adapter("fregeneda_almendra")(path, instrument="SR-6500", sample_id="a")
    assert result.frame.reflectance_fraction.tolist() == [.5,.6]
    assert result.evidence["training_label_certified"] is False
    path.write_text("400 .5\n500 .6\n", encoding="utf-8")
    assert get_adapter("fregeneda_almendra")(path,instrument="ASD FieldSpec 4",sample_id="a").frame.reflectance_fraction.tolist() == [.5,.6]

def test_greenpeg_preserves_representation_and_flags_physical_review(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("400 1.1\n500 .5\n", encoding="utf-8")
    result = get_adapter("greenpeg")(path, sample_id="a", representation="raw", reflectance_unit="fraction", header_lines=0)
    assert result.frame.physical_review.tolist() == [True,False]
    path.write_text("400 .5\n400 .6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        get_adapter("greenpeg")(path, sample_id="a", representation="raw", reflectance_unit="fraction", header_lines=0)

def test_band_response_refuses_extrapolation_and_gaps():
    assert response_weighted_reflectance([400,500,600],[.5]*3,[450,550],[1,1],maximum_gap_nm=100) == pytest.approx(.5)
    with pytest.raises(ValueError, match="cover"):
        response_weighted_reflectance([400,500],[.5]*2,[350,450],[1,1],maximum_gap_nm=100)
    with pytest.raises(ValueError, match="gap"):
        response_weighted_reflectance([400,600],[.5]*2,[450,550],[1,1],maximum_gap_nm=100)

def test_mamani_mirror_censor_and_original_provenance(tmp_path):
    path = tmp_path / "mirror.csv"
    path.write_text("Li_icpms,Logintude (X),Latitude (Y)\n<10,-70,-30\n20,-71,-31\n")
    result = get_adapter("mamani09_public_mirror")(path)
    assert np.isnan(result.frame.Li_icpms.iloc[0])
    assert result.frame.source_li_raw.iloc[0] == "<10"
    assert result.frame.Longitude.tolist() == [-70,-71]
    assert result.evidence["official_equivalence_certified"] is False

def test_sentinel_patch_hash_and_band_order(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path,np.ones((2,2,2)))
    meta = {"image_sha256":file_sha256(path),"collection":"sentinel-2-l2a","scene_id":"s",
            "bands":["red","nir"],"patch_pixels":2,"patch_size_m":40}
    path.with_suffix(".json").write_text(json.dumps(meta))
    inspect = get_adapter("sentinel2")
    _, evidence = inspect(path, expected_bands=("red","nir"),expected_pixels=2,expected_size_m=40)
    assert evidence["joint_valid_fraction"] == 1
    with pytest.raises(ValueError, match="contract"):
        inspect(path,expected_bands=("nir","red"),expected_pixels=2,expected_size_m=40)
