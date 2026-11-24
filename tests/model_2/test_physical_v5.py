import numpy as np
import pytest

from lithiumscope.model_2.preparation.features.physical_v5 import (
    BANDS, PhysicalPolicy, extract_physical_v5, physical_reflectance,
)


def test_physical_conversion_respects_nodata_and_true_zero():
    cube=np.full((10,32,32),1000,dtype=np.float32)
    cube[0,0,0]=0
    cube[1,0,0]=500
    metadata={b:{"scale":.0001,"offset":-.1,"nodata":0} for b in BANDS}
    physical=physical_reflectance(cube,metadata)
    assert np.isnan(physical[0,0,0])
    assert physical[1,0,0]==pytest.approx(-.05)
    assert physical[0,1,1]==0
    with pytest.raises(ValueError,match="ordered"):
        physical_reflectance(cube,metadata,band_names=BANDS[::-1])
    with pytest.raises(ValueError,match="nodata"):
        physical_reflectance(cube,{b:{"scale":.1,"offset":0} for b in BANDS})


def test_corrected_indices_and_order():
    values=dict(zip(BANDS,[.2,.3,.1,.6,.4,.5,.25,.35,.45,.65],strict=True))
    cube=np.broadcast_to(np.array(list(values.values()),dtype=np.float32)[:,None,None],(10,32,32))
    f,q=extract_physical_v5(cube)
    expected={
        "ndvi":(.6-.2)/(.6+.2),"ndmi":(.6-.4)/(.6+.4),
        "nbr":(.6-.5)/(.6+.5),"nbr2":(.4-.5)/(.4+.5),
        "mndwi":(.3-.4)/(.3+.4),
        "bare_soil_index":((.4+.2)-(.6+.1))/((.4+.2)+(.6+.1)),
        "dry_bare_soil_index":(.4-.3)/(.4+.3)-(.6-.2)/(.6+.2),
        "red_swir16_nd":(.2-.4)/(.2+.4),
        "swir16_swir22_ratio":.4/.5,"nir_swir16_ratio":.6/.4,
        "red_blue_ratio":.2/.1,"green_red_ratio":.3/.2,
        "ndre1_refl":(.65-.25)/(.65+.25),"ndre2_refl":(.45-.35)/(.45+.35)}
    assert len(f)==164
    assert not any(k.startswith("nir_swir22_nd_") for k in f)
    for name,value in expected.items():
        assert f[f"{name}_mean"]==pytest.approx(value,abs=1e-6)
        assert q["descriptor_support"][name]==1.
    assert f["red_texture_mean"]==0
    assert f["red_mean"]==pytest.approx(.2)
    assert q["eligible_for_experimental_fit"]
    assert not q["scl_applied"]


def test_empty_band_and_undefined_indices_are_not_zero_measurements():
    cube=np.ones((10,32,32),dtype=np.float32)
    cube[0]=np.nan
    f,q=extract_physical_v5(cube)
    assert all(np.isnan(v) for v in f.values())
    assert "UNDEFINED_DESCRIPTOR" in q["reasons"]
    assert q["band_finite_fraction"]["red"]==0
    cube[:]=0
    f,q=extract_physical_v5(cube)
    assert f["red_mean"]==0
    assert np.isnan(f["ndvi_mean"])
    assert q["descriptor_support"]["ndvi"]==0
    assert not q["eligible_for_experimental_fit"]


def test_texture_uses_only_adjacent_valid_pairs():
    cube=np.ones((10,32,32),dtype=np.float32)
    cube[0]=np.broadcast_to(np.arange(32,dtype=np.float32),(32,32))
    cube[1,:,15]=np.nan
    f,q=extract_physical_v5(cube)
    # 928 horizontal valid edges with gradient 1, 961 vertical with gradient 0.
    assert f["red_texture_mean"]==pytest.approx(928/1889)
    assert f["red_texture_p90"]==1
    assert q["descriptor_support"]["red_texture"]==pytest.approx(1889/1984)
    assert q["imputed_pixels"]==0


def test_optional_scl_masks_blocked_and_unknown_pixels():
    cube=np.ones((10,32,32),dtype=np.float32)
    cube[:,0,0]=10000
    scl=np.full((32,32),5,dtype=float)
    scl[0,0]=9
    scl[0,1]=np.nan
    f,q=extract_physical_v5(cube,scl=scl)
    assert f["red_mean"]==1
    assert q["usable_joint_fraction"]==1022/1024
    assert q["scl_applied"] and q["eligible_for_experimental_fit"]
    assert not q["field_rock_exposure_certified"]
    _,q=extract_physical_v5(cube,policy=PhysicalPolicy(require_scl=True))
    assert "SCL_UNAVAILABLE" in q["reasons"]
    scl[0,0]=5.5
    with pytest.raises(ValueError,match="categorical"):
        extract_physical_v5(cube,scl=scl)


def test_exact_shape_and_policy():
    with pytest.raises(ValueError,match="32x32"):
        extract_physical_v5(np.ones((10,2,2)))
    with pytest.raises(ValueError,match="threshold"):
        PhysicalPolicy(minimum_joint_valid=0)
    with pytest.raises(ValueError,match="tolerance"):
        PhysicalPolicy(denominator_epsilon=np.nan)
