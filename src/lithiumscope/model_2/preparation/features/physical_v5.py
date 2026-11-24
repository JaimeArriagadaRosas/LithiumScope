"""Experimental 164-D ten-band reflectance extractor; never substitute for v3/v4."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES
from lithiumscope.model_2.preparation.features.rededge import NEW_BANDS

VERSION = 5
BANDS = DEFAULT_BAND_NAMES + NEW_BANDS
BASE_FEATURES = 120
TOTAL_FEATURES = 164
# SCL no-data, defective, shadow, clouds and snow/ice; unknown codes are not valid.
SCL_BLOCKED = (0, 1, 3, 8, 9, 10, 11)
INDEX_NAMES = ("ndvi", "ndmi", "nbr", "nbr2", "mndwi", "bare_soil_index",
               "dry_bare_soil_index", "red_swir16_nd", "swir16_swir22_ratio",
               "nir_swir16_ratio", "red_blue_ratio", "green_red_ratio",
               "ndre1_refl", "ndre2_refl")
STAT_NAMES = ("mean", "std", "p10", "p50", "p90", "iqr")


@dataclass(frozen=True)
class PhysicalPolicy:
    minimum_joint_valid: float = .95
    minimum_descriptor_support: float = .95
    denominator_epsilon: float = 1e-6
    require_scl: bool = False

    def __post_init__(self):
        if not 0 < self.minimum_joint_valid <= 1 or not 0 < self.minimum_descriptor_support <= 1:
            raise ValueError("Coverage thresholds must be in (0,1]")
        if not np.isfinite(self.denominator_epsilon) or self.denominator_epsilon <= 0:
            raise ValueError("Invalid reflectance denominator tolerance")


def physical_reflectance(dn, metadata, *, band_names=BANDS):
    """Apply declared STAC radiometry and nodata before conversion. No clipping."""
    cube = np.asarray(dn, dtype=np.float32)
    if tuple(band_names) != BANDS or cube.shape != (10, 32, 32):
        raise ValueError("V5 requires ordered ten-band 32x32 Sentinel-2 patch")
    if set(metadata) != set(BANDS):
        raise ValueError("Every band requires exact radiometric metadata")
    result = np.full_like(cube, np.nan)
    for i, name in enumerate(BANDS):
        item = metadata[name]
        scale, offset = float(item["scale"]), float(item["offset"])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise ValueError(f"Invalid radiometry for {name}")
        if "nodata" not in item:
            raise ValueError(f"Unknown nodata convention for {name}")
        valid = np.isfinite(cube[i])
        nodata = item["nodata"]
        if nodata is not None:
            valid &= cube[i] != float(nodata)
        result[i, valid] = cube[i, valid].astype(np.float64) * scale + offset
    return result


def _stats(prefix, values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {f"{prefix}_{name}": float("nan") for name in STAT_NAMES}
    p10, p25, p50, p75, p90 = np.percentile(finite, [10,25,50,75,90])
    return dict(zip((f"{prefix}_{name}" for name in STAT_NAMES),
                    map(float, (finite.mean(), finite.std(), p10,p50,p90,p75-p25)), strict=True))


def _texture(prefix, values):
    # NaN at either endpoint invalidates the pair. Never bridge holes or fill them.
    edges = np.concatenate([np.abs(np.diff(values,axis=axis)).ravel() for axis in (0,1)])
    valid = edges[np.isfinite(edges)]
    support = len(valid)/len(edges)
    return {f"{prefix}_texture_mean": float(valid.mean()) if len(valid) else float("nan"),
            f"{prefix}_texture_p90": float(np.percentile(valid,90)) if len(valid) else float("nan")}, support


def _divide(a, b, epsilon, *, difference=False):
    numerator, denominator = (a-b,a+b) if difference else (a,b)
    valid = np.isfinite(a)&np.isfinite(b)&np.isfinite(denominator)&(np.abs(denominator)>epsilon)
    result = np.full_like(a,np.nan,dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _indices(b, epsilon):
    def nd(a,c):
        return _divide(a,c,epsilon,difference=True)
    def ratio(a,c):
        return _divide(a,c,epsilon)
    ndvi=nd(b["nir"],b["red"])
    return {
        "ndvi":ndvi,
        "ndmi":nd(b["nir"],b["swir16"]),
        "nbr":nd(b["nir"],b["swir22"]),
        "nbr2":nd(b["swir16"],b["swir22"]),
        "mndwi":nd(b["green"],b["swir16"]),
        "bare_soil_index":nd(b["swir16"]+b["red"],b["nir"]+b["blue"]),
        "dry_bare_soil_index":nd(b["swir16"],b["green"])-ndvi,
        "red_swir16_nd":nd(b["red"],b["swir16"]),
        "swir16_swir22_ratio":ratio(b["swir16"],b["swir22"]),
        "nir_swir16_ratio":ratio(b["nir"],b["swir16"]),
        "red_blue_ratio":ratio(b["red"],b["blue"]),
        "green_red_ratio":ratio(b["green"],b["red"]),
        "ndre1_refl":nd(b["nir08"],b["rededge1"]),
        "ndre2_refl":nd(b["rededge3"],b["rededge2"]),
    }


def extract_physical_v5(reflectance, *, scl=None, policy=PhysicalPolicy()):
    """Return descriptors and support. Nonfinite unsupported descriptors remain NaN."""
    cube=np.asarray(reflectance,dtype=np.float32)
    if cube.shape!=(10,32,32):
        raise ValueError("V5 requires ten ordered 32x32 bands")
    finite=np.isfinite(cube)
    joint=finite.all(axis=0)
    usable=joint.copy()
    if scl is not None:
        scl=np.asarray(scl,dtype=float)
        if scl.shape!=(32,32):
            raise ValueError("SCL must have same 32x32 grid")
        present=np.isfinite(scl)
        if not np.equal(scl[present],np.floor(scl[present])).all() or not ((scl[present]>=0)&(scl[present]<=11)).all():
            raise ValueError("SCL must contain categorical codes 0..11 or NaN")
        usable &= present & ~np.isin(scl,SCL_BLOCKED)
    arrays={name:np.where(usable,cube[i],np.nan) for i,name in enumerate(BANDS)}
    descriptors={}
    support={}
    # Preserve block order: six bands, twelve indices, four red-edge bands, two indices.
    indices=_indices(arrays,policy.denominator_epsilon)
    def add_band(name):
        prefix=name if name in DEFAULT_BAND_NAMES else f"{name}_refl"
        descriptors.update(_stats(prefix,arrays[name]))
        texture, fraction=_texture(prefix,arrays[name])
        descriptors.update(texture)
        support[f"{prefix}_texture"]=fraction
    def add_index(name):
        value=indices[name]
        descriptors.update(_stats(name,value))
        support[name]=float(np.isfinite(value).mean())
    for name in DEFAULT_BAND_NAMES:
        add_band(name)
    for name in INDEX_NAMES[:12]:
        add_index(name)
    for name in NEW_BANDS:
        add_band(name)
    for name in INDEX_NAMES[12:]:
        add_index(name)
    if len(descriptors)!=TOTAL_FEATURES or len(set(descriptors))!=TOTAL_FEATURES:
        raise RuntimeError("V5 descriptor schema changed")
    reasons=[]
    if policy.require_scl and scl is None:
        reasons.append("SCL_UNAVAILABLE")
    if float(usable.mean())<policy.minimum_joint_valid:
        reasons.append("INSUFFICIENT_JOINT_PIXELS")
    if any(v<policy.minimum_descriptor_support for v in support.values()):
        reasons.append("INSUFFICIENT_DESCRIPTOR_SUPPORT")
    if not np.isfinite(list(descriptors.values())).all():
        reasons.append("UNDEFINED_DESCRIPTOR")
    quality={
        "joint_finite_fraction":float(joint.mean()),
        "usable_joint_fraction":float(usable.mean()),
        "band_finite_fraction":dict(zip(BANDS, map(float, finite.mean(axis=(1,2))),strict=True)),
        "descriptor_support":support,
        "imputed_pixels":0,
        "scl_applied":scl is not None,
        "scl_status":"applied" if scl is not None else "not_available",
        "eligible_for_experimental_fit":not reasons,
        "reasons":reasons,
        "field_rock_exposure_certified":False,
    }
    return descriptors,quality
