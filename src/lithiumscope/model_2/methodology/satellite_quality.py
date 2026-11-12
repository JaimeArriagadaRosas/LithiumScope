"""Scientific Sentinel-2 per-patch QA and deterministic scene selection.

Opt-in research layer. It never updates old NPYs, active models, frozen folds,
or assumes that SCL class 5 implies a confirmed rock outcrop.

Official ESA SCL codes 0-11:
https://sentiwiki.copernicus.eu/web/s2-processing
Radiometric STAC metadata from Earth Search must be read per asset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable

import numpy as np

from lithiumscope.model_2.data.sentinel2 import (
    _read_patch, _asset_href, SentinelAssetError,
)


SPECTRAL_BANDS=("red","green","blue","nir","swir16","swir22")
SCL_MEANINGS={0:"no_data",1:"saturated",2:"dark_terrain",
              3:"cloud_shadow",4:"vegetation",5:"not_vegetated",
              6:"water",7:"unclassified",8:"cloud_medium",
              9:"cloud_high",10:"cirrus",11:"snow_ice"}
BLOCKING_CLASSES=frozenset({0,1,3,8,9,10,11})
AMBIGUOUS_CLASSES=frozenset({2,4,6,7})
DEFAULT_MIN_VALID=.95
DEFAULT_MAX_BLOCKED=.25
DEFAULT_MIN_CLASS5=.50


@dataclass(frozen=True)
class ScientificScenePolicy:
    min_joint_valid:float=DEFAULT_MIN_VALID
    max_scl_blocked:float=DEFAULT_MAX_BLOCKED
    min_not_vegetated:float=DEFAULT_MIN_CLASS5
    max_scene_cloud:float=35.0
    max_candidates:int=20

    def __post_init__(self):
        if not (0 < self.min_joint_valid <= 1):
            raise ValueError("min_joint_valid must lie in (0,1]")
        if not (0 <= self.max_scl_blocked <= 1):
            raise ValueError("max_scl_blocked must lie in [0,1]")
        if not (0 <= self.min_not_vegetated <= 1):
            raise ValueError("min_not_vegetated must lie in [0,1]")
        if not (0 <= self.max_scene_cloud <= 100):
            raise ValueError("max_scene_cloud must lie in [0,100]")
        if self.max_candidates<1 or self.max_candidates>100:
            raise ValueError("max_candidates must lie between 1 and 100")


def scene_day(scene:dict)->str:
    dt=scene.get("properties",{}).get("datetime")
    if dt:
        try:return date.fromisoformat(str(dt)[:10]).isoformat()
        except ValueError: pass
    # Earth Search first-party IDs encode acquisition date with _YYYYMMDD_
    parts=str(scene.get("id","")).split("_")
    for token in parts:
        if len(token)==8 and token.isdigit():
            try:return date(int(token[:4]),int(token[4:6]),int(token[6:])).isoformat()
            except ValueError:pass
    return ""


def scene_metadata(scene:dict,bands=SPECTRAL_BANDS)->dict:
    """Fail closed on missing scale/offset: never silently interpret DN as BOA."""
    output={}
    for band in bands:
        asset=scene.get("assets",{}).get(band,{})
        records=asset.get("raster:bands") or []
        if len(records)!=1 or "scale" not in records[0] or "offset" not in records[0]:
            raise ValueError(
                f"Scene {scene.get('id')} {band}: STAC scale/offset missing")
        entry=records[0]
        scale=float(entry["scale"]); offset=float(entry["offset"])
        if not np.isfinite(scale) or scale<=0 or not np.isfinite(offset):
            raise ValueError(f"Invalid STAC radiometric metadata for {band}")
        output[band]={
            "scale":scale,"offset":offset,
            "nodata":entry.get("nodata"),
            "asset_href":asset.get("href",""),
        }
    return output


def analyze_arrays(
    spectra:np.ndarray,scl:np.ndarray, *,
    policy:ScientificScenePolicy|None=None
)->dict:
    """One common 32x32 grid; categorical SCL must be nearest-neighbour resampled."""
    policy=policy or ScientificScenePolicy()
    raw=np.asarray(spectra)
    classification=np.asarray(scl)
    if raw.shape!=(6,32,32) or classification.shape!=(32,32):
        raise ValueError("Expected six spectral bands + 32x32 categorical SCL")
    joint=np.isfinite(raw).all(axis=0)
    scl_finite=np.isfinite(classification)
    valid_scl=(scl_finite & np.isclose(classification,np.rint(classification)))
    if not np.all((classification[valid_scl]>=0)&(classification[valid_scl]<=11)):
        raise ValueError("Invalid SCL classes outside [0,11]")
    values=np.full(classification.shape,-1,dtype=np.int16)
    values[valid_scl]=np.rint(classification[valid_scl]).astype(np.int16)
    freqs={i:int((values==i).sum()) for i in range(12)}
    area=classification.size
    class5=freqs[5]/area
    blocked=sum(freqs[i] for i in BLOCKING_CLASSES)/area
    ambiguous=sum(freqs[i] for i in AMBIGUOUS_CLASSES)/area
    invalid_scl=(~valid_scl).sum()/area
    rejected=[]
    valid=float(joint.mean())
    if valid<policy.min_joint_valid: rejected.append("SPECTRAL_COVERAGE")
    if invalid_scl>1-policy.min_joint_valid: rejected.append("SCL_COVERAGE")
    if blocked>policy.max_scl_blocked: rejected.append("LOCAL_CLOUD_SHADOW_SNOW_OR_DEFECT")
    if class5<policy.min_not_vegetated: rejected.append("INSUFFICIENT_NONVEGETATED_PROXY")
    if (freqs[11]/area)>policy.max_scl_blocked:
        rejected.append("SNOW_ICE")
    return {
        "joint_spectral_coverage":valid,
        "minimum_single_band_coverage":float(np.isfinite(raw).mean(axis=(1,2)).min()),
        "scl_coverage":float(valid_scl.mean()),
        "local_scl_blocked_fraction":float(blocked),
        "local_scl_ambiguous_fraction":float(ambiguous),
        "scl_class5_not_vegetated_fraction":float(class5),
        "scl_class11_snow_fraction":float(freqs[11]/area),
        "scl_counts":{str(i):int(n) for i,n in freqs.items()},
        "rejection_reasons":sorted(set(rejected)),
        "scientifically_accepted_proxy":not rejected,
        "field_rock_exposure_certified":False,
    }


def inspect_scene(
    scene:dict,latitude:float,longitude:float, *,
    patch_size_m:float=640, patch_pixels:int=32,
    bands=SPECTRAL_BANDS,
    reader:Callable=_read_patch,
    policy:ScientificScenePolicy|None=None,
    read_arrays:bool=True,
)->tuple[dict,np.ndarray|None]:
    """Read all six assets and SCL for each candidate, report exact acceptance reason."""
    policy=policy or ScientificScenePolicy()
    score=scene.get("properties",{}).get("eo:cloud_cover")
    try:cloud=float(score)
    except (ValueError,TypeError):cloud=float("inf")
    entry={
        "scene_id":str(scene.get("id","")),"date":scene_day(scene),
        "scene_cloud_percent":cloud if np.isfinite(cloud) else None,
        "scene_nodata_percent":scene.get("properties",{}).get("s2:nodata_pixel_percentage"),
        "satellite_processing_baseline":str(scene.get("properties",{}).get("s2:processing_baseline","")),
        "scene_collection":str(scene.get("collection","")),
    }
    try:
        if not np.isfinite(cloud) or cloud>policy.max_scene_cloud:
            raise ValueError("GLOBAL_CLOUD_LIMIT")
        if "scl" not in scene.get("assets",{}):
            raise ValueError("SCL_ASSET_MISSING")
        # Radiometry must be traceable, even when this step is only assessing QA.
        radiometry=scene_metadata(scene,bands)
        stack=np.stack([
            reader(_asset_href(scene,b),latitude,longitude,patch_size_m,patch_pixels)
            for b in bands
        ])
        scl=reader(_asset_href(scene,"scl"),latitude,longitude,
                   patch_size_m,patch_pixels,categorical=True)
        result=analyze_arrays(stack,scl,policy=policy)
        entry.update(result)
        entry["radiometry"]=radiometry
        return entry,stack if read_arrays else None
    except (SentinelAssetError,ValueError,OSError) as exc:
        entry.update({
            "joint_spectral_coverage":None,
            "local_scl_blocked_fraction":None,
            "scl_class5_not_vegetated_fraction":None,
            "scientifically_accepted_proxy":False,
            "field_rock_exposure_certified":False,
            "rejection_reasons":[str(exc)[:250]],
        })
        return entry,None


def choose_scene(
    candidates:list[dict],latitude:float,longitude:float, *,
    policy:ScientificScenePolicy|None=None,
    patch_size_m:float=640,patch_pixels:int=32,
    reader:Callable=_read_patch,
)->tuple[dict|None,dict|None,list[dict]]:
    """Rank all bounded candidates by measured local quality, not EO cloud.

    A candidate with 100% coverage but dominant snow/dark/water is rejected.
    Selection is deterministic and never mixes pixels from dates or scenes.
    """
    policy=policy or ScientificScenePolicy()
    if len(candidates)>policy.max_candidates:
        candidates=candidates[:policy.max_candidates]
    audits=[]
    candidates_map={}
    for scene in candidates:
        item,arrays=inspect_scene(scene,latitude,longitude,
            patch_size_m=patch_size_m,patch_pixels=patch_pixels,
            reader=reader,policy=policy)
        audits.append(item)
        candidates_map[item["scene_id"]]=(scene,arrays)
    allowed=[item for item in audits if item["scientifically_accepted_proxy"]]
    allowed.sort(key=lambda i:(
        -i["joint_spectral_coverage"],
        i["local_scl_blocked_fraction"],
        -i["scl_class5_not_vegetated_fraction"],
        i["scene_cloud_percent"],
        i["date"],i["scene_id"],
    ))
    if not allowed:
        return None,None,audits
    winner=allowed[0]
    original,_=candidates_map[winner["scene_id"]]
    return original,winner,audits
