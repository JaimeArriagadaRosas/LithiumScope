"""Read-only 783-row satellite provenance inventory and patch-local alternative QA.

Uses the existing immutable experimental manifest and patches. Fetches STAC
scene metadata once per scene ID with persistent provenance cache (safe to resume).
For inadequate patches only, probes same-day adjacent tiles; stores AUDITS
and proposed candidates but never writes NPY, labels, folds, or active models.

Examples:
 python -m lithiumscope.model_2.methodology.satellite_inventory \
  --output results/scientific_simulations/sentinel_methodology_20261112/inventory \
  --online-metadata --alternatives

No automatic cross-date acquisitions. SCL class 5 is a proxy, not field rock.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sentinel2 import _read_patch
from lithiumscope.model_2.methodology.satellite_quality import (
    ScientificScenePolicy, choose_scene, scene_day, scene_metadata,
)

MANIFEST="data/processed/model_2/vnext_expanded/training_manifest.csv"
EXPECTED_SHA="764d9ab4d9f548e96f39b9a4bc843c7cb903fabdbbd0a982aa78027cc045e623"
BASE_URL="https://earth-search.aws.element84.com/v1"
REQUIRED=(
    "sample_id","latitude","longitude","image_path","sentinel_scene_id",
    "sentinel_cloud_cover","Li_icpms",
)


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def get_json(url:str, timeout_s:int=15) -> dict:
    req=urllib.request.Request(
        url,headers={"User-Agent":"LithiumScope-Scientific-Audit/1.0"})
    with urllib.request.urlopen(req,timeout=timeout_s) as data:
        return json.load(data)


def fetch_metadata(scene_id:str,cache:Path, *,remote:bool)->tuple[dict|None,str]:
    safe=scene_id.replace("/","_").replace("\\","_")
    dest=cache/(safe+".json")
    if dest.is_file():
        existing=json.loads(dest.read_text(encoding="utf-8"))
        if str(existing.get("id"))!=scene_id:
            raise ValueError("Cached STAC ID mismatch")
        return existing,"cached"
    if not remote:return None,"not_requested"
    url=BASE_URL+"/collections/sentinel-2-l2a/items/"+urllib.parse.quote(scene_id)
    last_exc=None
    for trial in range(2):
        try:
            data=get_json(url,timeout_s=18)
            if data.get("id")!=scene_id:
                raise ValueError("Remote STAC item ID mismatch")
            cache.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix(".json.pending")
            tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),
                           encoding="utf-8")
            tmp.replace(dest)
            return data,"fetched"
        except (OSError,ValueError,TimeoutError) as exc:
            last_exc=exc
            if trial==0:time.sleep(.5)
    return None,f"error:{type(last_exc).__name__}:{str(last_exc)[:130]}"


def local_inventory(manifest:pd.DataFrame):
    """Read original NPYs and optional sidecars, never their historical features."""
    if not set(REQUIRED).issubset(manifest):
        raise ValueError("Missing required training manifest columns")
    img_cache={}
    rows=[]
    for i,record in enumerate(manifest.itertuples(index=False)):
        path=Path(str(record.image_path))
        if not path.is_file():
            raise FileNotFoundError(path)
        file_sha=sha(path)
        if file_sha not in img_cache:
            image=np.load(path,allow_pickle=False)
            if image.shape!=(6,32,32):
                raise ValueError(f"Incorrect original patch dimensions: {path}")
            valid=np.isfinite(image)
            common=valid.all(axis=0)
            img_cache[file_sha]={
                "image_sha256":file_sha,
                "sixband_joint_valid_fraction":float(common.mean()),
                "minimum_band_valid_fraction":float(valid.mean(axis=(1,2)).min()),
                "center_joint_valid_fraction":float(common[8:24,8:24].mean()),
                "has_shared_nodata_mask":bool(np.all(valid==valid[0:1])),
            }
        metadata=path.with_suffix(".json")
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        side=json.loads(metadata.read_text(encoding="utf-8"))
        if str(side.get("sample_id"))!=str(record.sample_id):
            raise ValueError(f"Sidecar sample differs for {record.sample_id}")
        if str(side.get("scene_id"))!=str(record.sentinel_scene_id):
            raise ValueError(f"Sidecar scene differs for {record.sample_id}")
        if (abs(float(side["latitude"])-float(record.latitude))>1e-6 or
            abs(float(side["longitude"])-float(record.longitude))>1e-6):
            raise ValueError(f"Sidecar location differs for {record.sample_id}")
        row={
            "manifest_index":i,"sample_id":str(record.sample_id),
            "latitude":float(record.latitude),"longitude":float(record.longitude),
            "scene_id":str(record.sentinel_scene_id),
            "scene_global_cloud_manifest":float(record.sentinel_cloud_cover),
            "source_cohort":"historic" if i<706 else "GEOROC_provisional",
            "scene_datetime_sidecar":str(side.get("datetime","")),
            "sidecar_collection":str(side.get("collection","")),
            "feature_representation":"RAW_DN_LEGACY_CACHE",
            "field_sample_patch_verified":False,
            **img_cache[file_sha],
        }
        row["under_95pct_coverage"]=bool(
            row["sixband_joint_valid_fraction"]<.95)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_scene(scene:dict) -> dict:
    info={
        "scene_id":str(scene["id"]),
        "scene_date":scene_day(scene),
        "cloud_global_percent":float(scene["properties"].get("eo:cloud_cover",math.nan)),
        "scene_nodata_percent":float(scene["properties"].get(
            "s2:nodata_pixel_percentage",math.nan)),
        "processing_baseline":str(scene["properties"].get("s2:processing_baseline","")),
        "scene_collection":str(scene.get("collection","")),
        "bbox_wgs84":json.dumps(scene.get("bbox")),
    }
    try:
        bands=scene_metadata(scene)
        info["complete_radiometric_metadata"]=True
        info["radiometric_scale"]=json.dumps(
            {k:v["scale"] for k,v in bands.items()},sort_keys=True)
        info["radiometric_offset"]=json.dumps(
            {k:v["offset"] for k,v in bands.items()},sort_keys=True)
    except (ValueError,TypeError) as exc:
        info["complete_radiometric_metadata"]=False
        info["metadata_error"]=str(exc)
    return info


def search_same_day(
    lat:float,lon:float,day:str,*,max_scene_cloud:float=35
) -> list[dict]:
    # Target a complete 640m footprint, not just a point.
    latrad=math.radians(lat)
    dl=320/110_540+.0001
    dx=320/(111_320*max(.02,math.cos(latrad)))+.0001
    bbox=[lon-dx,lat-dl,lon+dx,lat+dl]
    params={
        "collections":"sentinel-2-l2a",
        "bbox":",".join(f"{v:.7f}" for v in bbox),
        "datetime":day+"T00:00:00Z/"+day+"T23:59:59Z",
        "limit":"40",
    }
    url=BASE_URL+"/search?"+urllib.parse.urlencode(params)
    result=get_json(url,timeout_s=18)
    items=[
        s for s in result.get("features",[])
        if float(s.get("properties",{}).get("eo:cloud_cover",100))
        <=max_scene_cloud
    ]
    items.sort(key=lambda s:(
        float(s["properties"]["eo:cloud_cover"]),
        str(s.get("id","")),
    ))
    return items[:20]


def report_alternatives(
    manifest:pd.DataFrame,local:pd.DataFrame, *,
    output:Path, policy:ScientificScenePolicy,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    low=local[local.under_95pct_coverage].copy()
    if len(low)!=16:
        raise ValueError(f"Low quality inventory count changed: {len(low)}")
    unique=low.drop_duplicates("image_sha256").sort_values("manifest_index")
    choices=[];audits=[]
    for item in unique.itertuples(index=False):
        day=str(item.scene_id).split("_")[2]
        day=f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        try:
            candidates=search_same_day(
                item.latitude,item.longitude,day,
                max_scene_cloud=policy.max_scene_cloud)
            chosen,info,records=choose_scene(
                candidates,item.latitude,item.longitude,policy=policy)
            for cand in records:
                shallow={k:v for k,v in cand.items() if k!="radiometry"}
                audits.append({"sample_id":item.sample_id,
                    "original_scene_id":item.scene_id,
                    "candidate_scene":cand.get("scene_id"),
                    **{k:json.dumps(v) if isinstance(v,(list,dict)) else v
                       for k,v in shallow.items()},
                })
            # No writing replacement pixels: this is an evidence proposal.
            verdict=("SAME_DATE_PROXY_ACCEPTED" if chosen else
                     "NO_ACCEPTED_SAME_DAY_SCENE")
            choices.append({
                "sample_id":item.sample_id,
                "original_scene_id":item.scene_id,
                "original_joint_coverage":item.sixband_joint_valid_fraction,
                "same_day_candidate_count":len(candidates),
                "chosen_scene_id":chosen.get("id") if chosen else "",
                "chosen_joint_coverage":(
                    info["joint_spectral_coverage"] if info else None),
                "chosen_scl_class5":(
                    info["scl_class5_not_vegetated_fraction"] if info else None),
                "chosen_local_blocked":(
                    info["local_scl_blocked_fraction"] if info else None),
                "decision":verdict,
                "certified_outcrop":False,
                "raw_images_modified":False,
            })
        except (OSError,ValueError,TimeoutError) as exc:
            choices.append({
                "sample_id":item.sample_id,
                "original_scene_id":item.scene_id,
                "original_joint_coverage":item.sixband_joint_valid_fraction,
                "decision":"REMOTE_SOURCE_ERROR",
                "diagnostic_error":str(exc)[:350],
                "certified_outcrop":False,
                "raw_images_modified":False,
            })
        print("SAME_DAY_QA",choices[-1]["sample_id"],
              choices[-1]["decision"],
              choices[-1].get("chosen_scene_id",""),flush=True)
        pd.DataFrame(choices).to_csv(
            output/"same_day_alternative_decisions.csv",index=False)
        if audits:
            pd.DataFrame(audits).to_csv(
                output/"every_scene_candidate_audit.csv",index=False)
    return pd.DataFrame(choices),pd.DataFrame(audits)


from lithiumscope.model_2.methodology.satellite_inventory_run import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--online-metadata",action="store_true")
    p.add_argument("--alternatives",action="store_true")
    p.add_argument("--workers",type=int,default=6)
    args=p.parse_args()
    output=Path(args.output)
    if not output.is_absolute():output=PROJECT_ROOT/output
    run(PROJECT_ROOT,output,online=args.online_metadata,
        alternatives=args.alternatives,workers=args.workers)


if __name__=="__main__":
    main()
