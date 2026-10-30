"""Controlled spatial robustness: only compare patches from exactly the same scene."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from lithiumscope.experiments.acquisition import read_asset_isolated, quality_from_scl
from lithiumscope.model_2.data.sentinel2 import _asset_href
from lithiumscope.experiments.imagery import evaluate_model_2_image

BASE = Path("results/scientific_simulations")
DIRECTIONS = {"north": (1,0), "east": (0,1), "south": (-1,0), "west": (0,-1)}


def offset(lat: float, lon: float, distance_m: float, direction: str):
    dy, dx = DIRECTIONS[direction]
    return lat + dy * distance_m / 111_132, lon + dx * distance_m / (111_320 * math.cos(math.radians(lat)))


def plan(centers: pd.DataFrame, distances=(400,1000)) -> pd.DataFrame:
    records=[]
    for row in centers.to_dict("records"):
        if row.get("year") != 2025 or row.get("status") != "ACCEPT":
            continue
        region=str(row["image_id"]).rsplit("_",1)[0]
        records.append({**row,"region":region,"direction":"center","distance_m":0,"comparison_id":region})
        for distance in distances:
            for direction in DIRECTIONS:
                lat,lon=offset(row["latitude"],row["longitude"],distance,direction)
                records.append({**row,"region":region,"direction":direction,"distance_m":distance,
                    "comparison_id":region,"image_id":f"{region}_{distance}m_{direction}_2025",
                    "latitude":lat,"longitude":lon,"path":""})
    return pd.DataFrame(records)


def verify_existing_pairs(centers: pd.DataFrame, shifted: pd.DataFrame, destination: Path) -> Path:
    rows=[]
    for row in centers.to_dict("records"):
        if row.get("year") != 2025: continue
        region=row["image_id"].rsplit("_",1)[0]
        matches=shifted.loc[shifted.image_id.eq(f"{region}_east_2025")]
        if matches.empty: continue
        other=matches.iloc[0]
        rows.append({"region":region,"center_scene":row["scene_id"],
            "shifted_scene":other["scene_id"],"same_scene":row["scene_id"]==other["scene_id"],
            "center_path":row["path"],"shifted_path":other["path"],
            "center_lat":row["latitude"],"center_lon":row["longitude"],
            "shifted_lat":other["latitude"],"shifted_lon":other["longitude"]})
    destination.mkdir(parents=True,exist_ok=True)
    output=destination/"verified_pairs.csv"
    pd.DataFrame(rows).to_csv(output,index=False)
    return output


def acquire_same_scene(centers: pd.DataFrame, output: Path, distances=(400,1000),
                       max_new: int | None = None) -> Path:
    """Scene is retrieved by exact STAC item id, never a new best-scene search."""
    from pystac_client import Client
    from lithiumscope.core.config import load_config
    cfg=load_config("model_2")["imagery"]
    collection=Client.open(cfg["stac_url"]).get_collection(cfg["collection"])
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/"spatial_manifest.csv"
    planned=plan(centers,distances)
    if manifest.exists():
        old=pd.read_csv(manifest)
        by_id={r["image_id"]:r for r in old.to_dict("records")}
    else: by_id={}
    count=0
    cache={}
    for row in planned.to_dict("records"):
        name=row["image_id"]
        if row["direction"]=="center":
            by_id[name]={**row,"status":"ACCEPT"}
            continue
        target=output/f"{name}.npy"
        if name in by_id and by_id[name].get("status")=="ACCEPT" and target.exists():
            continue
        if max_new is not None and count >= max_new: break
        count+=1
        try:
            scene_id=row["scene_id"]
            if scene_id not in cache:
                item=collection.get_item(scene_id)
                if item is None: raise RuntimeError(f"Missing original STAC scene {scene_id}")
                cache[scene_id]=item.to_dict()
            scene=cache[scene_id]
            lat,lon=row["latitude"],row["longitude"]
            scl=read_asset_isolated(_asset_href(scene,"scl"),lat,lon,640,32,mode="scl")
            quality=quality_from_scl(scl)
            if quality["quality_status"]!="ACCEPT":
                raise ValueError("Local SCL quality rejected")
            bands=[]
            for band in cfg["bands"]:
                bands.append(read_asset_isolated(_asset_href(scene,band),lat,lon,640,32))
            stack=np.stack(bands).astype(np.float32)
            if np.isfinite(stack).mean()<.7: raise ValueError("Too much nodata")
            np.save(target,stack)
            by_id[name]={**row,"path":str(target.resolve()),"status":"ACCEPT",
                          "local_cloud_fraction":quality["local_cloud_fraction"]}
            print("ACCEPT",name,flush=True)
        except Exception as exc:
            by_id[name]={**row,"path":"","status":"FAILED",
                          "error":f"{type(exc).__name__}: {str(exc)[:200]}"}
            print("FAILED",name,str(exc)[:120],flush=True)
        finally:
            pd.DataFrame(by_id.values()).to_csv(manifest,index=False)
    return manifest


def score_and_report(bundle, manifest: Path, destination: Path) -> Path:
    destination.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(manifest)
    results=[]
    for row in table.to_dict("records"):
        if row.get("status")!="ACCEPT": continue
        image=np.load(row["path"])
        out=evaluate_model_2_image(bundle,image,destination/"per_image"/row["image_id"])
        metric=pd.read_csv(out)
        baseline=metric.loc[metric.scenario.eq("baseline")]
        if baseline.empty or not np.isfinite(baseline["score"].iloc[0]):continue
        results.append({**row,"score":float(baseline["score"].iloc[0]),
                        "ood_fraction":float(baseline["ood_fraction"].iloc[0])})
    result=pd.DataFrame(results)
    if result.empty: raise RuntimeError("No scored images")
    centers=result.loc[result.distance_m.eq(0),["comparison_id","score"]].rename(columns={"score":"center_score"})
    result=result.merge(centers,on="comparison_id",how="left")
    result["absolute_score_shift"]=(result["score"]-result["center_score"]).abs()
    result["score_delta"]=result["score"]-result["center_score"]
    path=destination/"spatial_robustness.csv"
    result.to_csv(path,index=False)
    report={"scored_patches":len(result),"regions":int(result.comparison_id.nunique()),
            "same_scene_enforced":True,"distances_m":sorted(result.distance_m.unique().tolist()),
            "by_distance":result.loc[result.distance_m.gt(0)].groupby("distance_m")["absolute_score_shift"].agg(["count","mean","max"]).reset_index().to_dict("records"),
            "note":"Unlabelled model score stability, not Li prediction accuracy. Overlapping 640 m patches are dependent."}
    (destination/"spatial_robustness.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return path
