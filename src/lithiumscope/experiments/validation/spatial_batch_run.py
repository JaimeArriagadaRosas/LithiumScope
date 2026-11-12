"""Resume 36 same-scene spatial patches with one request session per band/region."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pystac_client import Client
from lithiumscope.experiments.validation.spatial_control import plan
from lithiumscope.experiments.acquisition.acquisition import quality_from_scl
from lithiumscope.model_2.data.sentinel2 import _asset_href
from lithiumscope.core.config import load_config

ROOT=Path("results/scientific_simulations")
DEST=ROOT/"controlled_spatial"
CENTERS=pd.read_csv(ROOT/"sentinel2_real"/"manifest.csv")
BANDS=load_config("model_2")["imagery"]["bands"]


def worker(href, positions, mode, tmp, timeout=130):
    points=tmp/"positions.json"
    points.write_text(json.dumps(positions),encoding="utf8")
    output=tmp/(mode+".npy")
    cmd=[sys.executable,"-m","lithiumscope.experiments.validation.spatial_batch_worker",
        "--href",href,"--positions",str(points),"--output",str(output),"--mode",("scl" if mode == "scl" else "reflectance")]
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout,check=False)
    if result.returncode: raise RuntimeError(result.stderr[-500:])
    return np.load(output,allow_pickle=False).copy()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plan_table=plan(CENTERS)
    manifest=DEST/"spatial_manifest.csv"
    existing={r["image_id"]:r for r in pd.read_csv(manifest).to_dict("records")} if manifest.exists() else {}
    for row in plan_table.to_dict("records"):
        if row["direction"]=="center":existing[row["image_id"]]={**row,"status":"ACCEPT"}
    with tempfile.TemporaryDirectory(prefix="spatial_batch_") as temporary:
        temp=Path(temporary)
        client=Client.open(load_config("model_2")["imagery"]["stac_url"])
        collection=client.get_collection(load_config("model_2")["imagery"]["collection"])
        for region, table in plan_table.groupby("region",sort=True):
            pending=[]
            for row in table.to_dict("records"):
                if row["direction"]=="center":continue
                path=DEST/(row["image_id"]+".npy")
                if path.exists() and existing.get(row["image_id"],{}).get("status")=="ACCEPT":
                    continue
                pending.append(row)
            if not pending:
                print("CACHED",region,flush=True)
                continue
            scene_id=table.iloc[0].scene_id
            print("REGION",region,"SCENE",scene_id,"PENDING",len(pending),flush=True)
            item=collection.get_item(scene_id)
            if item is None:raise RuntimeError("Scene unavailable: "+scene_id)
            scene=item.to_dict()
            try:
                scl=worker(_asset_href(scene,"scl"),pending,"scl",temp)
                valid={i:quality_from_scl(s)["quality_status"]=="ACCEPT" for i,s in enumerate(scl)}
                print("SCL",region,sum(valid.values()),"/",len(valid),flush=True)
                cubes=[]
                for band in BANDS:
                    data=worker(_asset_href(scene,band),pending,band,temp)
                    cubes.append(data)
                    print("BAND",region,band,flush=True)
                stack=np.stack(cubes,axis=1)
                for i,row in enumerate(pending):
                    path=DEST/(row["image_id"]+".npy")
                    status="ACCEPT" if valid[i] and np.isfinite(stack[i]).mean()>=.7 else "REJECT_QUALITY"
                    if status=="ACCEPT":np.save(path,stack[i].astype("float32"))
                    existing[row["image_id"]]={**row,"status":status,"path":str(path.resolve()) if status=="ACCEPT" else "",
                      "local_cloud_fraction":quality_from_scl(scl[i])["local_cloud_fraction"]}
                print("COMPLETE",region,flush=True)
            except Exception as exc:
                print("ERROR",region,repr(exc)[:220],flush=True)
                for row in pending:
                    existing[row["image_id"]]={**row,"status":"FAILED","path":"","error":str(exc)[:200]}
            finally:
                pd.DataFrame(existing.values()).to_csv(manifest,index=False)
    print("FINAL",pd.read_csv(manifest).status.value_counts().to_dict(),flush=True)


if __name__=="__main__":main()
