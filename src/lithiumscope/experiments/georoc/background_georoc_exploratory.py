"""Resumable unattended exploratory Sentinel-2 acquisition for GEOROC candidates.

Does not certify sample independence or produce external accuracy metrics.
"""
from __future__ import annotations
from pathlib import Path
import json,os,time,traceback
from datetime import datetime,timezone
import pandas as pd
import numpy as np
from lithiumscope.experiments.acquisition.acquisition import acquire_pilot
from lithiumscope.experiments.scenarios.imagery import evaluate_model_2_image
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.persistence.load_model import load_latest_model

ROOT=Path("results/scientific_simulations/external_geochemistry_audit")
OUT=ROOT/"background_georoc_exploratory"
OUT.mkdir(parents=True,exist_ok=True)
STOP=OUT/"STOP"
STATUS=OUT/"status.json"

def snapshot(**kwargs):
    data={"updated_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),**kwargs}
    STATUS.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print("STATUS",data,flush=True)

def make_plan():
    src=pd.read_csv(ROOT/"full_mamani_contamination_audit.csv",low_memory=False)
    sel=src.loc[src.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE") &
      src.nearest_full_mamani_km.ge(100) & src.latitude.notna() & src.longitude.notna()].copy()
    sel["coordinate_key"]=sel.latitude.round(4).astype(str)+"_"+sel.longitude.round(4).astype(str)
    # Use exact reproducible stable coordinate IDs and avoid repeated STAC requests.
    coords=sel.drop_duplicates("coordinate_key").copy()
    coords=coords.sort_values(["latitude","longitude"])
    coords["image_id"]=["external_coord_"+str(i).zfill(4) for i in range(len(coords))]
    sel=sel.merge(coords[["coordinate_key","image_id"]],on="coordinate_key",validate="many_to_one")
    sel[["image_id","sample_name","li_ppm","citation_number","latitude","longitude",
         "nearest_full_mamani_km","coordinate_key"]].to_csv(OUT/"sample_image_map.csv",index=False)
    coords[["image_id","latitude","longitude","coordinate_key"]].to_csv(OUT/"coordinate_plan.csv",index=False)
    return coords

def main():
    coords=make_plan()
    total=len(coords)
    model,_=load_latest_model("model_2")
    existing=OUT/"patch_manifest.csv"
    records={z["image_id"]:z for z in pd.read_csv(existing).to_dict("records")} if existing.exists() else {}
    errors_file=OUT/"errors.csv"
    errors=pd.read_csv(errors_file).to_dict("records") if errors_file.exists() else []
    snapshot(state="RUNNING",coordinates_total=total,done=0,accepted=0,failed=0,scored=0)
    for i,row in enumerate(coords.itertuples(),1):
        if STOP.exists():
            snapshot(state="STOPPED_BY_USER",coordinates_total=total,done=i-1,accepted=sum(r.get("status")=="ACCEPT" for r in records.values()),failed=len(errors),scored=0)
            return
        image_id=row.image_id
        output=OUT/"patches"/f"{image_id}_2025.npy"
        result=OUT/"scores"/image_id/"image_scenario_metrics.csv"
        old=records.get(image_id)
        if old is not None and old.get("status")=="ACCEPT" and result.exists():
            continue
        try:
            if old is None or not output.exists() or old.get("status")!="ACCEPT":
                # Each run is restartable, with bounded worker timeouts and independent per-coordinate output.
                m,_=acquire_pilot(OUT/"patches",years=(2025,),
                    locations={image_id:(float(row.latitude),float(row.longitude))},
                    max_patches=1,timeout_seconds=40)
                raw=pd.read_csv(m)
                target=raw.loc[raw.image_id.eq(image_id+"_2025")]
                if target.empty:
                    records[image_id]={"image_id":image_id,"status":"NO_ACCEPTED_SCENE"}
                else:
                    records[image_id]={**target.iloc[0].to_dict(),"image_id":image_id}
            current=records[image_id]
            if current["status"]=="ACCEPT" and not result.exists():
                from pathlib import Path
                path=Path(str(current["path"]))
                evaluate_model_2_image(model,load_imagery(path),OUT/"scores"/image_id)
        except Exception as e:
            records[image_id]={"image_id":image_id,"status":"ERROR","reason":str(e)[:300]}
            errors.append({"image_id":image_id,"error":str(e)[:300]})
        finally:
            pd.DataFrame(records.values()).to_csv(existing,index=False)
            pd.DataFrame(errors).to_csv(errors_file,index=False)
            count=len(list((OUT/"scores").glob("*/image_scenario_metrics.csv"))) if (OUT/"scores").exists() else 0
            snapshot(state="RUNNING",coordinates_total=total,done=len(records),
                accepted=sum(v.get("status")=="ACCEPT" for v in records.values()),
                failed=sum(v.get("status") in ("ERROR","NO_ACCEPTED_SCENE") for v in records.values()),
                scored=count)
    snapshot(state="COMPLETE",coordinates_total=total,done=len(records),
        accepted=sum(v.get("status")=="ACCEPT" for v in records.values()),
        failed=sum(v.get("status") in ("ERROR","NO_ACCEPTED_SCENE") for v in records.values()),
        scored=len(list((OUT/"scores").glob("*/image_scenario_metrics.csv"))))
if __name__=="__main__":
    try: main()
    except BaseException as err:
        snapshot(state="FATAL",error=str(err),traceback=traceback.format_exc()[-2000:])
        raise
