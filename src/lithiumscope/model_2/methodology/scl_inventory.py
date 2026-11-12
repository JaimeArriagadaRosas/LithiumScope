"""Resumable SCL-local audit for every frozen M2 training patch.

Original spectral NPYs remain read-only. One categorical SCL STAC read per
distinct image + acquisition scene + coordinates, shared with duplicate labels.
A per-patch QA verdict is a satellite input proxy, not a field label or
an instruction to remove samples. JSONL checkpoint written after each site;
rows merged into CSV by original manifest index.

Run with --max-new 0 to cover all 593 distinct images; resume after failure.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sentinel2 import _read_patch
from lithiumscope.model_2.methodology.satellite_quality import (
    ScientificScenePolicy, analyze_arrays
)
from lithiumscope.model_2.methodology.radiometric_comparison import ROOT_CACHE
from lithiumscope.model_2.methodology.satellite_inventory import (
    local_inventory, sha, EXPECTED_SHA, MANIFEST,
)

def run(root:Path,out:Path, *,max_new:int=0)->dict:
    if max_new<0:raise ValueError("max_new must be nonnegative")
    manifest_path=root/MANIFEST
    if sha(manifest_path)!=EXPECTED_SHA:
        raise ValueError("Frozen 783-row manifest changed")
    src=pd.read_csv(manifest_path)
    inventory=root/"results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv"
    if not inventory.is_file():
        raise FileNotFoundError("Run the local satellite inventory first")
    local=pd.read_csv(inventory)
    if len(src)!=783 or len(local)!=783 or (
        src.sample_id.astype(str).tolist()!=local.sample_id.astype(str).tolist()
    ):
        raise ValueError("Local inventory was reordered or edited")
    cache=root/ROOT_CACHE
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"scl_pixel_qa_checkpoint.jsonl"
    done={}
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            record=json.loads(line)
            done[record["image_sha256"]]=record
    unique=local.drop_duplicates("image_sha256").reset_index(drop=True)
    count=0
    scenes={}
    for row in unique.itertuples(index=False):
        key=str(row.image_sha256)
        if key in done:continue
        if max_new and count>=max_new:
            break
        sample_id=str(row.sample_id)
        meta=src.iloc[int(row.manifest_index)]
        scene_id=str(row.scene_id)
        try:
            if scene_id not in scenes:
                path=cache/(scene_id+".json")
                item=json.loads(path.read_text(encoding="utf-8"))
                if item.get("id")!=scene_id:
                    raise ValueError("Scene metadata cache ID mismatch")
                scenes[scene_id]=item
            item=scenes[scene_id]
            asset=item["assets"].get("scl")
            if not asset or not asset.get("href"):
                raise ValueError("SCL asset missing from scene")
            pixels=np.load(Path(meta.image_path),allow_pickle=False)
            if pixels.shape!=(6,32,32):
                raise ValueError("Unexpected model patch dimensions")
            if sha(Path(meta.image_path))!=key:
                raise ValueError("Original NPY content was altered")
            scl=_read_patch(
                asset["href"],float(row.latitude),float(row.longitude),
                640,32,categorical=True)
            scored=analyze_arrays(
                pixels,scl,policy=ScientificScenePolicy())
            scored.pop("scl_counts",None)
            record={
                "image_sha256":key,
                "source_sample":sample_id,
                "scene_id":scene_id,
                "status":"SCENE_SCL_READ",
                **scored,
                "rejection_reasons":";".join(scored["rejection_reasons"]),
            }
        except Exception as exc:
            record={
                "image_sha256":key,
                "source_sample":sample_id,
                "scene_id":scene_id,
                "status":"SCL_UNAVAILABLE",
                "rejection_reasons":str(exc)[:300],
                "scientifically_accepted_proxy":False,
                "field_rock_exposure_certified":False,
            }
        with checkpoint.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(record,ensure_ascii=False)+"\n")
        done[key]=record
        count+=1
        if count%30==0:
            print("SCL_LOCAL_QA",len(done),"/",len(unique),flush=True)
    result=pd.DataFrame(list(done.values()))
    result.to_csv(out/"unique_scl_patch_quality.csv",index=False)
    per_sample=local.merge(result,on="image_sha256",how="left",validate="many_to_one",
                           suffixes=("","_scl"))
    per_sample.to_csv(out/"per_sample_scl_quality_783.csv",index=False)
    if per_sample.sample_id.nunique()!=783:
        raise ValueError("Sample identity was lost in local SCL audit")
    unknown=int(result.status.ne("SCENE_SCL_READ").sum()) if len(result) else 0
    accepted=int(sum(result.scientifically_accepted_proxy.fillna(False))) if len(result) else 0
    summary={
        "state":"complete" if len(result)==len(unique) else "partial",
        "original_patch_count":len(local),
        "distinct_patch_count":len(unique),
        "patches_scanned":len(result),
        "patches_remaining":len(unique)-len(result),
        "patches_scl_unavailable":unknown,
        "patches_proxy_accepted":accepted,
        "patches_proxy_abstained_or_unknown":len(result)-accepted,
        "field_rock_certifications":0,
        "all_original_images_unchanged":True,
        "test_labels_untouched":True,
        "limitations":[
            "SCL non-vegetated 5 is not field-confirmed rock.",
            "A pixel can be spectrally present yet snowy, water, or dark.",
            "Missing SCL gets explicit unknown state; it is not accepted.",
            "Scientific abstentions do not automatically remove training/test rows.",
        ],
    }
    (out/"scl_audit_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("SCL_AUDIT_STATE",json.dumps({k:v for k,v in summary.items()
          if not isinstance(v,list)}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--max-new",type=int,default=0)
    args=p.parse_args()
    out=Path(args.output)
    if not out.is_absolute():out=PROJECT_ROOT/out
    run(PROJECT_ROOT,out,max_new=args.max_new)

if __name__=="__main__":
    main()
