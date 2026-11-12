"""Explicit same-day Sentinel-2 scientific candidate acquisition (opt-in CLI).

Preview defaults to report only. --save-candidate writes a NEW, staged NPY
with QA sidecar, never overwrites historical image paths, training manifests,
feature caches, active models or existing candidate files.

This is technical optical screening, NOT a scientifically verified Li sample.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from datetime import date
import hashlib

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.builder import _sentinel_config
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider,download_patch
)
from lithiumscope.model_2.methodology.satellite_inventory import (
    MANIFEST, EXPECTED_SHA, sha
)


def proposal_for_sample(root:Path,sample_id:str, *,
                        selected_day:str|None=None,
                        out:Path,save_candidate:bool=False,
                        provider_factory=None)->dict:
    """Select/preview new imagery from the same acquisition date by default."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}",sample_id):
        raise ValueError("Unsafe sample ID")
    manifest=root/MANIFEST
    if sha(manifest)!=EXPECTED_SHA:
        raise ValueError("Frozen training manifest changed")
    df=pd.read_csv(manifest)
    hit=df.loc[df.sample_id.astype(str)==sample_id]
    if len(hit)!=1:raise ValueError("Sample must be present exactly once")
    row=hit.iloc[0]
    scene_id=str(row.sentinel_scene_id)
    original=Path(str(row.image_path))
    if not original.is_file():raise FileNotFoundError(original)
    original_hash=sha(original)
    if selected_day:
        day=date.fromisoformat(selected_day).isoformat()
    else:
        token=scene_id.split("_")[2]
        day=date(int(token[:4]),int(token[4:6]),int(token[6:8])).isoformat()
    interval=day+"T00:00:00Z/"+day+"T23:59:59Z"
    config=_sentinel_config(load_config("model_2"))
    provider=(provider_factory(config) if provider_factory is not None
              else Sentinel2Provider(config))
    scene,quality,audits=provider.search_validated_scene(
        float(row.latitude),float(row.longitude),datetime=interval)
    result={
        "sample_id":sample_id,
        "original_scene_id":scene_id,
        "original_image_sha256":original_hash,
        "candidate_scene_id":scene["id"],
        "explicit_acquisition_date":day,
        "historical_image_untouched":True,
        "label_unchanged":True,
        "field_rock_correspondence_verified":False,
        "optical_acceptance":True,
        "candidate_saved":False,
        "candidate_quality":quality,
        "candidate_audits":[
            {k:v for k,v in item.items() if k!="radiometry"} for item in audits
        ],
    }
    out.mkdir(parents=True,exist_ok=True)
    target=out/(sample_id+"__"+str(scene["id"])+".npy")
    sidecar=target.with_suffix(".json")
    report=out/(sample_id+"__selection_review.json")
    if save_candidate:
        if target.exists() or sidecar.exists():
            raise FileExistsError("Refusing to overwrite existing scientific candidate")
        path,metadata=download_patch(
            sample_id,float(row.latitude),float(row.longitude),
            target,provider,selected_scene=scene,scientific_quality=quality)
        if sha(original)!=original_hash:
            raise RuntimeError("Historical NPY changed during scientific acquisition")
        result["candidate_saved"]=True
        result["candidate_path"]=str(path)
        result["candidate_sha256"]=sha(path)
        result["sidecar_path"]=str(sidecar)
    report.write_text(json.dumps(
        result,indent=2,ensure_ascii=False),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-id",required=True)
    p.add_argument("--date",help="Explicit research date; defaults to original same day")
    p.add_argument("--output",required=True)
    p.add_argument("--save-candidate",action="store_true",
                   help="Write NEW staged candidate NPY; never overwrite sources")
    args=p.parse_args()
    dest=Path(args.output)
    if not dest.is_absolute():dest=PROJECT_ROOT/dest
    result=proposal_for_sample(
        PROJECT_ROOT,args.sample_id,selected_day=args.date,
        out=dest,save_candidate=args.save_candidate)
    print(json.dumps({
        "sample":result["sample_id"],
        "chosen":result["candidate_scene_id"],
        "day":result["explicit_acquisition_date"],
        "saved":result["candidate_saved"],
        "old_untouched":result["historical_image_untouched"],
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
