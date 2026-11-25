"""Stage temporal Sentinel-2 research patches using preregistered disjoint date windows."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.data.builder import _sentinel_config, _atomic_write_csv
from lithiumscope.tools.acquisition.sentinel2.provider import Sentinel2Provider, SentinelProviderError
from lithiumscope.tools.acquisition.sentinel2.registered_patch import (
    validate_interval, download_registered_patch, _atomic_json,
)


def validate_windows(windows):
    if len(windows)<3:
        raise ValueError("At least three preregistered windows required")
    intervals=sorted((validate_interval(window),window) for window in windows)
    if any(a[0][1]>=b[0][0] for a,b in zip(intervals,intervals[1:])):
        raise ValueError("Temporal acquisition windows must be disjoint")
    return [item[1] for item in intervals]


def run(root,manifest,expected_sha,output,windows,*,acquire=False,provider_factory=Sentinel2Provider):
    root,manifest,output=map(Path,(root,manifest,output))
    if not expected_sha or file_sha256(manifest)!=expected_sha:
        raise ValueError("Source manifest must match frozen hash")
    if not output.resolve().is_relative_to((root/"results").resolve()):
        raise ValueError("Temporal experiment must be isolated under results")
    windows=validate_windows(windows)
    # Never inspect Li or use its value to select dates or scenes.
    data=pd.read_csv(manifest,usecols=["sample_id","latitude","longitude"],dtype={"sample_id":str})
    if (data.empty or data.sample_id.isna().any() or not data.sample_id.is_unique
            or not data.latitude.between(-90,90).all() or not data.longitude.between(-180,180).all()):
        raise ValueError("Incomplete acquisition identities")
    config=load_config("model_2")
    policy={"version":1,"manifest_sha256":expected_sha,"windows":windows,"imagery":config["imagery"]}
    output.mkdir(parents=True,exist_ok=True)
    import json
    plan=output/"acquisition_plan.json"
    if plan.exists():
        if canonical_json_hash(json.loads(plan.read_text(encoding="utf-8")))!=canonical_json_hash(policy):
            raise ValueError("Output directory bound to another acquisition plan")
    else:
        _atomic_json(plan,policy)
    if not acquire:
        return {"state":"preflight","sites":len(data),"windows":len(windows),"planned_patches":len(data)*len(windows)}
    base=_sentinel_config(config)
    rows,failures=[],[]
    columns=["sample_id","image_path","image_sha256","sidecar_sha256","window","scene_id","datetime"]
    try:
        for window in windows:
            provider=provider_factory(replace(base,datetime=window))
            provider.validate()
            signature=canonical_json_hash({"plan":policy,"window":window})
            for sample in data.itertuples(index=False):
                sid=str(sample.sample_id)
                identity=canonical_json_hash({"sample":sid,"latitude":sample.latitude,"longitude":sample.longitude})
                dest=output/"patches"/signature/identity[:32]/"patch.npy"
                try:
                    path,meta=download_registered_patch(sid,float(sample.latitude),float(sample.longitude),dest,provider,signature)
                    rows.append({"sample_id":sid,"image_path":str(path.relative_to(output)),
                                 "image_sha256":file_sha256(path),"sidecar_sha256":file_sha256(path.with_suffix(".json")),
                                 "window":window,"scene_id":meta["scene_id"],"datetime":meta["datetime"]})
                except SentinelProviderError:
                    raise
                except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                    failures.append({"sample_id":sid,"window":window,"error":type(exc).__name__+": "+str(exc)[:250]})
    finally:
        _atomic_write_csv(pd.DataFrame(rows,columns=columns),output/"temporal_manifest.csv")
        _atomic_write_csv(pd.DataFrame(failures,columns=["sample_id","window","error"]),output/"failures.csv")
    if file_sha256(manifest)!=expected_sha:
        raise RuntimeError("Original manifest changed")
    complete=len(rows)==len(data)*len(windows)
    report={"state":"complete" if complete else "incomplete","sites":len(data),"windows":len(windows),
            "ready":len(rows),"failed":len(failures),"source_unchanged":True,"label_selection":False}
    _atomic_json(output/"summary.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--window",required=True,action="append")
    parser.add_argument("--acquire",action="store_true")
    args=parser.parse_args()
    print(run(PROJECT_ROOT,args.manifest,args.sha256,args.output,args.window,acquire=args.acquire))


if __name__=="__main__":
    main()
