"""Recover SCL evidence from exact archived scenes without changing frozen pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path
from lithiumscope.experiments.model_2.external_validation.imagery.scene_inventory import fetch_metadata
from lithiumscope.model_2.methodology.spectral.satellite_quality import analyze_arrays, ScientificScenePolicy
from lithiumscope.tools.acquisition.sentinel2.provider import _read_patch
from lithiumscope.tools.acquisition.sentinel2.registered_patch import _atomic_json, _atomic_npy

POLICY = "historical_scl_exact_scene_legacy_grid_v2"


def identity(row, image_hash, metadata_hash):
    return canonical_json_hash({"policy": POLICY, "image_sha256": image_hash,
        "scene": str(row.sentinel_scene_id), "latitude": float(row.latitude),
        "longitude": float(row.longitude), "metadata_sha256": metadata_hash})


def recover(root, manifest, expected_sha, output, *, metadata_cache, online=False,
            max_new=0, reader=_read_patch):
    root, manifest, output, metadata_cache = map(Path, (root, manifest, output, metadata_cache))
    if not expected_sha or file_sha256(manifest) != expected_sha or max_new < 0:
        raise ValueError("Unverified archived manifest or invalid budget")
    data = pd.read_csv(manifest,dtype={"sample_id":str,"sentinel_scene_id":str})
    required = {"sample_id", "image_path", "sentinel_scene_id", "latitude", "longitude"}
    if not required <= set(data) or data.sample_id.isna().any() or not data.sample_id.is_unique or not data.latitude.between(-90,90).all() or not data.longitude.between(-180,180).all():
        raise ValueError("Archived manifest identity incomplete")
    if not output.resolve().is_relative_to((root/"results").resolve()):
        raise ValueError("Recovery must remain under isolated results directory")
    output.mkdir(parents=True, exist_ok=True)
    run_contract = output/"recovery_contract.json"
    contract = {"manifest_sha256": expected_sha, "policy": POLICY}
    if run_contract.exists():
        if json.loads(run_contract.read_text(encoding="utf-8")) != contract:
            raise ValueError("Recovery directory belongs to another cohort")
    else:
        _atomic_json(run_contract, contract)
    evidence = output/"masks"
    evidence.mkdir(exist_ok=True)
    rows, original_hashes, attempted = [], {}, 0
    for position, row in enumerate(data.itertuples(index=False)):
        record = {"position": position, "sample_id": str(row.sample_id),
                  "scene_id": str(row.sentinel_scene_id), "state": "unavailable"}
        try:
            path = resolve_image_path(row.image_path, root=root)
            image_hash = file_sha256(path)
            if path in original_hashes and image_hash != original_hashes[path]:
                raise ValueError("Archived image changed during audit")
            original_hashes[path] = image_hash
            side = path.with_suffix(".json")
            side_hash = file_sha256(side)
            original_hashes.setdefault(side, side_hash)
            metadata = json.loads(side.read_text(encoding="utf-8"))
            if (metadata.get("scene_id") != row.sentinel_scene_id
                    or str(metadata.get("sample_id")) != str(row.sample_id)
                    or not np.allclose([metadata["latitude"],metadata["longitude"]],
                                       [row.latitude,row.longitude],rtol=0,atol=1e-6)):
                raise ValueError("Archived patch identity mismatch")
            if metadata.get("image_sha256") and metadata["image_sha256"] != image_hash:
                raise ValueError("Archived image differs from declared hash")
            item, state = fetch_metadata(str(row.sentinel_scene_id), metadata_cache, remote=online)
            if item is None:
                raise ValueError("Scene metadata unavailable: "+state)
            if item.get("id") != row.sentinel_scene_id or item.get("collection") != "sentinel-2-l2a":
                raise ValueError("Scene identity mismatch")
            item_hash = canonical_json_hash(item)
            key = identity(row,image_hash,item_hash)
            mask = evidence/(key+".npy")
            report_path = evidence/(key+".json")
            record.update(image_sha256=image_hash, sidecar_sha256=side_hash, scene_metadata_sha256=item_hash,
                          evidence_key=key)
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("evidence_key") != key or report.get("scl_sha256") != file_sha256(mask):
                    raise ValueError("Recovery mask evidence differs")
            else:
                if max_new and attempted >= max_new:
                    record["state"] = "deferred_budget"
                    rows.append(record)
                    continue
                href = item["assets"]["scl"]["href"]
                if not online and not Path(href).is_file():
                    record["state"] = "deferred_online_authorization"
                    rows.append(record)
                    continue
                attempted += 1
                raw = np.load(path,allow_pickle=False)
                scl = reader(href,float(row.latitude),float(row.longitude),640,32,categorical=True)
                qa = analyze_arrays(raw,scl,policy=ScientificScenePolicy())
                _atomic_npy(mask,scl)
                report = {"evidence_key":key,"scl_sha256":file_sha256(mask),"qa":qa,
                          "geometry":"legacy_window_reconstructed_not_independently_verified"}
                _atomic_json(report_path,report)
            record.update(state="recovered", scl_sha256=report["scl_sha256"],
                          local_proxy_accepted=report["qa"]["scientifically_accepted_proxy"],
                          geometry=report["geometry"])
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            record["reason"] = type(exc).__name__+": "+str(exc)[:250]
        rows.append(record)
    if file_sha256(manifest) != expected_sha or any(file_sha256(p)!=h for p,h in original_hashes.items()):
        raise RuntimeError("Archived source bytes changed")
    table = pd.DataFrame(rows)
    table.to_csv(output/"per_sample_recovery.csv",index=False)
    summary = {"rows":len(data),"recovered":int(table.state.eq("recovered").sum()),
               "state_counts":table.state.value_counts().to_dict(),"new_attempts":attempted,
               "manifest_sha256":expected_sha,"original_bytes_unchanged":True,
               "historical_coregistration_certified":False,"field_rock_certified":False}
    _atomic_json(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--metadata-cache",required=True,type=Path)
    parser.add_argument("--online",action="store_true")
    parser.add_argument("--max-new",type=int,default=0)
    args=parser.parse_args()
    print(recover(PROJECT_ROOT,args.manifest,args.sha256,args.output,metadata_cache=args.metadata_cache,
                  online=args.online,max_new=args.max_new))


if __name__=="__main__":
    main()
