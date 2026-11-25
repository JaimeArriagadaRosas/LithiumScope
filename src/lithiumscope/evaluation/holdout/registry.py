"""Freeze external site identities, exposure inventory, trusted model hashes and label commitment."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.core.scientific.spatial_audit import geographic_coordinates, EARTH_RADIUS_M

IDENTITY = ("sample_id", "source_id", "site_id", "spatial_group", "latitude", "longitude")
UNKNOWN = {"", "unknown", "unreported", "none", "nan", "n/a", "na", "null"}
BLIND_COLUMNS = {"li", "lippm", "liicpms", "lithium", "lithiumppm", "ytrue", "label", "target",
                 "score", "prospectivityscore", "liicpmspredicted", "candidateprediction", "referenceprediction"}


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        header=next(csv.reader(handle),[])
    fields=[c.strip().casefold() for c in header]
    if not fields or any(not c for c in fields) or len(fields)!=len(set(fields)):
        raise ValueError("Holdout CSV requires unique nonempty headers")
    return pd.read_csv(path, dtype={key:"string" for key in (
        "sample_id", "source_id", "site_id", "spatial_group", "region", "scene_id", "image_sha256",
        "sidecar_sha256", "rededge_patch_sha256", "scene_band_hash")}, keep_default_na=False)


def normalized(values):
    return values.astype(str).str.strip().str.casefold()


def check_identity(frame, *, external, model_group):
    required = set(IDENTITY) | ({"region"} if external else set())
    if model_group == "model_2":
        required |= {"scene_id", "image_sha256"}
        if external:
            required |= {"image_path", "sidecar_sha256"}
    if frame.empty or not required <= set(frame):
        raise ValueError(f"Missing identity or no rows: {sorted(required-set(frame))}")
    for column in required - {"latitude","longitude"}:
        if normalized(frame[column]).isin(UNKNOWN).any():
            raise ValueError(f"Unknown required provenance: {column}")
    if external:
        if normalized(frame.sample_id).duplicated().any():
            raise ValueError("Repeated external sample identity")
        headers = {re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in frame.columns}
        if BLIND_COLUMNS & headers or any(c.startswith("liicpms") for c in headers):
            raise ValueError("External registration cannot contain labels or predictions")
    if model_group == "model_2":
        for column in ("image_sha256","sidecar_sha256"):
            if column in required and not frame[column].map(valid_sha).all():
                raise ValueError(f"Invalid image identity: {column}")
    return geographic_coordinates(frame[["latitude","longitude"]])


def exposure_audit(sites, exposures, *, model_group, minimum_distance_m):
    if not np.isfinite(minimum_distance_m) or minimum_distance_m < 100:
        raise ValueError("Declare a finite minimum distance of at least 100 m")
    if model_group not in {"model_1","model_2"} or not exposures:
        raise ValueError("Select a model group and provide the complete exposure inventory")
    external_coords=check_identity(sites,external=True,model_group=model_group)
    for exposed in exposures:
        check_identity(exposed,external=False,model_group=model_group)
    all_exposed=pd.concat(exposures,ignore_index=True)
    compared=["sample_id","source_id","site_id","spatial_group"]
    if model_group=="model_2":
        compared += ["scene_id","image_sha256"]
    for column in compared:
        if set(normalized(sites[column])) & set(normalized(all_exposed[column])):
            raise ValueError(f"Holdout overlaps previously consulted {column}")
    d,_=BallTree(geographic_coordinates(all_exposed[["latitude","longitude"]]),metric="haversine").query(external_coords,k=1)
    distance=d[:,0]*EARTH_RADIUS_M
    if (distance<minimum_distance_m).any():
        raise ValueError("Holdout fails predeclared geographic buffer")
    if model_group=="model_2" and (distance<=np.sqrt(2)*640).any():
        raise ValueError("Possible overlapping historical Sentinel patches")
    return {"exposed_rows":len(all_exposed), "holdout_rows":len(sites),
            "minimum_nearest_distance_m":float(distance.min()),
            "median_nearest_distance_m":float(np.median(distance)), "required_distance_m":float(minimum_distance_m),
            "checked_identity_fields":compared,"numeric_overlap_detected":False,
            "independence_documentary_review_required":True}


def artifact(path):
    path=Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Required artifact must be a local file")
    return {"path":str(path),"sha256":file_sha256(path)}


def verify_artifact(record):
    path=Path(record["path"])
    if not path.is_file() or file_sha256(path)!=record["sha256"]:
        raise ValueError(f"Frozen artifact missing or changed: {path.name}")
    return path


def write_sealed(path, payload):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump({"payload":payload,"sha256":canonical_json_hash(payload)},handle,indent=2,ensure_ascii=False,allow_nan=False)


def read_sealed(path):
    record=json.loads(Path(path).read_text(encoding="utf-8"))
    if canonical_json_hash(record["payload"])!=record["sha256"]:
        raise ValueError("Registration or evidence seal changed")
    return record["payload"]


def review_document(path):
    review=json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("reviewer","label_custodian","documentary_basis"):
        if not isinstance(review.get(field),str) or len(review[field].strip())<3:
            raise ValueError(f"Missing independent review field: {field}")
    for field in ("complete_exposure_inventory","independent_source_and_site","labels_unconsulted",
                  "site_identity_verified","coordinates_wgs84_verified"):
        if review.get(field) is not True:
            raise ValueError(f"Documentary attestation absent: {field}")
    return review


def freeze_imagery(sites, sites_path):
    images=[]
    for row in sites.itertuples(index=False):
        image=(sites_path.parent/str(row.image_path)).resolve(strict=True)
        base=artifact(image)
        sidecar=next((p for p in (image.with_suffix(image.suffix+".json"),image.with_suffix(".json")) if p.is_file()),None)
        if sidecar is None:
            raise ValueError("Every external image needs a local provenance sidecar")
        meta=json.loads(sidecar.read_text(encoding="utf-8"))
        if base["sha256"]!=row.image_sha256 or file_sha256(sidecar)!=row.sidecar_sha256:
            raise ValueError("Image or provenance sidecar hash mismatch")
        if (meta.get("scene_id")!=row.scene_id or meta.get("collection")!="sentinel-2-l2a"
                or not np.allclose([float(meta["latitude"]),float(meta["longitude"])],
                                   [float(row.latitude),float(row.longitude)],rtol=0,atol=1e-8)):
            raise ValueError("External image provenance does not match registered site")
        images.append({"sample_id":str(row.sample_id),"image":base,"sidecar":artifact(sidecar)})
    return images


def register(root, *, model_group, sites_path, exposure_paths, reference, candidate, review_path,
             label_commitment_sha256, minimum_distance_m, label_cutoff_ppm=None,
             min_samples=100,min_regions=3,min_groups=5,bootstrap_iterations=500,seed=42):
    root=Path(root).resolve()
    if root.exists():
        raise FileExistsError("A prospective registration cannot overwrite existing evidence")
    if not valid_sha(label_commitment_sha256):
        raise ValueError("Supply the custodian's label-file SHA256 without opening labels")
    if any(type(n) is not int or n<1 for n in (min_samples,min_regions,min_groups,bootstrap_iterations)):
        raise ValueError("Invalid support or bootstrap policy")
    if min_groups<5 or bootstrap_iterations<100:
        raise ValueError("At least five groups and 100 bootstrap iterations required")
    if model_group=="model_2" and (label_cutoff_ppm is None or not np.isfinite(label_cutoff_ppm) or label_cutoff_ppm<0):
        raise ValueError("M2 requires a common fixed Li endpoint before scoring")
    sites_record=artifact(sites_path)
    sites_path=Path(sites_record["path"])
    sites=read_csv(sites_path)
    exposure_records=[artifact(p) for p in exposure_paths]
    overlap=exposure_audit(sites,[read_csv(Path(p["path"])) for p in exposure_records],
                           model_group=model_group,minimum_distance_m=minimum_distance_m)
    if len(sites)<min_samples or normalized(sites.region).nunique()<min_regions or normalized(sites.spatial_group).nunique()<min_groups:
        raise ValueError("Registered holdout lacks predeclared sample, region or group support")
    review_record=artifact(review_path)
    review_document(review_path)
    models={"reference":artifact(reference),"candidate":artifact(candidate)}
    if models["reference"]["sha256"]==models["candidate"]["sha256"]:
        raise ValueError("Reference and candidate artifacts are identical")
    imagery=freeze_imagery(sites,sites_path) if model_group=="model_2" else []
    for earlier in root.parent.glob("*/registration.json"):
        if read_sealed(earlier)["label_commitment_sha256"]==label_commitment_sha256:
            raise ValueError("This label commitment was already registered in the local registry")
    record={"version":1,"registered_at_utc":datetime.now(timezone.utc).isoformat(),"model_group":model_group,
            "sites":sites_record,"exposure":exposure_records,"review":review_record,"models":models,
            "label_commitment_sha256":label_commitment_sha256,"label_cutoff_ppm":label_cutoff_ppm,
            "support":{"min_samples":min_samples,"min_regions":min_regions,"min_groups":min_groups},
            "bootstrap":{"iterations":bootstrap_iterations,"seed":seed},
            "imagery":imagery,"overlap_audit":overlap,
            "scope":"Local prospective record; hash seals detect drift, not prior human access or backdating; no release authorization."}
    if model_group=="model_2":
        from dataclasses import asdict
        from lithiumscope.model_2.evaluation.release_gate import ExternalGatePolicy
        record["m2_gate_policy"]=asdict(ExternalGatePolicy(min_samples=min_samples,min_geographic_regions=min_regions))
    root.mkdir(parents=True,exist_ok=False)
    write_sealed(root/"registration.json",record)
    return record


def verified_registration(root):
    record=read_sealed(Path(root)/"registration.json")
    sites=read_csv(verify_artifact(record["sites"]))
    exposures=[read_csv(verify_artifact(p)) for p in record["exposure"]]
    review_document(verify_artifact(record["review"]))
    for model in record["models"].values():
        verify_artifact(model)
    for image in record["imagery"]:
        verify_artifact(image["image"])
        verify_artifact(image["sidecar"])
    current=exposure_audit(sites,exposures,model_group=record["model_group"],
                           minimum_distance_m=record["overlap_audit"]["required_distance_m"])
    if current!=record["overlap_audit"]:
        raise ValueError("Exposure audit no longer matches frozen registration")
    return record,sites
