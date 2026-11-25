"""Pre-training M2 reservation; labels remain with the independent custodian."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
from sklearn.neighbors import BallTree

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.scientific.spatial_audit import geographic_coordinates, EARTH_RADIUS_M
from lithiumscope.evaluation.holdout import registry as reg


def _training_overlap(sites, manifest, minimum_distance):
    from lithiumscope.datasets.validation.paths import resolve_image_path
    from lithiumscope.evaluation.holdout.registry import normalized
    required={"sample_id","spatial_group","latitude","longitude","image_path"}
    if not required<=set(manifest) or manifest.empty:
        raise ValueError("Cannot certify development cohort identities")
    for field in ("sample_id","spatial_group","scene_id"):
        if field in manifest and field in sites and set(normalized(manifest[field])) & set(normalized(sites[field])):
            raise ValueError(f"Reserved cohort overlaps development {field}")
    image_hashes={file_sha256(resolve_image_path(p)) for p in manifest.image_path}
    if image_hashes & set(sites.image_sha256):
        raise ValueError("Reserved images overlap development images")
    training=geographic_coordinates(manifest[["latitude","longitude"]])
    external=geographic_coordinates(sites[["latitude","longitude"]])
    distance=BallTree(training,metric="haversine").query(external,k=1)[0][:,0]*EARTH_RADIUS_M
    if (distance<minimum_distance).any() or (distance<=np.sqrt(2)*640).any():
        raise ValueError("Reserved cohort violates the development geographic buffer")


def reserve(root, *, plan_path, manifest_path, fixed_endpoint):
    """Seal site, imagery, exposure and reference evidence before invoking competition."""
    root=Path(root).resolve()
    if root.exists():
        raise FileExistsError("Reservation already exists; cannot reuse a consulted holdout")
    plan_record=reg.artifact(plan_path)
    plan=json.loads(Path(plan_record["path"]).read_text(encoding="utf-8"))
    allowed={"sites_path","exposure_paths","review_path","reference","label_commitment_sha256",
             "minimum_distance_m","label_cutoff_ppm","min_samples","min_regions","min_groups"}
    if set(plan)-allowed:
        raise ValueError("Unknown reservation field; labels must remain outside training")
    base=Path(plan_record["path"]).parent
    for key in ("sites_path","review_path","reference"):
        plan[key]=str((base/plan[key]).resolve(strict=True))
    plan["exposure_paths"]=[str((base/p).resolve(strict=True)) for p in plan["exposure_paths"]]
    if fixed_endpoint is None or float(fixed_endpoint)!=float(plan["label_cutoff_ppm"]):
        raise ValueError("Prospective comparison needs the same fixed Li endpoint in M2 configuration and plan")
    if not reg.valid_sha(plan["label_commitment_sha256"]):
        raise ValueError("Missing label commitment")
    sites=reg.read_csv(plan["sites_path"])
    exposures=[reg.artifact(p) for p in plan["exposure_paths"]]
    audit=reg.exposure_audit(sites,[reg.read_csv(p["path"]) for p in exposures],model_group="model_2",
                             minimum_distance_m=float(plan["minimum_distance_m"]))
    for key,default,column in (("min_samples",100,None),("min_regions",3,"region"),("min_groups",5,"spatial_group")):
        value=plan.get(key,default)
        support=len(sites) if column is None else reg.normalized(sites[column]).nunique()
        if type(value) is not int or value<default or support<value:
            raise ValueError(f"Insufficient reserved cohort support: {key}")
    reg.review_document(plan["review_path"])
    _training_overlap(sites,reg.read_csv(manifest_path),float(plan["minimum_distance_m"]))
    # Prior reservations, even abandoned ones, must not be silently reused.
    for earlier in root.parent.glob("*/reservation.json"):
        if reg.read_sealed(earlier)["plan"]["label_commitment_sha256"]==plan["label_commitment_sha256"]:
            raise ValueError("Label commitment already reserved")
    record={"version":1,"reserved_at_utc":datetime.now(timezone.utc).isoformat(),
            "plan":plan,"plan_artifact":plan_record,"manifest":reg.artifact(manifest_path),
            "sites":reg.artifact(plan["sites_path"]),"review":reg.artifact(plan["review_path"]),
            "reference":reg.artifact(plan["reference"]),"exposures":exposures,
            "imagery":reg.freeze_imagery(sites,Path(plan["sites_path"])),"audit":audit,
            "scope":"Pre-training reservation; no labels read, no proof of documentary truth"}
    root.mkdir(parents=True,exist_ok=False)
    reg.write_sealed(root/"reservation.json",record)
    return record


def verify(root):
    record=reg.read_sealed(Path(root)/"reservation.json")
    for key in ("plan_artifact","manifest","sites","review","reference"):
        reg.verify_artifact(record[key])
    for item in record["exposures"]:
        reg.verify_artifact(item)
    for item in record["imagery"]:
        reg.verify_artifact(item["image"])
        reg.verify_artifact(item["sidecar"])
    _training_overlap(reg.read_csv(record["sites"]["path"]),reg.read_csv(record["manifest"]["path"]),
                      float(record["plan"]["minimum_distance_m"]))
    return record


def bind_and_score(root, candidate, *, trusted_local_bundles=False):
    if not trusted_local_bundles:
        raise ValueError("Explicit trust required for local model deserialization")
    root=Path(root)
    record=verify(root)
    reg.register(root/"holdout",model_group="model_2",candidate=candidate,**record["plan"])
    from lithiumscope.evaluation.holdout.scoring import score_registered
    scores=score_registered(root/"holdout",trusted_local_bundles=True)
    reg.write_sealed(root/"scoring_link.json",{"reservation_sha256":file_sha256(root/"reservation.json"),
        "registration_sha256":file_sha256(root/"holdout/registration.json"),
        "scores_sha256":file_sha256(root/"holdout/scores.json"),"labels_read":False})
    return scores
