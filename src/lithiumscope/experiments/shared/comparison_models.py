"""Read-only comparison inventory with verified local model bytes."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from hashlib import sha256

import joblib

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.persistence.active_models import active_model_path, read_active_models
from lithiumscope.prediction.execution.session import _latest_local_candidate
from lithiumscope.persistence.candidates import read_candidate_status


def _verified(path, group, *, payload=None):
    path=Path(path)
    meta=json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    if meta.get("model_group")!=group or meta.get("run_id")!=path.parent.name:
        raise ValueError("Model identity mismatch")
    integrity=path.with_name("artifact_integrity.json")
    if integrity.is_file():
        entries=json.loads(integrity.read_text(encoding="utf-8"))["files"]
        if not {"model.joblib","metadata.json"}<=set(entries) or any(
            Path(name).name!=name or (path.parent/name).is_symlink()
            or file_sha256(path.parent/name)!=digest for name,digest in entries.items()):
            raise ValueError("Model artifact integrity mismatch")
    raw=path.read_bytes() if payload is None else payload
    digest=sha256(raw).hexdigest()
    if digest!=meta.get("model_sha256"):
        raise ValueError("Model SHA256 mismatch")
    bundle=joblib.load(BytesIO(raw))
    if not isinstance(bundle,dict) or "estimator" not in bundle:
        raise ValueError("Invalid fitted model bundle")
    return bundle,{"group":group,"run_id":meta["run_id"],"algorithm":meta.get("algorithm",bundle.get("algorithm")),
                   "path":str(path),"sha256":digest,"dataset_sha256":meta.get("dataset_sha256"),
                   "feature_extractor_version":bundle.get("feature_extractor_version")}


def discover_models(group):
    """A release is not silently replaced by a candidate, or vice versa."""
    result, issues={},[]
    try:
        path=active_model_path(group)
        entry=read_active_models()["models"].get(group,{})
        if path is None or entry.get("source",{}).get("type")!="github_release":
            raise FileNotFoundError("No hay release activo instalado desde GitHub (opción principal 4)")
        bundle,identity=_verified(path,group)
        identity["release_tag"]=entry["source"].get("tag")
        result["release"]=(bundle,identity)
    except Exception as exc:
        issues.append({"role":"release","state":"BLOQUEADO" if isinstance(exc,FileNotFoundError) else "ERROR","reason":str(exc)})
    try:
        run_id,payload=_latest_local_candidate(group)
        path=MODELS_DIR/group/"trained"/run_id/"model.joblib"
        if read_candidate_status(path).get("status")!="pending":
            raise ValueError("Candidate state changed")
        result["candidate"]=_verified(path,group,payload=payload)
    except Exception as exc:
        issues.append({"role":"candidate","state":"BLOQUEADO" if isinstance(exc,FileNotFoundError) else "ERROR","reason":str(exc)})
    return result,issues
