"""Persist candidates atomically, without overwriting a published model identity."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import joblib

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import CONFIG_DIR, MODELS_DIR
from lithiumscope.persistence.active_models import set_active_model
from lithiumscope.persistence.final_fit_contract import verify_frozen_fit, verify_serialized_candidate


def _serializable_schema(schema: Any) -> Any:
    if schema is None:
        return None
    if hasattr(schema,"__dict__"):
        return dict(schema.__dict__)
    return schema


def _write_json(path,payload):
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding="utf-8")


def save_model_bundle(
    model_group: str, name: str, bundle: dict[str,Any], metadata: dict[str,Any], *,
    run_id: str | None = None, dataset_manifest_path: Path | None = None,
    feature_schema: Any = None, config_name: str | None = None,
    activate: bool = False, verification_probe=None, config_snapshot_path: Path | None = None,
) -> tuple[Path,Path]:
    if model_group not in {"model_1","model_2"}:
        raise ValueError("Unsupported model group")
    stamp=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not stamp or Path(stamp).name!=stamp or stamp in {".",".."} or "/" in stamp or "\\" in stamp:
        raise ValueError("Invalid artifact identity")
    if activate and (bundle.get("experimental_only") or metadata.get("experimental_only") or metadata.get("activation_forbidden")):
        raise ValueError("Experimental candidate cannot be activated during save")
    contract=bundle.get("final_fit_contract")
    if contract is not None:
        verify_frozen_fit(contract)
        if activate or verification_probe is None:
            raise ValueError("Frozen final fit requires nonactivating serialization verification")
        if metadata.get("final_fit_contract_sha256")!=contract["identity_sha256"]:
            raise ValueError("Metadata does not identify frozen final fit")
    artifact_dir=MODELS_DIR/model_group/"trained"/stamp
    if artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model version: {artifact_dir}")
    artifact_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=artifact_dir.parent/f".staging_{uuid4().hex}"
    stage.mkdir()
    model_path=artifact_dir/"model.joblib"
    metadata_path=artifact_dir/"metadata.json"
    try:
        staged_model=stage/"model.joblib"
        joblib.dump(bundle,staged_model)
        parity=None
        if verification_probe is not None:
            # Only deserialize the bundle just created by this process.
            parity=verify_serialized_candidate(bundle,joblib.load(staged_model),verification_probe,model_group=model_group)
        payload={**metadata,"artifact_schema_version":2,"artifact_name":name,"run_id":stamp,
                 "model_path":str(model_path.relative_to(MODELS_DIR.parent)),
                 "model_sha256":file_sha256(staged_model),
                 "saved_at_utc":datetime.now(timezone.utc).isoformat(),
                 "activation_at_save":"active" if activate else "candidate",
                 "serialization_verification":parity}
        _write_json(stage/"metadata.json",payload)
        schema=_serializable_schema(feature_schema)
        if schema is not None:
            _write_json(stage/"feature_schema.json",schema)
        if dataset_manifest_path is not None:
            shutil.copy2(dataset_manifest_path,stage/"dataset_manifest.json")
        if config_name:
            path=config_snapshot_path or CONFIG_DIR/f"{config_name}.yaml"
            if path.exists():
                shutil.copy2(path,stage/"training_config.yaml")
            elif contract is not None:
                raise FileNotFoundError("Frozen candidate requires training config snapshot")
        if contract is not None:
            _write_json(stage/"final_fit_contract.json",contract)
        if not activate:
            _write_json(stage/"candidate_status.json",{"schema_version":1,"status":"pending",
                "run_id":stamp,"model_group":model_group,"created_at_utc":datetime.now(timezone.utc).isoformat()})
        immutable={p.name:file_sha256(p) for p in sorted(stage.iterdir()) if p.name!="candidate_status.json"}
        _write_json(stage/"artifact_integrity.json",{"version":1,"files":immutable})
        # Rename publishes the complete directory in one filesystem operation.
        # Refuse replacement even when a second process races to use this run ID.
        if artifact_dir.exists():
            raise FileExistsError(f"Refusing to overwrite model version: {artifact_dir}")
        stage.rename(artifact_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if activate:
        set_active_model(model_group,model_path,source={"type":"local_training","run_id":stamp})
    return model_path,metadata_path
