"""Normal inference requires an explicitly activated model, never newest candidate."""
from __future__ import annotations

from pathlib import Path
import json

from lithiumscope.persistence.model_deserialization import load_model_artifact

from lithiumscope.core.exceptions import ModelNotReadyError
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.persistence.active_models import active_model_path


def latest_model_path(model_group: str, prefix: str | None = None) -> Path:
    active=active_model_path(model_group)
    if active is None:
        raise ModelNotReadyError(f"No explicitly active model for {model_group}. Load a verified historical release.")
    if prefix and prefix not in active.name and prefix not in active.parent.name:
        raise ModelNotReadyError(f"Active {model_group} model does not match requested prefix.")
    return active


def load_latest_model(model_group: str, prefix: str | None = None):
    path=latest_model_path(model_group,prefix=prefix)
    integrity_path=path.with_name("artifact_integrity.json")
    if integrity_path.is_file():
        record=json.loads(integrity_path.read_text(encoding="utf-8"))
        files=record.get("files",{})
        if not {"model.joblib","metadata.json"}<=set(files) or any(
            Path(name).name!=name or not (path.parent/name).is_file() or file_sha256(path.parent/name)!=digest
            for name,digest in files.items()):
            raise ModelNotReadyError("Active artifact integrity check failed.")
    metadata_path=path.with_name("metadata.json")
    if metadata_path.is_file():
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        expected=metadata.get("model_sha256")
        if expected and file_sha256(path)!=expected:
            raise ModelNotReadyError("Active model integrity check failed.")
    # An active model was explicitly selected via a trusted local/release workflow.
    return load_model_artifact(path, model_group=model_group),path
