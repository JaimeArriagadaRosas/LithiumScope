from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import joblib
import yaml

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import CONFIG_DIR, MODELS_DIR
from lithiumscope.persistence.active_models import set_active_model


def _serializable_schema(schema: Any) -> Any:
    if schema is None:
        return None
    if hasattr(schema, "__dict__"):
        return dict(schema.__dict__)
    return schema


def save_model_bundle(
    model_group: str,
    name: str,
    bundle: dict[str, Any],
    metadata: dict[str, Any],
    *,
    run_id: str | None = None,
    dataset_manifest_path: Path | None = None,
    feature_schema: Any = None,
    config_name: str | None = None,
) -> tuple[Path, Path]:
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = MODELS_DIR / model_group / "trained" / stamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    joblib.dump(bundle, model_path)

    payload = {
        **metadata,
        "artifact_schema_version": 1,
        "artifact_name": name,
        "run_id": stamp,
        "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
        "model_sha256": file_sha256(model_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    schema_payload = _serializable_schema(feature_schema)
    if schema_payload is not None:
        (artifact_dir / "feature_schema.json").write_text(
            json.dumps(schema_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if dataset_manifest_path is not None and dataset_manifest_path.exists():
        shutil.copy2(dataset_manifest_path, artifact_dir / "dataset_manifest.json")

    if config_name:
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if config_path.exists():
            shutil.copy2(config_path, artifact_dir / "training_config.yaml")

    set_active_model(
        model_group,
        model_path,
        source={
            "type": "local_training",
            "run_id": stamp,
        },
    )
    return model_path, metadata_path
