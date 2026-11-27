"""Hash-bound spectral cache; sample identity stays outside predictor columns."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path
from lithiumscope.model_2.preparation.features.spectral import (
    FEATURE_EXTRACTOR_VERSION, extract_spectral_features,
)


def validate_manifest(manifest: pd.DataFrame, config: dict) -> None:
    training = config["training"]
    target = training["lithium_column"]
    group = training.get("spatial_group_column", "spatial_group")
    sample = training.get("sample_id_column", "sample_id")
    required = {target, group, sample, "image_path"}
    if not required.issubset(manifest) or manifest.empty:
        raise ValueError("Missing required spectral manifest fields or empty cohort")
    for key in (sample, group, "image_path"):
        values = manifest[key]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing spectral manifest identity: {key}")
    if manifest[sample].astype(str).duplicated().any():
        raise ValueError("Duplicate sample IDs in spectral manifest")
    values = pd.to_numeric(manifest[target], errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid physical concentration in spectral manifest")


def source_identity(manifest_path: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    sample = config["training"].get("sample_id_column", "sample_id")
    manifest = pd.read_csv(manifest_path,dtype={sample:"string"})
    validate_manifest(manifest, config)
    hashes = {}
    identities = []
    for key, raw in zip(manifest[sample], manifest["image_path"]):
        path = resolve_image_path(raw)
        if path not in hashes:
            hashes[path] = file_sha256(path)
        identities.append({"sample_id": str(key), "image_sha256": hashes[path]})
    bands = config["imagery"]["bands"]
    columns = list(extract_spectral_features(np.ones((len(bands), 2, 2), dtype=np.float32), bands))
    columns += [config["training"]["lithium_column"], config["training"].get("spatial_group_column", "spatial_group")]
    module = Path(__file__).resolve().parents[1]
    code = [module / "preparation" / name for name in (
        "imagery.py", "image_preprocessing.py", "features/spectral.py", "training_matrix.py")]
    payload = {"manifest_sha256": file_sha256(manifest_path), "identities": identities,
               "imagery": deepcopy(config["imagery"]), "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
               "columns": columns, "code": {p.name: file_sha256(p) for p in [*code, Path(__file__)]},
               "numpy_version": np.__version__, "pandas_version": pd.__version__}
    return manifest, payload


def validate_matrix(frame: pd.DataFrame, manifest: pd.DataFrame, config: dict, columns: list[str]) -> None:
    target = config["training"]["lithium_column"]
    group = config["training"].get("spatial_group_column", "spatial_group")
    if list(frame.columns) != columns or len(frame) != len(manifest):
        raise ValueError("Spectral matrix schema/count mismatch")
    if not np.array_equal(frame[target].to_numpy(float), manifest[target].to_numpy(float)):
        raise ValueError("Spectral matrix target order mismatch")
    if not np.array_equal(frame[group].astype(str), manifest[group].astype(str)):
        raise ValueError("Spectral matrix group order mismatch")
    if not np.isfinite(frame.drop(columns=[target, group]).to_numpy(float)).all():
        raise ValueError("Nonfinite spectral predictor")


def read_cache(path: Path, manifest: pd.DataFrame, config: dict, identity: dict) -> pd.DataFrame | None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["signature"] != canonical_json_hash(identity) or meta["identity"] != identity:
            return None
        if meta["matrix_sha256"] != file_sha256(path):
            return None
        frame = pd.read_csv(path)
        validate_matrix(frame, manifest, config, identity["columns"])
        return frame
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_cache(path: Path, frame: pd.DataFrame, manifest: pd.DataFrame, config: dict, identity: dict) -> None:
    validate_matrix(frame, manifest, config, identity["columns"])
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = uuid4().hex
    tmp = path.with_name(f".{path.name}.{unique}.tmp")
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_tmp = meta_path.with_name(f".{meta_path.name}.{unique}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        record = {"schema_version": 2, "signature": canonical_json_hash(identity),
                  "identity": identity, "matrix_sha256": file_sha256(tmp)}
        meta_tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        meta_tmp.replace(meta_path)
    finally:
        tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
