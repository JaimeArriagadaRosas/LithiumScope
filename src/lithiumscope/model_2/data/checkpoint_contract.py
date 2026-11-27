"""Strict new-cohort checkpoint reconciliation; frozen releases are read-only."""
from __future__ import annotations

import json
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path
from lithiumscope.model_2.preparation.patch_contract import make_patch_contract, load_verified_patch


def acquisition_signature(imagery):
    return canonical_json_hash({"version": 3, **imagery})


def source_signature(source_sha, imagery, training):
    return canonical_json_hash({"source": source_sha, "acquisition": acquisition_signature(imagery),
                                "group_degrees": training.get("spatial_group_degrees", .5),
                                "zero_policy": training.get("target_zero_policy", "retain"),
                                "sample_policy": "source_reconciliation_v2"})


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def reconcile_checkpoint(checkpoint, samples, imagery, *, root=None):
    """Return only current source rows with validated, compatible cached patches."""
    contract = make_patch_contract(bands=imagery["bands"], pixels=imagery["patch_pixels"],
                                   size_m=imagery["patch_size_m"])
    expected = acquisition_signature(imagery)
    if not samples.sample_id.is_unique:
        raise ValueError("Current sample IDs must be unique")
    current = samples.set_index("sample_id")
    result = []
    rejected = []
    duplicate_ids = set(checkpoint.loc[checkpoint.sample_id.duplicated(keep=False), "sample_id"])
    for row in checkpoint.to_dict("records"):
        sid = row["sample_id"]
        try:
            if sid in duplicate_ids or sid not in current.index:
                raise ValueError("Obsolete or ambiguous sample")
            sample = current.loc[sid]
            path = resolve_image_path(str(row["image_path"]), root=root)
            raw, meta, _ = load_verified_patch(path, contract)
            from lithiumscope.tools.acquisition.sentinel2.registered_patch import verify_registered_mask
            verify_registered_mask(path, meta)
            if (meta.get("acquisition_signature") != expected
                    or meta.get("sample_id") != sid
                    or meta.get("scene_id") != row["sentinel_scene_id"]
                    or not np.allclose([meta["latitude"], meta["longitude"]],
                                       [sample.latitude, sample.longitude], rtol=0, atol=1e-8)
                    or not np.isfinite(raw).any(axis=(1, 2)).all()):
                raise ValueError("Patch identity or acquisition policy changed")
            updated = {**row, "Li_icpms": float(sample.Li_icpms),
                       "latitude": float(sample.latitude), "longitude": float(sample.longitude),
                       "spatial_group": str(sample.spatial_group)}
            result.append(updated)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rejected.append({"sample_id": str(sid), "reason": type(exc).__name__, "detail": str(exc)})
    return pd.DataFrame(result, columns=checkpoint.columns), rejected


def archive_failures(path):
    if path.is_file():
        archive = path.with_name(f"{path.stem}.history.{file_sha256(path)[:20]}{path.suffix}")
        if not archive.exists():
            atomic_text(archive, path.read_text(encoding="utf-8"))
