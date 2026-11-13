"""Versioned train-only augmentation cache for frozen M2 spectral study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.model_2.evaluation.canonical_protocol import sha256
from lithiumscope.model_2.methodology.spectral_augmentation import (
    augmented_feature_record,
)


def patch_source_identity(manifest: pd.DataFrame) -> dict:
    """Verify source existence; fingerprint bytes, sample ID and ordering."""
    rows = []
    for item in manifest.itertuples(index=False):
        source = Path(str(item.image_path))
        if not source.is_file() or not source.stat().st_size:
            raise FileNotFoundError(f"Missing source patch {source}")
        rows.append([str(item.sample_id), str(source), sha256(source)])
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"rows": len(rows), "sha256": digest, "source_files": rows}


def build_augmentation_cache(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    folder: Path,
    *,
    protocol_sha: str,
) -> pd.DataFrame:
    """Return generated 126-D variants with immutable source fingerprint."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "augmented_spectral_features.csv"
    meta = folder / "augmented_feature_cache_manifest.json"
    provenance = patch_source_identity(manifest)
    payload = {
        "protocol_sha256": protocol_sha,
        "patch_set_sha256": provenance["sha256"],
        "rows": len(manifest),
        "columns": list(features.columns),
        "method": "experimental_correlated_DN_gain_up_to_3pct_band_0p7pct",
        "seed": 20261112,
        "created_independent_observations": False,
    }
    if target.exists() or meta.exists():
        if not (target.exists() and meta.exists()):
            raise ValueError("Partial augmented cache; remove it manually")
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old["contract"] != payload or old["cache_sha256"] != sha256(target):
            raise ValueError("Refusing mismatched augmented spectral cache")
        cached = pd.read_csv(target)
        if list(cached.columns) != list(features.columns) or cached.shape != features.shape:
            raise ValueError("Augmented cache schema mismatch")
        return cached

    rows = []
    for i, item in enumerate(manifest.itertuples(index=False)):
        patch = np.load(str(item.image_path), allow_pickle=False)
        augmented = augmented_feature_record(patch, sample_id=str(item.sample_id))
        if list(augmented) != list(features.columns):
            raise ValueError(f"Augmented features changed schema at row {i}")
        rows.append(augmented)
        if (i + 1) % 150 == 0:
            print(f"AUGMENTED_PATCH_FEATURES {i + 1}/{len(manifest)}", flush=True)
    cache = pd.DataFrame.from_records(rows, columns=features.columns)
    if cache.shape != features.shape or not np.isfinite(cache.to_numpy(float)).all():
        raise ValueError("Invalid augmented spectral feature matrix")
    cache.to_csv(target, index=False)
    meta.write_text(
        json.dumps({"contract": payload, "cache_sha256": sha256(target)},
                   indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cache
