"""Re-extract red-edge features from exact local scenes; never use network."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.core.hashing import file_sha256
from lithiumscope.tools.acquisition.sentinel2.rededge import cached_rededge_sample


def verify_rededge_cohort(manifest_path: Path, matrix_path: Path, quality_path: Path) -> dict:
    manifest, matrix, quality = map(pd.read_csv, (manifest_path,matrix_path,quality_path))
    if not (len(manifest)==len(matrix)==len(quality)) or manifest.sample_id.duplicated().any():
        raise ValueError("Red-edge cohort identity is not unique/aligned")
    if manifest.sample_id.astype(str).tolist() != quality.sample_id.astype(str).tolist():
        raise ValueError("Red-edge QA sample order mismatch")
    cache = PROJECT_ROOT/"results/scientific_simulations/m2_rededge_v4_20261113/patches"
    expected_columns = list(matrix.columns[-44:])
    records = []
    max_abs_difference = 0.0
    for i, (sample, qa) in enumerate(zip(manifest.itertuples(index=False), quality.to_dict("records"), strict=True)):
        features, actual = cached_rededge_sample(sample, output=cache, allow_acquisition=False)
        if list(features) != expected_columns:
            raise ValueError(f"Red-edge feature schema/order mismatch at row {i}")
        for key in ("sample_id","scene_id","rededge_patch_sha256","original_patch_sha256","scene_band_hash"):
            if str(actual[key]) != str(qa[key]):
                raise ValueError(f"Red-edge provenance mismatch {key}: {sample.sample_id}")
        if not np.isclose(actual["rededge_joint_valid_fraction"],qa["rededge_joint_valid_fraction"],rtol=0,atol=1e-12):
            raise ValueError(f"Red-edge QA coverage mismatch: {sample.sample_id}")
        reconstructed = np.array(list(features.values()),dtype=float)
        stored = matrix.iloc[i,-44:].to_numpy(float)
        if not np.allclose(reconstructed,stored,rtol=1e-12,atol=5e-13):
            raise ValueError(f"Red-edge feature values differ: {sample.sample_id}")
        max_abs_difference = max(max_abs_difference,float(np.max(np.abs(reconstructed-stored))))
        token = hashlib.sha256(str(sample.sample_id).encode()).hexdigest()[:22]
        records.append({**actual,"sidecar_sha256":file_sha256(cache/f"{token}.json")})
    return {"sample_count":len(records),"feature_count":44,"maximum_absolute_difference":max_abs_difference,
            "verified_local_only":True,"source_identity_sha256":canonical_json_hash(records)}
