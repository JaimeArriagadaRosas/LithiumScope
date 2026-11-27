"""Read-only contract for a frozen M2 cohort, independent of model orchestration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256


def require_frozen_manifest(training: dict, root: Path) -> Path:
    root = root.resolve()
    raw = Path(training["manifest_path"])
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Frozen manifest outside workspace")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen Model 2 manifest missing: {path}. No automatic re-download is allowed.")
    digest = file_sha256(path)
    expected = str(training.get("frozen_manifest_sha256", "")).strip()
    if expected and digest != expected:
        raise RuntimeError("The Model 2 training manifest changed. Aborting to avoid unintended new labels/images.")
    frame = pd.read_csv(path)
    expected_rows = int(training.get("frozen_expected_rows", 706))
    if len(frame) != expected_rows:
        raise RuntimeError(f"Model 2 expected {expected_rows} spectral samples, got {len(frame)}.")
    target = training.get("lithium_column", "Li_icpms")
    group = training.get("spatial_group_column", "spatial_group")
    sample = training.get("sample_id_column", "sample_id")
    required = {target, group, "image_path"}
    if not required <= set(frame):
        raise ValueError("Frozen Model 2 manifest lacks essential columns.")
    values = pd.to_numeric(frame[target], errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Frozen Model 2 manifest has invalid concentrations.")
    for name in (group, "image_path", *([sample] if sample in frame else [])):
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError("Frozen Model 2 manifest has missing row identity.")
    if sample in frame and frame[sample].astype(str).duplicated().any():
        raise ValueError("Frozen Model 2 manifest has duplicate sample IDs.")
    if file_sha256(path) != digest:
        raise ValueError("Frozen manifest changed during validation.")
    return path
