"""Read-only validation of an existing expanded M2 cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256


def require_expanded_manifest(training: dict, root: Path) -> Path:
    root = Path(root).resolve()
    settings = training["expansion"]
    path = (root / settings["output_manifest_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Expanded manifest outside workspace")
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Expanded manifest unavailable: {path}; no fallback")
    digest = file_sha256(path)
    target = training.get("lithium_column", "Li_icpms")
    group = training.get("spatial_group_column", "spatial_group")
    sample = training.get("sample_id_column", "sample_id")
    frame = pd.read_csv(path, dtype={sample: str, group: str, "image_path": str})
    required = {target, group, sample, "image_path"}
    if frame.empty or not required <= set(frame):
        raise ValueError("Expanded manifest lacks required observations")
    values = pd.to_numeric(frame[target], errors="coerce")
    if not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError("Expanded manifest has invalid concentrations")
    for name in (sample, group, "image_path"):
        if frame[name].isna().any() or frame[name].str.strip().eq("").any():
            raise ValueError("Expanded manifest lacks row identity")
    if frame[sample].duplicated().any():
        raise ValueError("Expanded manifest has duplicate sample IDs")
    if "state_dir" in settings:
        evidence = (root / settings["state_dir"] / "expansion_summary.json").resolve()
        if not evidence.is_relative_to(root):
            raise ValueError("Expansion evidence outside workspace")
        if not evidence.is_file():
            raise FileNotFoundError("Expanded manifest evidence unavailable")
        summary = json.loads(evidence.read_text(encoding="utf-8"))
        if summary.get("new_manifest_sha256") != digest or summary.get("total") != len(frame):
            raise ValueError("Expanded manifest evidence mismatch")
        expected = training.get("frozen_manifest_sha256")
        if expected and summary.get("legacy_sha256") != expected:
            raise ValueError("Expanded manifest does not match configured frozen cohort")
    if file_sha256(path) != digest:
        raise ValueError("Expanded manifest changed during validation")
    return path
