"""Frozen and expanded M2 source contracts; no acquisition or reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from .model_2_paths import inspect_image_links


def _read_manifest(path: Path, training: dict, *, require_sample: bool,
                   expected_rows: int | None = None, expected_hash: str = ""):
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"M2 manifest unavailable: {path}; no automatic replacement")
    digest = file_sha256(path)
    if expected_hash and digest != expected_hash:
        raise RuntimeError("The Model 2 training manifest changed. Aborting to avoid unintended new labels/images.")
    target = training.get("lithium_column", "Li_icpms")
    group = training.get("spatial_group_column", "spatial_group")
    sample = training.get("sample_id_column", "sample_id")
    frame = pd.read_csv(path, dtype={sample: str, group: str, "image_path": str})
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(f"Model 2 expected {expected_rows} spectral samples, got {len(frame)}.")
    required = {target, group, "image_path"} | ({sample} if require_sample else set())
    if frame.empty or not required <= set(frame):
        raise ValueError("M2 manifest lacks essential columns or observations.")
    values = pd.to_numeric(frame[target], errors="coerce")
    if not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError("M2 manifest has invalid concentrations.")
    for name in (group, "image_path", *([sample] if sample in frame else [])):
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError("M2 manifest has missing row identity.")
    if sample in frame and frame[sample].astype(str).duplicated().any():
        raise ValueError("M2 manifest has duplicate sample IDs.")
    return frame, digest


def _workspace_path(root: Path, raw: str | Path) -> Path:
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError("M2 manifest outside workspace")
    return path


def _finish(path, frame, digest, root, *, verify_images):
    if verify_images:
        links = inspect_image_links(frame["image_path"], root=root, verify_hash=True)
        if links["unavailable"]:
            raise ValueError(f"M2 manifest has {links['unavailable']} unavailable or invalid image references")
    if file_sha256(path) != digest:
        raise ValueError("M2 manifest changed during validation")
    get_logger("datasets.management.sources.model_2").info(
        "Verified M2 manifest %s rows=%d image_verification=%s", path, len(frame), verify_images)
    return path


def require_frozen_manifest(training: dict, root: Path, *, verify_images: bool = False) -> Path:
    root = Path(root).resolve()
    path = _workspace_path(root, training["manifest_path"])
    frame, digest = _read_manifest(path, training, require_sample=False,
        expected_rows=int(training.get("frozen_expected_rows", 706)),
        expected_hash=str(training.get("frozen_manifest_sha256", "")).strip())
    return _finish(path, frame, digest, root, verify_images=verify_images)


def require_expanded_manifest(training: dict, root: Path, *, verify_images: bool = False) -> Path:
    root = Path(root).resolve()
    settings = training["expansion"]
    path = _workspace_path(root, settings["output_manifest_path"])
    frame, digest = _read_manifest(path, training, require_sample=True)
    if "state_dir" in settings:
        evidence = _workspace_path(root, Path(settings["state_dir"]) / "expansion_summary.json")
        if not evidence.is_file():
            raise FileNotFoundError("Expanded manifest evidence unavailable")
        summary = json.loads(evidence.read_text(encoding="utf-8"))
        if summary.get("new_manifest_sha256") != digest or summary.get("total") != len(frame):
            raise ValueError("Expanded manifest evidence mismatch")
        expected = training.get("frozen_manifest_sha256")
        if expected and summary.get("legacy_sha256") != expected:
            raise ValueError("Expanded manifest does not match configured frozen cohort")
    return _finish(path, frame, digest, root, verify_images=verify_images)
