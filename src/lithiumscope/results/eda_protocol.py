"""Identity and acceptance of EDA evidence bound to actual outer development folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.core.scientific_checks import assert_group_isolation, assert_partition_coverage
from lithiumscope.results.eda_artifacts import verify_eda


def fold_protocol(x, y, groups, splits, *, eligible=None, endpoints=None, strategy="spatial"):
    if groups is None or not x.index.equals(y.index) or not x.index.equals(groups.index):
        raise ValueError("EDA row identity mismatch")
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Missing EDA spatial group")
    if not np.isfinite(np.asarray(y, dtype=float)).all():
        raise ValueError("EDA target must be finite")
    splits = list(splits)
    assert_partition_coverage(splits, len(x))
    mask = np.ones(len(x), dtype=bool) if eligible is None else np.asarray(eligible)
    if mask.shape != (len(x),) or mask.dtype.kind != "b":
        raise ValueError("EDA eligibility must be aligned boolean values")
    if endpoints is not None:
        if len(endpoints) != len(splits) or not np.isfinite(endpoints).all():
            raise ValueError("EDA endpoint count or values differ")
        endpoints = [float(v) for v in endpoints]
    records = []
    for fold, (train, test) in enumerate(splits, 1):
        train, test = np.asarray(train), np.asarray(test)
        assert_group_isolation(train, test, groups)
        fit = train[mask[train]]
        if not len(fit):
            raise ValueError("EDA has empty eligible TRAIN")
        record = {"fold": fold, "train": train.tolist(), "test": test.tolist(),
                  "eligible_train": fit.tolist()}
        if endpoints is not None:
            labels = (y.iloc[fit].to_numpy(float) >= endpoints[fold - 1]).astype(int)
            if len(np.unique(labels)) != 2:
                raise ValueError("EDA eligible TRAIN contains only one class")
            record["label_threshold_ppm"] = endpoints[fold - 1]
        records.append(record)
    identity = {"strategy": strategy, "folds": records,
                "row_index": list(map(str, x.index)), "groups": groups.astype(str).tolist(),
                "eligibility": mask.tolist()}
    return {"schema_version": 1, "fold_count": len(records),
            "identity_sha256": canonical_json_hash(identity),
            "scope": "outer_train_descriptive_only", "fit_performed": False}


def attach_training_eda(directory: Path, context, protocol: dict) -> Path:
    """Reference exactly the inspected evidence, without consuming its statistics for fit."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not verify_eda(directory, manifest["signature"]):
        raise ValueError("EDA evidence failed integrity verification")
    if manifest.get("development_protocol") != protocol:
        raise ValueError("EDA evidence belongs to different development folds")
    required = {"tables/outer_train_summary.csv", "tables/outer_train_features.csv"}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("EDA TRAIN-only evidence is incomplete")
    summary = pd.read_csv(directory / "tables/outer_train_summary.csv")
    if list(summary["fold"]) != list(range(1, protocol["fold_count"] + 1)):
        raise ValueError("EDA fold coverage differs")
    target = context.manifests / "eda_evidence.json"
    payload = {"eda_run_id": directory.name, "manifest_sha256": file_sha256(manifest_path),
               "signature": manifest["signature"], "development_protocol": protocol,
               "scope": "reporting_and_preflight_only", "fitted_transform_reuse": False}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
