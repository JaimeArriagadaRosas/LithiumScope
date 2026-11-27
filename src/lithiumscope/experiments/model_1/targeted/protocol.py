"""Verify the previous competition cohort, folds and sealed hyperparameters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.checkpoints import read_checkpoint_evidence
from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.diagnostic import prepare_diagnostic_source
from lithiumscope.experiments.model_1.targeted.partitions import outer_from_manifest, find_plan
from lithiumscope.experiments.generalization.prepared import load_snapshot, prepare_version
from lithiumscope.experiments.model_1.targeted.policies import ALGORITHMS


def available_runs():
    root = RESULTS_DIR / "model_1" / "runs"
    candidates = []
    for path in root.glob("training_*"):
        try:
            record = json.loads((path/"run.json").read_text(encoding="utf-8"))
            if record.get("state") != "completed":
                continue
            if not all((path/"tables"/f"oof_predictions_{a}__identity.csv").is_file()
                       and (path/"checkpoints"/f"{a}__identity"/"fold_01.json").is_file()
                       for a in ALGORITHMS):
                continue
            candidates.append(path.name)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(candidates, reverse=True)


def load_protocol(run_id):
    if not run_id or Path(run_id).name != run_id or any(c in run_id for c in ("/", chr(92), ":", "..")):
        raise ValueError("Invalid run ID")
    run = RESULTS_DIR/"model_1"/"runs"/run_id
    record = json.loads((run/"run.json").read_text(encoding="utf-8"))
    if record.get("state") != "completed":
        raise ValueError("Reference competition not completed")
    source = prepare_diagnostic_source("model_1")
    digest = file_sha256(source)
    if record["summary"]["dataset_sha256"] != digest:
        raise ValueError("Source differs from reference competition")
    cfg = load_config("model_1")
    snapshot, _ = prepare_version("model_1", source, run_eda=False)
    data = load_snapshot(snapshot, snapshot.name)
    x, y, groups, sources, eligible, _, _ = data
    if not eligible.all():
        raise ValueError("Original M1 cohort must be preserved")
    manifest_path = run/"tables"/"outer_fold_manifest.csv"
    saved = pd.read_csv(manifest_path, dtype={"row_index":str}).sort_values("row_position")
    if (not np.array_equal(saved.row_position, np.arange(len(x)))
            or not np.array_equal(saved.row_index.to_numpy(str), x.index.to_numpy(str))
            or not np.array_equal(saved.spatial_group.to_numpy(str), groups.to_numpy(str))
            or not np.array_equal(saved.source_dataset.to_numpy(str), sources.to_numpy(str))
            or not np.allclose(saved[["Longitude","Latitude"]], x[["Longitude","Latitude"]], rtol=0, atol=1e-8)):
        raise ValueError("Reference cohort or spatial folds changed")
    fold_ids = saved.outer_fold.to_numpy(int)
    outer = outer_from_manifest(fold_ids, groups, int(cfg["validation"]["outer_folds"]))
    splits, partition_evidence = find_plan(snapshot, digest, outer, groups, int(cfg["validation"]["inner_folds"]))
    params, references, hashes = {}, {}, {"outer_fold_manifest.csv": file_sha256(manifest_path)}
    for algorithm in ALGORITHMS:
        variant = algorithm+"__identity"
        path = run/"tables"/f"oof_predictions_{variant}.csv"
        table = pd.read_csv(path, dtype={"row_index":str}).sort_values("row_position")
        if (not np.array_equal(table.row_position, np.arange(len(x)))
                or not np.array_equal(table.row_index.to_numpy(str), x.index.to_numpy(str))
                or not np.array_equal(table.outer_fold, fold_ids)
                or not np.array_equal(table.spatial_group.to_numpy(str), groups.to_numpy(str))
                or not np.allclose(table.Li_icpms_real, y, rtol=0, atol=1e-8)
                or not np.isfinite(table.Li_icpms_predicted).all()):
            raise ValueError("Archived OOF does not match frozen cohort")
        hashes[path.name] = file_sha256(path)
        references[algorithm] = table.Li_icpms_predicted.to_numpy(float)
        for fold, (_, test, _) in enumerate(splits, 1):
            ck = run/"checkpoints"/variant/f"fold_{fold:02d}.json"
            saved_fold = read_checkpoint_evidence(ck)
            if (saved_fold["fold"] != fold or saved_fold["test_indices"] != test.tolist()
                    or not np.allclose(saved_fold["predictions"], references[algorithm][test], rtol=0, atol=1e-8)
                    or "treatment_policy" not in saved_fold["params"]):
                raise ValueError("Checkpoint and OOF differ or treatment absent")
            params[algorithm, fold] = saved_fold["params"]
            hashes[f"{variant}/fold_{fold:02d}.json"] = file_sha256(ck)
    evidence = {**partition_evidence,"reference_run":run_id, "source_sha256":digest, "snapshot":snapshot.name,
                "reference_artifact_sha256":hashes, "rows":len(x), "groups":int(groups.nunique()),
                "seed":int(cfg["validation"]["random_seed"])}
    return data, splits, params, references, evidence
