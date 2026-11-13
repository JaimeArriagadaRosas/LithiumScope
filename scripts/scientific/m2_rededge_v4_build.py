"""M2 v4 real imagery: separately cached 170-D dataset, resumable and strict.

Usage: python scripts/scientific/m2_rededge_v4_build.py --cohort 706 --limit 3
Then without --limit to complete. A partial dataset CANNOT train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.rededge_acquisition import (
    cached_rededge_sample,
)
from lithiumscope.model_2.methodology.rededge_v4 import (
    TOTAL_FEATURES,
    append_rededge_features,
)
from lithiumscope.model_2.pipeline import load_model_2_training_frame

BASE = PROJECT_ROOT / "results/scientific_simulations/m2_rededge_v4_20261113"


def build(cohort: int, *, limit: int | None = None, root: Path = PROJECT_ROOT) -> dict:
    if cohort not in (706, 783) or (limit is not None and limit < 1):
        raise ValueError("M2 v4 supports only the frozen 706 or expanded 783")
    manifest_path = root / (
        "data/processed/model_2/training_manifest.csv"
        if cohort == 706
        else "data/processed/model_2/vnext_expanded/training_manifest.csv"
    )
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != cohort or manifest.sample_id.duplicated().any():
        raise ValueError("M2 v4: selected original cohort invalid")
    frame = load_model_2_training_frame(manifest_path)
    if len(frame) != cohort or not np.array_equal(
        frame.Li_icpms.to_numpy(float),
        manifest.Li_icpms.to_numpy(float),
    ):
        raise ValueError("Legacy 126 features not aligned to manifest")
    non_features = ("Li_icpms", "spatial_group")
    original = frame.drop(columns=list(non_features))
    if len(original.columns) != 126 or not np.isfinite(original.to_numpy(float)).all():
        raise ValueError("Unexpected historical 126-feature schema")
    # The experiment never changes the M2 source or historical extractor.
    output = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    folder = output / f"cohort_{cohort}"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    quality = []
    failures = []
    subset = manifest if limit is None else manifest.iloc[:limit]
    for i, sample in enumerate(subset.itertuples(index=False)):
        try:
            extra, evidence = cached_rededge_sample(
                sample,
                output=output / "patches",
                min_joint_valid=0.05,
            )
            row = append_rededge_features(
                original.iloc[i].to_dict(),
                extra,
            )
            results.append(row)
            quality.append(evidence)
        except Exception as exc:
            failures.append(
                {
                    "row_index": i,
                    "sample_id": str(sample.sample_id),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:380],
                }
            )
            print("RED_EDGE_FAILURE", i, str(sample.sample_id), str(exc)[:190], flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(subset):
            print(
                "RED_EDGE_PROGRESS",
                i + 1,
                "/",
                len(subset),
                "successful",
                len(quality),
                "failed",
                len(failures),
                flush=True,
            )
    audit = pd.DataFrame(quality)
    audit.to_csv(folder / "rededge_quality.csv", index=False)
    pd.DataFrame(
        failures,
        columns=[
            "row_index",
            "sample_id",
            "error_type",
            "error",
        ],
    ).to_csv(folder / "rededge_failures.csv", index=False)
    ready = len(results) == cohort and not failures and limit is None
    evidence = {
        "status": "complete" if ready else "incomplete_no_training",
        "cohort": cohort,
        "attempted": len(subset),
        "downloaded": len(results),
        "failed": len(failures),
        "manifest_sha256": file_sha256(manifest_path),
        "feature_extractor_version": 4,
        "feature_count": TOTAL_FEATURES,
        "preserved_original_features": 126,
        "new_real_rededge_features": 44,
        "schema": "historical_126_and_real_S2_L2A_B05_B06_B07_B8A_44",
        "synthetic_label_count": 0,
        "active_models_unchanged": True,
    }
    if ready:
        expanded = pd.DataFrame.from_records(results)
        if expanded.shape != (cohort, TOTAL_FEATURES):
            raise ValueError("M2 v4 schema drift or repeated row")
        assert list(expanded.columns[:126]) == list(original.columns)
        if not np.array_equal(expanded.iloc[:, :126].to_numpy(float), original.to_numpy(float)):
            raise ValueError("M2 v4 modified frozen 126-D feature values")
        # Do not overwrite legacy training caches, ever.
        matrix = expanded.copy()
        matrix.insert(0, "Li_icpms", frame.Li_icpms)
        matrix.insert(1, "spatial_group", frame.spatial_group)
        destination = folder / "training_v4_170.csv"
        tmp = destination.with_suffix(".tmp")
        matrix.to_csv(tmp, index=False)
        tmp.replace(destination)
        evidence["matrix_sha256"] = file_sha256(destination)
        evidence["new_band_coverage_median"] = float(audit.rededge_joint_valid_fraction.median())
        evidence["new_band_coverage_below95"] = int((audit.rededge_joint_valid_fraction < 0.95).sum())
    (folder / "build_status.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M2_RED_EDGE_BUILD_STATUS", json.dumps(evidence), flush=True)
    return evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=int, choices=(706, 783), default=706)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    outcome = build(args.cohort, limit=args.limit)
    # Partial pilot is deliberately not a process failure; it is never trainable.
    if args.limit is None and outcome["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
