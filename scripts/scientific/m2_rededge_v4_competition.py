"""Six-family M2 170-D evaluation with historical fold hyperparameters frozen.

Not the interactive training entrypoint. Never promotes or writes a model.
Requires COMPLETE real four-band dataset; never manufactures spectra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from lithiumscope.core.device import detect_device
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.rededge_training import fit_frozen_model
from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.training.scientific_validation import (
    make_scientific_folds,
    make_anchored_scientific_folds,
)
from lithiumscope.model_2.methodology.rededge_v4 import TOTAL_FEATURES

ARMS = (
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
    "svm_rbf",
)
RUNS = {
    706: "training_20261113_133905_m0300",
    783: "training_20261113_114607_m0300",
}
OUTPUT = PROJECT_ROOT / "results/scientific_simulations/m2_rededge_v4_20261113"


def _metrics(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError("OOF score count/finite check failed")
    tn, fp, fn, tp = confusion_matrix(y, score >= 0.5, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "VP": int(tp),
        "VN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def _splits(cohort, manifest):
    if cohort == 706:
        folds = make_scientific_folds(
            manifest.Li_icpms,
            manifest.spatial_group,
            n_splits=5,
        )
    else:
        from lithiumscope.core.config import load_config

        cfg = load_config("model_2")
        frozen_path = PROJECT_ROOT / str(cfg["validation"]["anchor_legacy"]["folds_file"])
        folds = make_anchored_scientific_folds(
            manifest.Li_icpms,
            manifest.spatial_group,
            json.loads(frozen_path.read_text(encoding="utf-8")),
            baseline_count=706,
        )
    return folds


def compare(cohort: int, *, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    if cohort not in RUNS:
        raise ValueError("Unsupported experiment cohort")
    output = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    matrix_path = output / f"cohort_{cohort}/training_v4_170.csv"
    build_meta = output / f"cohort_{cohort}/build_status.json"
    if not matrix_path.is_file() or not build_meta.is_file():
        raise FileNotFoundError("Complete 170-feature red-edge matrix is required")
    status = json.loads(build_meta.read_text(encoding="utf-8"))
    if (
        status.get("status") != "complete"
        or status["cohort"] != cohort
        or file_sha256(matrix_path) != status["matrix_sha256"]
    ):
        raise ValueError("Never train on a partial or stale red-edge dataset")
    manifest_path = root / (
        "data/processed/model_2/training_manifest.csv"
        if cohort == 706
        else "data/processed/model_2/vnext_expanded/training_manifest.csv"
    )
    manifest = pd.read_csv(manifest_path)
    if file_sha256(manifest_path) != status["manifest_sha256"]:
        raise ValueError("M2 historical cohort manifest has drifted")
    features = pd.read_csv(matrix_path)
    if (
        len(features) != cohort
        or len(features.columns) != TOTAL_FEATURES + 2
        or not np.array_equal(
            features.Li_icpms.to_numpy(float),
            manifest.Li_icpms.to_numpy(float),
        )
        or not np.array_equal(
            features.spatial_group.astype(str),
            manifest.spatial_group.astype(str),
        )
    ):
        raise ValueError("v4 rows or label alignment incorrect")
    x = features.drop(columns=["Li_icpms", "spatial_group"])
    if not np.isfinite(x.to_numpy(float)).all():
        raise ValueError("M2 spectral v4 matrix has nonfinite values")
    qpath = root / (
        "results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv"
    )
    quality, audit = training_quality_mask(
        manifest_path,
        inventory_path=qpath,
        minimum_joint_valid=0.95,
    )
    from lithiumscope.model_2.methodology.rededge_training import (
        combine_quality_masks,
    )

    quality, v4_quality_audit = combine_quality_masks(
        quality,
        output / f"cohort_{cohort}/rededge_quality.csv",
        manifest,
    )
    print("V4_QA95", v4_quality_audit, flush=True)
    folds = _splits(cohort, manifest)
    prior = root / "results/model_2/runs" / RUNS[cohort]
    snapshots = {}
    for name in ARMS:
        old = pd.read_csv(prior / "tables" / f"oof_predictions_{name}.csv")
        if len(old) != cohort:
            raise ValueError("Previous baseline OOF missing or incomplete")
        snapshots[name] = old
    all_labels = np.full(cohort, -1, int)
    for k, fold in enumerate(folds, 1):
        all_labels[fold.test_idx] = fold.y_test
        for old in snapshots.values():
            if not np.array_equal(
                old.outer_fold.iloc[fold.test_idx].to_numpy(int),
                np.full(len(fold.test_idx), k, int),
            ) or not np.array_equal(
                old.y_true.iloc[fold.test_idx].to_numpy(int),
                fold.y_test,
            ):
                raise ValueError("Previous test folds or labels changed")
    if (all_labels < 0).any():
        raise ValueError("Outer fold coverage incomplete")
    manifest_active = root / "models/active_models.json"
    active_digest = file_sha256(manifest_active)
    device = detect_device(prefer_gpu=False)
    out = output / f"comparison_{cohort}"
    out.mkdir(parents=True, exist_ok=True)
    contract = {
        "cohort": cohort,
        "matrix_sha256": file_sha256(matrix_path),
        "manifest_sha256": file_sha256(manifest_path),
        "original_126_run": RUNS[cohort],
        "active_model_registry_sha256": active_digest,
        "algorithms": list(ARMS),
        "fixed_outer_fold_hyperparameters": True,
    }
    signature = out / "experiment_contract.json"
    if signature.exists():
        if json.loads(signature.read_text(encoding="utf-8")) != contract:
            raise ValueError("Cached fixed-v4 run uses another dataset or release")
    else:
        signature.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    rows = []
    for algorithm in ARMS:
        checkpoint = prior / "checkpoints" / algorithm
        scores = np.full(cohort, np.nan, float)
        apparent = []
        for k, fold in enumerate(folds, 1):
            dest = out / f"{algorithm}_fold_{k:02d}.csv"
            if dest.is_file():
                previous = pd.read_csv(dest)
                if len(previous) != len(fold.test_idx) or not np.array_equal(previous.test_idx, fold.test_idx):
                    raise ValueError("Saved v4 fold checkpoint indices drifted")
                scores[fold.test_idx] = previous.score.to_numpy(float)
                apparent.append(float(previous.train_auc_gap.iloc[0]))
                continue
            fit = fold.train_idx[quality[fold.train_idx]]
            groups = manifest.spatial_group.iloc[fit]
            ytrain = (manifest.Li_icpms.iloc[fit].to_numpy(float) >= fold.label_threshold_ppm).astype(int)
            if len(np.unique(ytrain)) != 2:
                raise ValueError("M2 v4 fit lacks a class")
            stored = json.loads((checkpoint / f"fold_{k:02d}.json").read_text(encoding="utf-8"))
            if not np.array_equal(stored["test_indices"], fold.test_idx):
                raise ValueError("Prior fold provenance changed")
            model = fit_frozen_model(
                algorithm,
                stored["params"],
                x.iloc[fit],
                ytrain,
                groups,
                device,
                42 + k,
            )
            s = np.asarray(model.predict_proba(x.iloc[fold.test_idx])[:, 1], float)
            apparent_score = np.asarray(model.predict_proba(x.iloc[fit])[:, 1], float)
            train_auc = roc_auc_score(ytrain, apparent_score)
            test_auc = roc_auc_score(fold.y_test, s)
            gap = float(train_auc - test_auc)
            if not np.isfinite(s).all() or not np.all((0 <= s) & (s <= 1)):
                raise ValueError("Invalid model probability scores")
            scores[fold.test_idx] = s
            apparent.append(gap)
            pd.DataFrame(
                {
                    "test_idx": fold.test_idx,
                    "score": s,
                    "train_auc_gap": gap,
                }
            ).to_csv(dest, index=False)
            print("M2_V4_FOLD", algorithm, k, "test_auc", round(test_auc, 5), "train_gap", round(gap, 5), flush=True)
        if not np.isfinite(scores).all():
            raise ValueError("Missing experimental outer predictions")
        np.save(out / f"oof_{algorithm}.npy", scores, allow_pickle=False)
        new = _metrics(all_labels, scores)
        old = _metrics(all_labels, snapshots[algorithm].prospectivity_score)
        rows.append(
            {
                "algorithm": algorithm,
                **{f"v4_{key}": v for key, v in new.items()},
                **{f"v3_{key}": v for key, v in old.items()},
                "delta_roc_auc": new["roc_auc"] - old["roc_auc"],
                "delta_ap": new["average_precision"] - old["average_precision"],
                "delta_brier": new["brier"] - old["brier"],
                "apparent_train_auc_gap_mean": float(np.mean(apparent)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "six_algorithm_170_vs_126.csv", index=False)
    release_evidence = None
    if cohort == 706:
        from scripts.scientific.m2_rededge_v4_report import write_release_comparison

        release_evidence = write_release_comparison(out, prior, ARMS, manifest)
    meta = {
        "status": "complete_experimental_development_only",
        "six_algorithms_kept": list(ARMS),
        "source_cohort": cohort,
        "fixed_hyperparameters_from_prior_outer_train": RUNS[cohort],
        "new_optuna_trials": 0,
        "source_matrix_sha256": file_sha256(matrix_path),
        "original_126_preserved": True,
        "features": TOTAL_FEATURES,
        "spatial_folds_reused": True,
        "new_release_activated": False,
        "all_5_outer_test_folds_evaluated": True,
        "source_quality_mask": audit,
        "no_independent_external_holdout": True,
        "release_comparison": release_evidence,
    }
    if file_sha256(manifest_active) != active_digest:
        raise ValueError("Active model registry changed during experiment")
    (out / "experiment_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(result[["algorithm", "v4_roc_auc", "v3_roc_auc", "delta_roc_auc"]].to_string(index=False))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=int, choices=RUNS, default=706)
    args = p.parse_args()
    compare(args.cohort)


if __name__ == "__main__":
    main()
