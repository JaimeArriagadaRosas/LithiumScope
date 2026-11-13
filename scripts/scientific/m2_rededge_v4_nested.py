"""Isolated complete six-algorithm nested Optuna competition on real 170-D M2.

No interactive menu, release, or active-model changes. Existing QA and CV
policies are reused without retraining the 126-D baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from lithiumscope.core.config import load_config
from lithiumscope.core.device import detect_device
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.training.cv_runner import run_classification_cv
from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.training.scientific_validation import scientific_labels

from scripts.scientific.m2_rededge_v4_competition import (
    ARMS,
    RUNS,
    _splits,
)
from scripts.scientific.m2_rededge_v4_report import write_release_comparison


def _score(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def train_all(cohort: int, *, root: Path = PROJECT_ROOT):
    if cohort not in RUNS:
        raise ValueError("Only original 706 and expanded 783 cohorts supported")
    output = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    folder = output / f"cohort_{cohort}"
    matrix_file = folder / "training_v4_170.csv"
    status_file = folder / "build_status.json"
    if not matrix_file.is_file() or not status_file.is_file():
        raise FileNotFoundError("Cannot train without all actual red-edge images")
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or (file_sha256(matrix_file) != status.get("matrix_sha256")):
        raise ValueError("Cannot train incomplete or changed imagery matrix")
    cfg = load_config("model_2")
    if tuple(cfg["model"]["algorithms"]) != ARMS:
        raise ValueError("All six M2 algorithm families must be retained")
    mpath = root / (
        "data/processed/model_2/training_manifest.csv"
        if cohort == 706
        else "data/processed/model_2/vnext_expanded/training_manifest.csv"
    )
    if status["manifest_sha256"] != file_sha256(mpath):
        raise ValueError("Frozen cohort fingerprint changed")
    manifest = pd.read_csv(mpath)
    frame = pd.read_csv(matrix_file)
    if (
        len(frame) != cohort
        or len(frame.columns) != 172
        or not np.array_equal(
            frame.Li_icpms.to_numpy(float),
            manifest.Li_icpms.to_numpy(float),
        )
        or not np.array_equal(
            frame.spatial_group.astype(str),
            manifest.spatial_group.astype(str),
        )
    ):
        raise ValueError("M2 v4 feature count, labels, or group mismatch")
    x = frame.drop(columns=["Li_icpms", "spatial_group"])
    if not np.isfinite(x.to_numpy(float)).all():
        raise ValueError("Nonfinite v4 features; no silent imputation")
    quality, _ = training_quality_mask(
        mpath,
        inventory_path=root / str(cfg["quality_training"]["inventory_path"]),
        minimum_joint_valid=float(cfg["quality_training"]["minimum_joint_valid_fraction"]),
    )
    from lithiumscope.model_2.methodology.rededge_training import (
        combine_quality_masks,
    )

    quality, v4_quality_audit = combine_quality_masks(
        quality,
        folder / "rededge_quality.csv",
        manifest,
    )
    print("V4_QA95", v4_quality_audit, flush=True)
    folds = _splits(cohort, manifest)
    labels = scientific_labels(folds, cohort)
    y = pd.Series(labels, index=manifest.index)
    prior = root / "results/model_2/runs" / RUNS[cohort] / "tables"
    for alg in ARMS:
        old = pd.read_csv(prior / f"oof_predictions_{alg}.csv")
        if len(old) != cohort or not np.array_equal(old.y_true.to_numpy(int), labels):
            raise ValueError("Frozen 126-D baseline label mismatch")
    out = output / f"nested_170_{cohort}"
    out.mkdir(parents=True, exist_ok=True)
    active = root / "models/active_models.json"
    previous_registry_sha = file_sha256(active)
    contract = {
        "cohort": cohort,
        "full_nested_search": True,
        "source_matrix_sha256": file_sha256(matrix_file),
        "manifest_sha256": file_sha256(mpath),
        "cfg_sha256": file_sha256(root / "config/model_2.yaml"),
        "previous_release_registry_sha256": previous_registry_sha,
        "six_algorithm_families": list(ARMS),
        "folds": 5,
        "inner_folds": 3,
        "model1_chemistry_as_features": False,
        "source": RUNS[cohort],
        "experiment_only": True,
    }
    contract_path = out / "experiment_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing nested experiment uses different settings")
    else:
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    device = detect_device(prefer_gpu=False)
    rows = []
    for alg in ARMS:
        print("NESTED_M2_V4", alg, "START", flush=True)
        result = run_classification_cv(
            x,
            y,
            alg,
            device,
            int(cfg["model"]["random_seed"]),
            5,
            groups=manifest.spatial_group,
            checkpoint_root=out / "checkpoints",
            optimization=cfg["optimization"],
            inner_folds=int(cfg["validation"]["inner_folds"]),
            scientific_folds=folds,
            lithium_values=manifest.Li_icpms,
            label_quantile=float(cfg["model"]["high_lithium_quantile"]),
            quality_training_mask=quality,
        )
        predictions = np.asarray(result.probabilities, float)
        if len(predictions) != cohort or not np.isfinite(predictions).all():
            raise ValueError("Full M2 v4 nested OOF incomplete")
        np.save(out / f"oof_{alg}.npy", predictions, allow_pickle=False)
        result.fold_table.to_csv(out / f"fold_metrics_{alg}.csv", index=False)
        (out / f"final_params_{alg}.json").write_text(
            json.dumps(result.final_params, indent=2),
            encoding="utf-8",
        )
        new = _score(labels, predictions)
        old = _score(labels, pd.read_csv(prior / f"oof_predictions_{alg}.csv").prospectivity_score)
        rows.append(
            {
                "algorithm": alg,
                **{f"v4_{k}": v for k, v in new.items()},
                **{f"v3_{k}": v for k, v in old.items()},
                "delta_auc": new["roc_auc"] - old["roc_auc"],
                "delta_ap": new["ap"] - old["ap"],
                "delta_brier": new["brier"] - old["brier"],
                "roc_auc_mean": float(result.fold_table.roc_auc.mean()),
                "roc_auc_std": float(result.fold_table.roc_auc.std(ddof=0)),
            }
        )
        pd.DataFrame(rows).to_csv(out / "partial_six_algorithm_ranking.csv", index=False)
        print("NESTED_M2_V4_DONE", alg, new, flush=True)
    result = (
        pd.DataFrame(rows)
        .sort_values(
            ["roc_auc_mean", "v4_ap"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    result.to_csv(out / "six_algorithm_nested_v4_vs_v3.csv", index=False)
    report = None
    if cohort == 706:
        report = write_release_comparison(
            out,
            root / "results/model_2/runs" / RUNS[cohort],
            ARMS,
            manifest,
        )
    if file_sha256(active) != previous_registry_sha:
        raise ValueError("Active-model registry changed during nested experiment")
    (out / "final_status.json").write_text(
        json.dumps(
            {
                "status": "complete_experimental_no_release",
                "trained_algorithms": list(ARMS),
                "compared_against_reconstructed_release": report is not None,
                "release_comparison": report,
                "source_matrix_sha256": contract["source_matrix_sha256"],
                "no_independent_external_holdout": True,
                "none_activated": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("NESTED_M2_V4_ALL_SIX_COMPLETE", result.to_string(index=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=int, choices=RUNS, default=706)
    args = parser.parse_args()
    train_all(args.cohort)


if __name__ == "__main__":
    main()
