"""Read-only provenance for the model-choice section in the original PDF.

All numeric results are recovered from real local training outputs. An entire
snapshot is embedded in prediction_manifest.json so rebuild_pdf is stable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT

M1_RUNS = (
    "training_20260925_111245_m0300",
    "training_20261105_204029_m0300",
    "training_20261106_111653_m0300",
)
M2_RUNS = (
    "training_20260920_195933_m0300",
    "training_20261105_223618_m0300",
    "training_20261106_135029_m0300",
    "training_20261111_170832_m0300",
    "training_20261112_123451_m0300",
    "training_20261113_114607_m0300",
    "training_20261113_133905_m0300",
    "training_20261113_200558_m0300",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_row(group: str, run_id: str, root: Path) -> dict:
    source = root / "models" / group / "trained" / run_id
    meta = _read_json(source / "metadata.json")
    status = _read_json(source / "candidate_status.json")
    ranking = root / "results" / group / "runs" / run_id / "tables" / "competition_ranking.csv"
    record = {
        "run_id": run_id,
        "algorithm": str(meta["algorithm"]),
        "samples": int(meta["samples"]),
        "status": status["status"],
        "metrics": dict(meta.get("metrics", {})),
        "feature_extractor_version": meta.get("feature_extractor_version"),
        "spectral_representation": meta.get("spectral_representation"),
        "training_samples_after_quality": meta.get("training_samples_after_quality"),
    }
    if ranking.is_file():
        df = pd.read_csv(ranking)
        win = df.loc[df.algorithm.astype(str).str.contains(
            str(meta["algorithm"]), case=False, regex=False
        ) & df.status.eq("ok")]
        if len(win):
            winner = win.iloc[0]
            for key in (
                "roc_auc_mean", "average_precision_mean", "brier_score_mean",
                "three_zone_top1_accuracy", "rmse_mean", "mae_mean", "r2_mean",
            ):
                if key in winner and pd.notna(winner[key]):
                    record[key] = float(winner[key])
    return record


def gather_selection_evidence(m1_identity: dict, m2_identity: dict,
                              *, root: Path = PROJECT_ROOT) -> dict:
    if m1_identity.get("run_id") != M1_RUNS[2]:
        raise ValueError("The planned demonstration must use expanded M1 SVM")
    if m2_identity.get("run_id") != M2_RUNS[0]:
        raise ValueError("The planned demonstration must use previous M2 release")
    m1 = [_selection_row("model_1", run_id, root) for run_id in M1_RUNS]
    m2 = [_selection_row("model_2", run_id, root) for run_id in M2_RUNS]
    paired = _read_json(
        root / "results/scientific_simulations/vnext_20261106"
        / "stage5_training_comparison/m1_paired_comparison.json"
    )
    v4_dir = root / "results/model_2/runs" / M2_RUNS[-1] / "tables"
    compare = pd.read_csv(v4_dir / "old_release_protocol_comparison_706.csv")
    intervals = pd.read_csv(v4_dir / "old_release_protocol_intervals_706.csv")
    selected = compare[compare.algorithm.isin((
        "original_release_protocol_retrained", "random_forest",
    ))]
    if len(selected) != 2:
        raise ValueError("M2 release-v4 paired comparison is incomplete")
    selected_ci = intervals[intervals.algorithm.eq("random_forest")]
    if len(selected_ci) != 1:
        raise ValueError("M2 bootstrap confidence intervals are missing")
    spectral = _read_json(
        root / "results/scientific_simulations/m2_rededge_v4_20261113"
        / "cohort_706/spectral_identity_summary.json"
    )
    v4_ranking = pd.read_csv(v4_dir / "competition_ranking.csv")
    six = v4_ranking[v4_ranking.status.eq("ok") &
                     v4_ranking.algorithm.ne("baseline_prior")]
    if len(six) != 6:
        raise ValueError("Six-model v4 ranking is incomplete")
    out = {
        "source": "training_artifacts_only_no_manual_metrics",
        "model_1_selected_run": m1_identity["run_id"],
        "model_2_selected_run": m2_identity["run_id"],
        "model_1_runs": m1,
        "model_2_runs": m2,
        "model_1_paired": {
            "matched": paired["comparison"]["matched"],
            "different_outer_fold": paired["comparison"]["different_outer_fold"],
            "common": paired["by_subset"]["all_shared"],
            "high_lithium": paired["by_subset"]["high_lithium_q75_common"],
            "bootstrap": paired["cluster_bootstrap"],
            "certified": paired["development_improvement_gate"]["release_certified"],
        },
        "model_2_same_706_comparison": selected.to_dict(orient="records"),
        "model_2_rf_delta_ci": selected_ci.iloc[0].to_dict(),
        "model_2_170_six_algorithms": six.to_dict(orient="records"),
        "model_2_spectral_identity": spectral,
        "model_2_reference_is_retrained_protocol_not_deployed_artifact": True,
        "model_2_external_independent_holdout": False,
        "v4_original_prediction_interface_has_44_missing_features": True,
        "selection_reason": {
            "model_1": "Pareado exploratorio 1004 casos: RMSE y MAE menores, con limites de folds cambiados y sensibilidad a Li extremo.",
            "model_2": "CatBoost publicado de 126 features mantiene inferencia operativa; v4 no demuestra superioridad estadistica y el scorer aun no usa sus 44 features adicionales.",
        },
        "user_release_approval_required": True,
    }
    return out
