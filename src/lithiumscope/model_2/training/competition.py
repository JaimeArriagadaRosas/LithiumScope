from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.reproducibility import set_global_seed
from lithiumscope.core.run_context import RunContext
from lithiumscope.core.training_diagnostics import save_oof_diagnostics
from lithiumscope.core.run_resume import build_training_signature
from lithiumscope.core.states import RunState
from lithiumscope.datasets.manifest import build_tabular_manifest, write_manifest
from lithiumscope.model_2.evaluation.baseline import evaluate_prior_baseline
from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold
from lithiumscope.model_2.evaluation.maps import save_spatial_score_map
from lithiumscope.model_2.evaluation.plots import (
    save_competition_chart,
    save_probability_histogram,
    save_roc_pr,
)
from lithiumscope.model_2.pipeline import load_model_2_training_frame
from lithiumscope.model_2.steps.step_05_spectral_features import (
    FEATURE_EXTRACTOR_VERSION,
)
from lithiumscope.model_2.schema import feature_range_profile
from lithiumscope.model_2.training.cv_runner import run_classification_cv
from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.training.postprocess import finalize_reports
from lithiumscope.model_2.training.scientific_validation import (
    make_scientific_folds, make_anchored_scientific_folds,
    scientific_labels, evaluate_scientific_baseline,
)
from lithiumscope.model_2.training.factory import create_model, get_label
from lithiumscope.persistence.save_model import save_model_bundle
from lithiumscope.results.dashboard import build_dashboard
from lithiumscope.results.excel_exporter import export_workbook
from lithiumscope.runtime.graceful_shutdown import get_shutdown_manager

logger = get_logger("model_2.competition")


@dataclass
class CompetitionOutcome:
    run_dir: Path
    ranking: pd.DataFrame
    winner: str | None
    successful: list[str]
    failed: list[str]


def _ranking_row(result) -> dict:
    folds = result.fold_table
    return {
        "algorithm": result.algorithm,
        "label": result.label,
        "roc_auc_mean": float(folds["roc_auc"].mean()),
        "roc_auc_std": float(folds["roc_auc"].std(ddof=0)),
        "average_precision_mean": float(folds["average_precision"].mean()),
        "balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
        "f1_mean": float(folds["f1"].mean()),
        "precision_mean": float(folds["precision"].mean()),
        "recall_mean": float(folds["recall"].mean()),
        "brier_score_mean": float(folds["brier_score"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
        "is_baseline": False,
    }


def _excel_sheets(
    ranking: pd.DataFrame,
    results: dict[str, object],
    y: pd.Series,
    baseline,
    evidence_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {
        "ranking": ranking,
        "baseline_folds": baseline[1],
    }
    for algorithm, result in results.items():
        sheets[f"folds_{algorithm}"[:31]] = result.fold_table
        sheets[f"pred_{algorithm}"[:31]] = pd.DataFrame(
            {
                "y_true": y.to_numpy(),
                "prospectivity_score": result.probabilities,
                "predicted_class_0_5": (result.probabilities >= 0.5).astype(int),
            }
        )
    if evidence_dir is not None:
        for filename,sheet in (
            ("old_release_protocol_comparison_706.csv","release_706"),
            ("old_release_protocol_intervals_706.csv","release_intervals"),
            ("scientific_stratified_metrics.csv","scientific_strata"),
            ("three_zone_competition_cases.csv","three_zone_cases"),
        ):
            evidence_path=evidence_dir/filename
            if evidence_path.exists():
                sheets[sheet]=pd.read_csv(evidence_path)
    return sheets


def run_model_2_competition(
    manifest_path: Path,
    device: DeviceInfo,
) -> CompetitionOutcome:
    cfg = load_config("model_2")
    app_cfg = load_config("app")
    algorithms = list(cfg["model"]["algorithms"])
    seed = int(cfg["model"]["random_seed"])
    set_global_seed(
        seed,
        deterministic_torch=bool(
            app_cfg.get("reproducibility", {}).get("deterministic_torch", False)
        ),
    )

    lithium_column = str(cfg["training"]["lithium_column"])
    spatial_group_column = str(
        cfg["training"].get("spatial_group_column", "spatial_group")
    )
    requested_folds = int(cfg["validation"].get("folds", 5))
    q = float(cfg["model"]["high_lithium_quantile"])
    fixed_threshold = cfg["model"].get("fixed_threshold_ppm")
    optimization_cfg = cfg.get("optimization", {})
    inner_folds = int(cfg["validation"].get("inner_folds", 3))

    signature = build_training_signature(
        "model_2",
        manifest_path,
        ("app", "model_2"),
    )
    context = RunContext.create(
        "model_2",
        training_signature=signature,
        resume=True,
    )

    print("\n=== MODELO 2 · COMPETENCIA DE ALGORITMOS ===")
    print(f"Entrenamiento: {context.run_id}")

    if context.resumed and context.tracker.state == RunState.COMPLETED:
        ranking_path = context.tables / "competition_ranking.csv"
        ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
        summary = context.tracker.payload.get("summary", {})
        winner = summary.get("winner")
        successful = (
            ranking[
                (ranking.get("status") == "ok")
                & (ranking.get("is_baseline") != True)  # noqa: E712
            ]["algorithm"].astype(str).tolist()
            if not ranking.empty
            else []
        )
        print("↻ Entrenamiento compatible ya completado; se reutiliza sin regenerar modelos.")
        print(f"Resultados: {context.root}")
        return CompetitionOutcome(context.root, ranking, winner, successful, [])

    if context.resumed:
        print(f"↻ Reanudando entrenamiento compatible: {context.run_id}")
        context.tracker.event(
            "run_resumed",
            previous_state=context.tracker.state.value,
        )
    context.tracker.set_state(RunState.RUNNING)
    get_shutdown_manager().register_cleanup(context.tracker.cancel_if_active)

    frame = load_model_2_training_frame(manifest_path)
    # Full-data target threshold is reserved for the final fitted model.
    # Evaluation labels are defined independently in each outer training fold.
    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else float(frame[lithium_column].quantile(q))
    )
    groups = (
        frame[spatial_group_column]
        if spatial_group_column in frame.columns
        else None
    )
    x = frame.drop(
        columns=[
            column
            for column in (lithium_column, spatial_group_column)
            if column in frame.columns
        ]
    )
    quality_cfg=cfg.get("quality_training", {})
    quality_mask=None
    quality_audit={"enabled": False, "manifest_rows":len(frame),
                   "training_eligible":len(frame),"training_excluded":0}
    if bool(quality_cfg.get("enabled",False)):
        inventory=PROJECT_ROOT / str(quality_cfg["inventory_path"])
        quality_mask,quality_audit=training_quality_mask(
            manifest_path,inventory_path=inventory,
            minimum_joint_valid=float(
                quality_cfg.get("minimum_joint_valid_fraction",0.95)),
        )
        if len(quality_mask)!=len(frame):
            raise ValueError("Model 2 QA mask must align with all feature rows")
        (context.manifests/"train_only_quality_policy.json").write_text(
            json.dumps(quality_audit,indent=2,ensure_ascii=False),
            encoding="utf-8")
        print(
            "Calidad científica SOLO TRAIN: "
            f"{quality_audit['training_eligible']} / {len(frame)} "
            "observaciones elegibles, sin eliminar ninguna observación TEST.",
            flush=True
        )
    anchor = cfg["validation"].get("anchor_legacy", {})
    legacy_manifest = PROJECT_ROOT / str(cfg["training"]["manifest_path"])
    extended_manifest = manifest_path.resolve() != legacy_manifest.resolve()
    if extended_manifest and bool(anchor.get("enabled", False)):
        old_sha = str(cfg["training"]["frozen_manifest_sha256"])
        if file_sha256(legacy_manifest) != old_sha:
            raise RuntimeError("M2 frozen historical reference dataset changed")
        base = pd.read_csv(legacy_manifest)
        expanded = pd.read_csv(manifest_path)
        baseline_count = int(cfg["training"]["frozen_expected_rows"])
        if (len(base) != baseline_count
                or len(expanded) < baseline_count
                or not np.array_equal(
                    base.sample_id.to_numpy(),
                    expanded.sample_id.iloc[:baseline_count].to_numpy(),
                )
                or not np.allclose(
                    base.Li_icpms.to_numpy(dtype=float),
                    expanded.Li_icpms.iloc[:baseline_count].to_numpy(dtype=float),
                )
                or not np.array_equal(
                    base.spatial_group.to_numpy(),
                    expanded.spatial_group.iloc[:baseline_count].to_numpy(),
                )):
            raise RuntimeError("M2 expanded manifest altered historical sample rows")
        reference_file = PROJECT_ROOT / str(anchor["folds_file"])
        if file_sha256(reference_file) != str(anchor["folds_sha256"]):
            raise RuntimeError("M2 anchored baseline fold snapshot changed")
        historic_splits = json.loads(reference_file.read_text(encoding="utf-8"))
        scientific_folds = make_anchored_scientific_folds(
            frame[lithium_column], groups, historic_splits,
            baseline_count=baseline_count, quantile=q,
            fixed_threshold_ppm=fixed_threshold,
        )
        print(
            f"M2: evaluacion pareada con {baseline_count} muestras "
            "historicas en sus mismos folds espaciales."
        )
    else:
        scientific_folds = make_scientific_folds(
            frame[lithium_column], groups,
            n_splits=requested_folds, random_state=seed,
            quantile=q, fixed_threshold_ppm=fixed_threshold,
        )
    y = pd.Series(scientific_labels(scientific_folds, len(frame)), index=frame.index)
    y_full = (frame[lithium_column] >= threshold).astype(int)
    fold_ids = pd.Series(-1, index=frame.index, dtype=int)
    for fold_number, split in enumerate(scientific_folds, start=1):
        fold_ids.iloc[split.test_idx] = fold_number
    (context.manifests / "scientific_folds.json").write_text(
        __import__("json").dumps(
            [{
                "fold": i, "n_train": len(f.train_idx),
                "n_test": len(f.test_idx),
                "label_threshold_ppm": f.label_threshold_ppm,
                "n_groups_train": f.n_groups_train,
                "n_groups_test": f.n_groups_test,
                "train_indices": f.train_idx.tolist(),
                "test_indices": f.test_idx.tolist(),
            } for i, f in enumerate(scientific_folds, 1)],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset_manifest = build_tabular_manifest(
        manifest_path,
        dataset_name="model_2_training_manifest",
        model_group="model_2",
        stage="spectral_feature_training_input",
        frame=frame,
        metadata={
            "threshold_ppm": threshold,
            "target_quantile": (
                None
                if fixed_threshold is not None
                else q
            ),
            "threshold_source": (
                "fixed_threshold_ppm"
                if fixed_threshold is not None
                else "dataset_quantile_final_fit_only"
            ),
            "cv_label_source": "outer_train_quantile_only",
            "cv_split": "group_kfold_strict",
            "spatial_groups": int(groups.nunique()) if groups is not None else 0,
        },
    )
    dataset_manifest_path = write_manifest(
        dataset_manifest,
        context.manifests / "dataset_manifest.json",
    )
    context.tracker.attach_summary(dataset_sha256=dataset_manifest.source_sha256)

    folds = len(scientific_folds)
    threshold_source = (
        "fijo"
        if fixed_threshold is not None
        else f"cuantil {q:.2f}"
    )
    print(
        f"Muestras: {len(frame)} | Umbral Li alto: "
        f"{threshold:.4f} ppm ({threshold_source}) | "
        f"Folds: {folds}"
    )
    if groups is not None and groups.nunique() >= folds:
        print(
            f"Validación espacial: {groups.nunique()} grupos "
            f"| inner folds: {inner_folds}"
        )

    baseline = evaluate_scientific_baseline(scientific_folds, len(frame))
    baseline[1].to_csv(
        context.tables / "baseline_fold_metrics.csv",
        index=False,
    )

    rows: list[dict] = [baseline[0]]
    results: dict[str, object] = {}
    failed: list[str] = []

    for index, algorithm in enumerate(algorithms, start=1):
        print(f"\n[{index}/{len(algorithms)}] Entrenando {get_label(algorithm)}")
        context.tracker.event("algorithm_started", algorithm=algorithm)

        try:
            result = run_classification_cv(
                x,
                y,
                algorithm,
                device,
                seed,
                folds,
                groups,
                checkpoint_root=context.checkpoints,
                optimization=optimization_cfg,
                inner_folds=inner_folds,
                scientific_folds=scientific_folds,
                lithium_values=frame[lithium_column].to_numpy(dtype=float),
                label_quantile=q,
                fixed_threshold_ppm=fixed_threshold,
                quality_training_mask=quality_mask,
            )
            results[algorithm] = result
            rows.append(_ranking_row(result))

            result.fold_table.to_csv(
                context.tables / f"fold_metrics_{algorithm}.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "y_true": y,
                    "li_ppm": frame[lithium_column].to_numpy(dtype=float),
                    "outer_fold": fold_ids.to_numpy(),
                    "prospectivity_score": result.probabilities,
                    "predicted_class_0_5": (
                        result.probabilities >= 0.5
                    ).astype(int),
                }
            ).to_csv(
                context.tables / f"oof_predictions_{algorithm}.csv",
                index=False,
            )

            save_probability_histogram(
                y,
                result.probabilities,
                context.figures / algorithm / "score_distribution.png",
                f"{result.label} — Distribución de scores",
            )
            save_roc_pr(
                y,
                result.probabilities,
                context.figures / algorithm / "roc_pr.png",
                result.label,
            )
            save_spatial_score_map(
                manifest_path,
                result.probabilities,
                context.figures / algorithm / "spatial_scores.png",
                f"{result.label} — scores espaciales OOF",
            )
            context.tracker.event(
                "algorithm_completed",
                algorithm=algorithm,
                metrics=result.overall_metrics,
            )
        except Exception as exc:
            logger.exception("Model 2 algorithm failed: %s", algorithm)
            failed.append(algorithm)
            rows.append(
                {
                    "algorithm": algorithm,
                    "label": get_label(algorithm),
                    "status": "failed",
                    "is_baseline": False,
                    "error": str(exc),
                }
            )
            context.tracker.event(
                "algorithm_failed",
                algorithm=algorithm,
                error=str(exc),
            )
            print(f"  ✗ {get_label(algorithm)} falló: {exc}")

    ranking = pd.DataFrame(rows)
    candidates = ranking[
        (ranking["status"] == "ok")
        & (ranking["is_baseline"] != True)  # noqa: E712
    ].copy()
    baseline_rows = ranking[ranking["is_baseline"] == True].copy()  # noqa: E712
    failed_rows = ranking[ranking["status"] != "ok"].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            [
                "roc_auc_mean",
                "average_precision_mean",
                "balanced_accuracy_mean",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
    if not baseline_rows.empty:
        baseline_rows.insert(0, "rank", pd.NA)

    ranking = pd.concat(
        [candidates, baseline_rows, failed_rows],
        ignore_index=True,
        sort=False,
    )
    if results:
        # All three reports are repairable from saved OOF without retraining.
        # Preserve a long-running competition even when one report needs repair,
        # but never hide the reason or mark the report as successful.
        evidence_status=finalize_reports(
            manifest_path,context.tables,context.tables,
        )
        per_report=evidence_status.get("reports",{})
        context.tracker.event(
            "scientific_postprocessing_complete",
            status=evidence_status["state"],
            report_statuses={
                name:details.get("state")
                for name,details in per_report.items()
            },
            recovery_command=evidence_status.get("recovery_command"),
        )
        triple=per_report.get("three_zone",{})
        if triple.get("state")=="complete":
            arms=triple["result"]["summary"]["arms"]
            ranking["three_zone_top1_accuracy"]=ranking["algorithm"].map({
                name:float(value["top1_accuracy"])
                for name,value in arms.items()
            })
            print("Priorizacion de tres zonas por algoritmo:",flush=True)
            for name,value in arms.items():
                print(f"  {name}: {value['top1_accuracy']:.2%}",flush=True)
        if evidence_status["state"]!="complete":
            print(
                "ATENCION: evidencias complementarias incompletas; "
                "los modelos y checkpoints quedan conservados. "
                "Consultar evidence_report_status.json para recuperar "
                "SIN reentrenar.",flush=True
            )
        else:
            print("Reportes del release previo, QA cientifica y tres zonas: OK",
                  flush=True)
    ranking.to_csv(
        context.tables / "competition_ranking.csv",
        index=False,
    )

    if candidates.empty:
        export_workbook(
            context.exports / "model_2_results.xlsx",
            _excel_sheets(ranking, results, y, baseline, context.tables),
        )
        build_dashboard(context.root, "LithiumScope — Modelo 2")
        context.tracker.set_state(
            RunState.FAILED,
            failure="No trainable algorithm completed successfully.",
        )
        return CompetitionOutcome(context.root, ranking, None, [], failed)

    save_competition_chart(
        candidates,
        context.figures / "competition_roc_auc.png",
    )

    winner = str(candidates.iloc[0]["algorithm"])
    print(f"\n🏆 Ganador de esta ejecución Modelo 2: {get_label(winner)}")

    winner_result = results[winner]
    diagnostic = save_oof_diagnostics(
        context.root / "scientific_diagnostics", y.to_numpy(),
        winner_result.probabilities, groups=groups, kind="classification",
        seed=seed,
    )
    context.tracker.attach_summary(scientific_diagnostics=diagnostic["metrics"])
    threshold_cfg = cfg.get("decision_threshold", {})
    threshold_objective = str(
        threshold_cfg.get("objective", "balanced_accuracy")
    )
    threshold_fallback = float(
        threshold_cfg.get("fallback", 0.5)
    )
    # Threshold is a deployable candidate only. Its OOF objective is optimistic
    # because model selection and threshold choice reuse the same OOF pool.
    threshold_selection = select_operating_threshold(
        y.to_numpy(),
        winner_result.probabilities,
        objective=threshold_objective,
        fallback=threshold_fallback,
    )
    operating_threshold = float(
        threshold_selection.threshold
    )
    print(
        "Umbral operativo Modelo 2: "
        f"{operating_threshold:.4f} "
        f"| {threshold_objective} OOF="
        f"{threshold_selection.objective_value:.4f}"
    )
    pd.DataFrame(
        [
            {
                "threshold": operating_threshold,
                "objective": threshold_selection.objective,
                "objective_value": threshold_selection.objective_value,
                "candidate_count": threshold_selection.candidate_count,
                **{
                    f"metric_{key}": value
                    for key, value in threshold_selection.metrics.items()
                },
            }
        ]
    ).to_csv(
        context.tables / "winner_operating_threshold.csv",
        index=False,
    )

    final_fit_idx=(
        np.flatnonzero(quality_mask)
        if quality_mask is not None else np.arange(len(x),dtype=int)
    )
    x_final=x.iloc[final_fit_idx]
    y_final=y_full.iloc[final_fit_idx]
    groups_final=groups.iloc[final_fit_idx] if groups is not None else None
    if y_final.nunique()!=2:
        raise ValueError("M2 quality TRAIN has only one class")
    final_calibration_cv = None
    if winner in {"svm_rbf", "random_forest_spatial_calibrated"}:
        from sklearn.model_selection import GroupKFold

        if groups_final is None or groups_final.nunique() < inner_folds:
            raise ValueError("No spatial groups for final calibration.")
        final_calibration_cv = list(GroupKFold(n_splits=inner_folds).split(
            x_final, groups=groups_final,
        ))
        for train_idx, valid_idx in final_calibration_cv:
            if (
                y_final.iloc[train_idx].nunique() < 2
                or y_final.iloc[valid_idx].nunique() < 2
            ):
                raise ValueError(
                    "Final calibration requires both classes per spatial fold."
                )

    final_estimator = create_model(
        winner,
        device,
        seed,
        winner_result.final_params,
        calibrated=True,
        calibration_cv=final_calibration_cv,
    )
    final_estimator.fit(x_final, y_final)
    profile = feature_range_profile(x_final)

    baseline_row = baseline[0]
    roc_gain = float(
        candidates.iloc[0]["roc_auc_mean"]
        - baseline_row["roc_auc_mean"]
    )
    ap_gain = float(
        candidates.iloc[0]["average_precision_mean"]
        - baseline_row["average_precision_mean"]
    )
    gate = cfg.get("release_gate", {})
    min_roc_gain = float(
        gate.get("min_roc_auc_gain_over_baseline", 0.0)
    )
    min_ap_gain = float(
        gate.get(
            "min_average_precision_gain_over_baseline",
            0.0,
        )
    )
    # OOF selected both the winning candidate and an operating threshold.
    # It cannot certify a release without a separate, unconsulted external holdout.
    exploratory_gate_pass = (
        roc_gain >= min_roc_gain and ap_gain >= min_ap_gain
    )
    release_gate_pass = False
    print(
        "Release gate Modelo 2: "
        + ("CANDIDATO / NO VERIFICADO" if exploratory_gate_pass else "FAIL")
        + (
            f" | ΔROC-AUC={roc_gain:+.4f} "
            f"(mín {min_roc_gain:+.4f})"
            f" | ΔAP={ap_gain:+.4f} "
            f"(mín {min_ap_gain:+.4f})"
        )
    )

    metadata = {
        "model_group": "model_2",
        "algorithm": winner,
        "label": get_label(winner),
        "samples": len(frame),
        "training_samples_after_quality": len(x_final),
        "train_only_quality": quality_audit,
        "experimental_only": True,
        "eligible_for_individual_release": False,
        "provisional_georoc_sample_count": max(0,len(frame)-706),
        "threshold_ppm": threshold,
        "target_quantile": (
            None if fixed_threshold is not None else q
        ),
        "threshold_source": (
            "fixed_threshold_ppm"
            if fixed_threshold is not None
            else "dataset_quantile_final_fit_only"
        ),
        "metrics": winner_result.overall_metrics,
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "operating_threshold": operating_threshold,
        "operating_threshold_objective": threshold_selection.objective,
        "operating_threshold_objective_value": threshold_selection.objective_value,
        "operating_threshold_metrics": threshold_selection.metrics,
        "competition_primary_metric": "roc_auc_mean",
        "winner_rule": (
            "highest mean ROC-AUC; tie-break Average Precision then "
            "Balanced Accuracy"
        ),
        "ranking": candidates.to_dict(orient="records"),
        "run_id": context.run_id,
        "device": device.accelerator,
        "dataset_sha256": dataset_manifest.source_sha256,
        "release_gate": {
            "pass": release_gate_pass,
            "exploratory_gate_pass": exploratory_gate_pass,
            "reason": "no_independent_external_holdout",
            "roc_auc_gain_over_baseline": roc_gain,
            "average_precision_gain_over_baseline": ap_gain,
            "min_roc_auc_gain_over_baseline": min_roc_gain,
            "min_average_precision_gain_over_baseline": min_ap_gain,
        },
        "cv_label_protocol": "group_kfold_fold_training_only",
        "operating_threshold_validity": "OOF_selection_exploratory_not_holdout",
        "scope_note": (
            "Score de priorización exploratoria; no es probabilidad de "
            "yacimiento económicamente viable."
        ),
    }

    bundle = {
        "estimator": final_estimator,
        "features": list(x.columns),
        "band_names": list(cfg["imagery"]["bands"]),
        "normalize_per_band": bool(
            cfg["imagery"].get("normalize_per_band", False)
        ),
        "threshold_ppm": threshold,
        "target_quantile": (
            None if fixed_threshold is not None else q
        ),
        "algorithm": winner,
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "operating_threshold": operating_threshold,
        "operating_threshold_objective": threshold_selection.objective,
        "operating_threshold_metrics": threshold_selection.metrics,
        "applicability_profile": profile,
        "experimental_only": True,
        "quality_min_joint_valid": (
            float(quality_cfg.get("minimum_joint_valid_fraction",0.95))
            if quality_mask is not None else None
        ),
        "spectral_representation": "legacy_DN_median_126",
    }
    model_path, metadata_path = save_model_bundle(
        "model_2",
        f"winner_{winner}",
        bundle,
        metadata,
        run_id=context.run_id,
        dataset_manifest_path=dataset_manifest_path,
        feature_schema={"features": list(x.columns)},
        config_name="model_2",
        activate=False,
    )

    export_workbook(
        context.exports / "model_2_results.xlsx",
        _excel_sheets(ranking, results, y, baseline, context.tables),
    )
    (context.exports / "winner.txt").write_text(
        f"algorithm={winner}\nmodel={model_path}\nmetadata={metadata_path}\n",
        encoding="utf-8",
    )

    build_dashboard(
        context.root,
        "LithiumScope — Modelo 2 · Competencia",
    )

    final_state = RunState.PARTIAL if failed else RunState.COMPLETED
    context.tracker.set_state(
        final_state,
        winner=winner,
        primary_metric="roc_auc_mean",
        primary_metric_value=float(candidates.iloc[0]["roc_auc_mean"]),
        model_path=str(model_path),
        failed_algorithms=failed,
        release_gate_pass=release_gate_pass,
        exploratory_gate_pass=exploratory_gate_pass,
        release_gate_reason="no_independent_external_holdout",
        roc_auc_gain_over_baseline=roc_gain,
        average_precision_gain_over_baseline=ap_gain,
        operating_threshold=operating_threshold,
        operating_threshold_objective=threshold_selection.objective,
        operating_threshold_objective_value=threshold_selection.objective_value,
    )
    return CompetitionOutcome(
        context.root,
        ranking,
        winner,
        list(results),
        failed,
    )
