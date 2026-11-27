"""Publish integrated predictions and coordinate independent reporting services."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.results.atomic_exports import write_table, write_text, write_legacy_evidence_json as write_json
from lithiumscope.prediction.execution.session import PredictionSession
from lithiumscope.prediction.execution.contracts import PredictionFrames, PredictionEvidence, PredictionThresholds

logger = get_logger("prediction.outputs")


def _write_outputs(session: PredictionSession, frames: PredictionFrames, evidence: PredictionEvidence,
                   thresholds: PredictionThresholds, run_log_path: Path) -> tuple[Path, Path, Path]:
    status = session.root / "publication_status.json"
    write_json(status, {"state": "running", "run_id": session.run_id})
    try:
        result = _publish_outputs(session, frames, evidence, thresholds, run_log_path)
    except Exception as exc:
        logger.exception("Prediction artifact publication failed run=%s", session.run_id)
        write_json(status, {"state": "failed", "run_id": session.run_id, "reason": str(exc)})
        raise
    write_json(status, {"state": "complete", "run_id": session.run_id})
    return result


def _publish_outputs(session, frames, evidence, thresholds, run_log_path):
    from lithiumscope.reporting.figures.maps import save_spatial_maps
    from lithiumscope.reporting.figures.charts import save_integrated_figures
    from lithiumscope.reporting.figures.satellite import save_satellite_input_preview
    from lithiumscope.reporting.service import report_title, write_pdf_report
    from lithiumscope.reporting.workbook import write_workbook

    def relative(path):
        return Path(path).resolve().relative_to(session.root.resolve()).as_posix()

    model_1_predictions = frames.model_1_predictions
    model_2_predictions = frames.model_2_predictions
    model_2_cases = frames.model_2_cases
    paired = frames.paired
    correlations = frames.correlations
    concordance = frames.concordance
    training_vs_external = frames.training_vs_external
    model_1_metrics = evidence.model_1_metrics
    model_2_metrics = evidence.model_2_metrics
    model_1_identity = evidence.model_1_identity
    model_2_identity = evidence.model_2_identity
    model_1_diagnostics = evidence.model_1_diagnostics
    model_2_diagnostics = evidence.model_2_diagnostics
    input_info = evidence.input_info
    overlap_audit = evidence.overlap_audit
    interpretation = evidence.interpretation
    lithium_threshold = thresholds.lithium_threshold
    probability_threshold = thresholds.probability_threshold
    model_2_predicted_positive_count = thresholds.model_2_predicted_positive_count
    logger.info("Publishing prediction artifacts run=%s", session.run_id)
    write_table(session.model_1 / "predictions.csv", model_1_predictions)
    write_table(session.model_2 / "predictions.csv", model_2_predictions)
    write_table(session.cross_model / "joined_cases.csv", paired)
    write_table(session.cross_model / "correlations.csv", correlations)
    write_table(session.cross_model / "concordance.csv", concordance)
    write_table(session.root / "training_vs_external.csv", training_vs_external)
    write_json(session.model_1 / "diagnostics.json", model_1_diagnostics)
    write_json(session.model_2 / "diagnostics.json", model_2_diagnostics)

    interpretation_path = session.root / "interpretation.txt"
    write_text(interpretation_path, interpretation + "\n")
    maps = save_spatial_maps(
        paired,
        session.figures / "maps",
    )
    satellite_preview = save_satellite_input_preview(
        model_2_cases,
        session.figures / "sentinel_inputs.png",
        bands=model_2_identity.get("bands"),
    )
    figures = save_integrated_figures(paired, session.figures)
    if satellite_preview is not None:
        figures = [satellite_preview, *figures]

    workbook_path = write_workbook(
        session.root / "evaluation.xlsx",
        model_1=model_1_predictions,
        model_2=model_2_predictions,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        training_vs_external=training_vs_external,
    )
    runtime = runtime_fingerprint()
    if session.mode == "demonstration":
        from lithiumscope.prediction.analysis.model_selection_evidence import (
            gather_selection_evidence,
        )

        selection_evidence = gather_selection_evidence(
            model_1_identity, model_2_identity
        )
        from lithiumscope.reporting.figures.eda_figure_snapshot import snapshot_eda_figures
        from lithiumscope.core.paths import RESULTS_DIR
        figures = [*figures, *snapshot_eda_figures(selection_evidence, session.figures, root=RESULTS_DIR.parent)]
        write_json(session.root / "model_selection_evidence.json", selection_evidence)
        runtime["model_selection_evidence"] = selection_evidence
    title = report_title(session.mode, model_1_identity, model_2_identity)
    report_path = write_pdf_report(
        session.root / "report.pdf",
        title=title,
        run_id=session.run_id,
        mode=session.mode,
        interpretation=interpretation,
        model_1_identity=model_1_identity,
        model_2_identity=model_2_identity,
        model_1_metrics=model_1_metrics,
        model_2_metrics=model_2_metrics,
        model_1_predictions=model_1_predictions,
        model_2_predictions=model_2_predictions,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        training_vs_external=training_vs_external,
        overlap_audit=overlap_audit,
        input_info=input_info,
        model_1_diagnostics=model_1_diagnostics,
        model_2_diagnostics=model_2_diagnostics,
        maps=maps,
        figures=figures,
        lithium_threshold_ppm=lithium_threshold,
        model_2_classification_threshold=probability_threshold,
        runtime=runtime,
    )

    manifest = {
        "schema_version": 2,
        "state": "complete",
        "title": title,
        "run_id": session.run_id,
        "mode": session.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "model_1": model_1_identity,
            "model_2": model_2_identity,
        },
        "inputs": input_info,
        "metrics": {
            "model_1_external": model_1_metrics,
            "model_2_external": model_2_metrics,
        },
        "overlap_audit": overlap_audit,
        "counts": {
            "model_1_cases": len(model_1_predictions),
            "model_2_cases": len(model_2_predictions),
            "paired_cases": len(paired),
            "model_2_predicted_positive": model_2_predicted_positive_count,
        },
        "thresholds": {
            "lithium_reference_ppm": lithium_threshold,
            "model_2_classification_score": probability_threshold,
        },
        "runtime": runtime,
        "artifacts": {
            "model_1_predictions": relative(session.model_1 / "predictions.csv"),
            "model_2_predictions": relative(session.model_2 / "predictions.csv"),
            "joined_cases": relative(session.cross_model / "joined_cases.csv"),
            "correlations": relative(session.cross_model / "correlations.csv"),
            "concordance": relative(session.cross_model / "concordance.csv"),
            "training_vs_external": relative(session.root / "training_vs_external.csv"),
            "workbook": relative(workbook_path),
            "report_pdf": relative(report_path),
            "interpretation": relative(interpretation_path),
            "run_log": str(run_log_path),
            "maps": [relative(path) for path in maps],
            "figures": [relative(path) for path in figures],
            "satellite_input_preview": (
                relative(satellite_preview)
                if satellite_preview is not None
                else None
            ),
        },
    }
    manifest["artifact_sha256"] = {
        key: file_sha256(session.root / value)
        for key, value in manifest["artifacts"].items()
        if key != "run_log" and isinstance(value, str) and (session.root / value).is_file()
    }
    manifest_path = write_json(session.root / "prediction_manifest.json", manifest)
    logger.info("Prediction artifacts published run=%s manifest=%s", session.run_id, manifest_path)
    return report_path, workbook_path, manifest_path
