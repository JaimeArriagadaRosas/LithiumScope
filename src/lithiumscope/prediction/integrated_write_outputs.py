"""Focused integrated prediction phase; stable original public entrypoint."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lithiumscope.prediction.integrated import Path, PredictionSession, pd

def _write_outputs(
    *,
    session: PredictionSession,
    model_1_predictions: pd.DataFrame,
    model_2_predictions: pd.DataFrame,
    model_2_cases: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
    model_1_metrics: dict,
    model_2_metrics: dict,
    model_1_identity: dict,
    model_2_identity: dict,
    model_1_diagnostics: dict,
    model_2_diagnostics: dict,
    input_info: dict,
    overlap_audit: dict | None,
    interpretation: str,
    run_log_path: Path,
    lithium_threshold: float,
    probability_threshold: float,
    model_2_predicted_positive_count: int,
) -> tuple[Path, Path, Path]:
    from lithiumscope.prediction.integrated import Path, PredictionSession, datetime, pd, runtime_fingerprint, save_integrated_figures, save_satellite_input_preview, save_spatial_maps, timezone, write_json, write_pdf_report, write_workbook
    model_1_predictions.to_csv(session.model_1 / "predictions.csv", index=False)
    model_2_predictions.to_csv(session.model_2 / "predictions.csv", index=False)
    paired.to_csv(session.cross_model / "joined_cases.csv", index=False)
    correlations.to_csv(session.cross_model / "correlations.csv", index=False)
    concordance.to_csv(session.cross_model / "concordance.csv", index=False)
    training_vs_external.to_csv(session.root / "training_vs_external.csv", index=False)
    write_json(session.model_1 / "diagnostics.json", model_1_diagnostics)
    write_json(session.model_2 / "diagnostics.json", model_2_diagnostics)

    interpretation_path = session.root / "interpretation.txt"
    interpretation_path.write_text(interpretation + "\n", encoding="utf-8")
    maps = save_spatial_maps(
        paired,
        session.figures / "maps",
    )
    satellite_preview = save_satellite_input_preview(
        model_2_cases,
        session.figures / "sentinel_inputs.png",
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
        from lithiumscope.prediction.reporting.model_selection_evidence import (
            gather_selection_evidence,
        )

        selection_evidence = gather_selection_evidence(
            model_1_identity, model_2_identity
        )
        write_json(session.root / "model_selection_evidence.json", selection_evidence)
        runtime["model_selection_evidence"] = selection_evidence
    report_path = write_pdf_report(
        session.root / "report.pdf",
        title=(
            "LithiumScope - Demostración integrada automática"
            if session.mode == "demonstration"
            else "LithiumScope - Predicción completa"
        ),
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
        "schema_version": 1,
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
            "model_1_predictions": str(session.model_1 / "predictions.csv"),
            "model_2_predictions": str(session.model_2 / "predictions.csv"),
            "joined_cases": str(session.cross_model / "joined_cases.csv"),
            "correlations": str(session.cross_model / "correlations.csv"),
            "concordance": str(session.cross_model / "concordance.csv"),
            "training_vs_external": str(session.root / "training_vs_external.csv"),
            "workbook": str(workbook_path),
            "report_pdf": str(report_path),
            "interpretation": str(interpretation_path),
            "run_log": str(run_log_path),
            "maps": [str(path) for path in maps],
            "figures": [str(path) for path in figures],
            "satellite_input_preview": (
                str(satellite_preview)
                if satellite_preview is not None
                else None
            ),
        },
    }
    manifest_path = write_json(session.root / "prediction_manifest.json", manifest)
    return report_path, workbook_path, manifest_path
