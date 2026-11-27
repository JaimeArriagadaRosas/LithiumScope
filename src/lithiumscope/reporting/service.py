"""Prepare a report from supplied evidence; no fitting, prediction or acquisition."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from lithiumscope.reporting.context import ReportContext


def report_title(mode: str, *identities: dict) -> str:
    if mode != "demonstration":
        return "LithiumScope - Predicción completa"
    if any(identity.get("selection") == "experimental_demonstration" for identity in identities):
        return "LithiumScope - Demostración integrada experimental"
    return "LithiumScope - Demostración integrada automática"


def write_pdf_report(
    path: Path,
    *,
    title: str,
    run_id: str,
    mode: str,
    interpretation: str,
    model_1_identity: dict,
    model_2_identity: dict,
    model_1_metrics: dict,
    model_2_metrics: dict,
    model_1_predictions: Any,
    model_2_predictions: Any,
    paired: Any,
    correlations: Any,
    concordance: Any,
    training_vs_external: Any,
    overlap_audit: dict | None,
    input_info: dict | None,
    model_1_diagnostics: dict | None,
    model_2_diagnostics: dict | None,
    maps: list[Path],
    figures: list[Path],
    lithium_threshold_ppm: float | None = None,
    model_2_classification_threshold: float | None = None,
    runtime: dict | None = None,
    decision_previews: dict[str, Path] | None = None,
) -> Path:
    from lithiumscope.core.scientific.decision import calculate_decision
    from lithiumscope.core.scientific.report_context import report_statistics

    from lithiumscope.core.scientific.report_context import prepare_cases, prepare_selection_evidence
    model_1_predictions, model_2_predictions, paired = prepare_cases(
        model_1_predictions, model_2_predictions, paired, model_2_classification_threshold)
    runtime = dict(runtime or {})
    if "model_selection_evidence" in runtime:
        runtime["model_selection_evidence"] = prepare_selection_evidence(runtime["model_selection_evidence"])
    decision = calculate_decision(paired)
    statistics = report_statistics(model_1_predictions, model_2_predictions, concordance,
                                   model_2_classification_threshold, model_1_diagnostics)
    from lithiumscope.reporting.figures.decision_previews import prepare_decision_previews
    if decision_previews is None:
        decision_previews = prepare_decision_previews(
            decision.selected, model_2_identity, path.parent / "decision_previews"
        )
    from lithiumscope.reporting.pdf import write_pdf_report as render_pdf
    context = ReportContext(
        title=title,
        run_id=run_id,
        mode=mode,
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
        maps=tuple(maps),
        figures=tuple(figures),
        lithium_threshold_ppm=lithium_threshold_ppm,
        model_2_classification_threshold=(
            model_2_classification_threshold
        ),
        runtime=runtime,
        decision=decision,
        statistics=statistics,
        decision_previews=decision_previews,
    )

    return render_pdf(path, context)
