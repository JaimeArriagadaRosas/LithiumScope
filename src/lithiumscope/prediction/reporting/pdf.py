from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

from lithiumscope.prediction.reporting.context import ReportContext
from lithiumscope.prediction.reporting.sections.cases import build_cases
from lithiumscope.prediction.reporting.sections.figures import build_figures
from lithiumscope.prediction.reporting.sections.notes import build_notes
from lithiumscope.prediction.reporting.sections.overview import build_overview
from lithiumscope.prediction.reporting.sections.performance import build_performance
from lithiumscope.prediction.reporting.sections.spatial import build_spatial
from lithiumscope.prediction.reporting.styles import (
    build_styles,
    page_footer,
)


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
    model_1_predictions: pd.DataFrame,
    model_2_predictions: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
    overlap_audit: dict | None,
    input_info: dict | None,
    model_1_diagnostics: dict | None,
    model_2_diagnostics: dict | None,
    maps: list[Path],
    figures: list[Path],
    lithium_threshold_ppm: float | None = None,
    model_2_classification_threshold: float | None = None,
    runtime: dict | None = None,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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
        runtime=dict(runtime or {}),
    )
    styles = build_styles()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.6 * cm,
        title=title,
        author="LithiumScope",
        subject=(
            "Prediccion integrada y demostracion "
            "cientifica reproducible"
        ),
    )

    story: list = []
    for builder in (
        build_spatial,
        build_overview,
        build_performance,
        build_figures,
        build_cases,
        build_notes,
    ):
        story.extend(
            builder(
                context,
                styles,
            )
        )

    document.build(
        story,
        onFirstPage=page_footer,
        onLaterPages=page_footer,
    )
    return path
