from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

from lithiumscope.core.logging.session import get_logger
from lithiumscope.reporting.context import ReportContext
from lithiumscope.reporting.sections.cases import build_cases
from lithiumscope.reporting.sections.figures import build_figures
from lithiumscope.reporting.sections.notes import build_notes
from lithiumscope.reporting.sections.model_selection import build_model_selection
from lithiumscope.reporting.sections.overview import build_overview
from lithiumscope.reporting.sections.performance import build_performance
from lithiumscope.reporting.sections.spatial import build_spatial
from lithiumscope.reporting.styles import (
    build_styles,
    page_footer,
)


def write_pdf_report(path: Path, context: ReportContext) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger("reporting.pdf")
    logger.info("PDF generation started run=%s path=%s", context.run_id, path)
    styles = build_styles()

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    document = SimpleDocTemplate(
        str(temporary),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.6 * cm,
        title=context.title,
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
        build_model_selection,
        build_figures,
        build_cases,
        build_notes,
    ):
        logger.debug("Building PDF section %s", builder.__name__)
        story.extend(
            builder(
                context,
                styles,
            )
        )

    try:
        document.build(
            story,
            onFirstPage=page_footer,
            onLaterPages=page_footer,
        )
        temporary.replace(path)
        logger.info("PDF generated run=%s path=%s", context.run_id, path)
    except Exception:
        logger.exception("PDF generation failed run=%s", context.run_id)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return path
