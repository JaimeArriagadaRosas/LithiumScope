from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Spacer

from lithiumscope.prediction.reporting.primitives import paragraph


FIGURE_EXPLANATIONS = {
    "li_real_vs_model_1": (
        "La diagonal discontinua es la predicción ideal (y=x); la "
        "recta continua resume la tendencia lineal ajustada a estos casos."
    ),
    "li_real_vs_model_2_score": (
        "La recta resume la asociación lineal observada entre Li medido "
        "y score M2. No implica causalidad ni convierte el score en ppm."
    ),
    "model_1_vs_model_2_score": (
        "La recta describe la asociación de las dos salidas independientes; "
        "no constituye una función de combinación ni una validación."
    ),
    "model_1_vs_model_2_ood": (
        "La diagonal indica igual fracción fuera de dominio; "
        "las líneas punteadas marcan el corte OOD de 25 %."
    ),
    "model_1_absolute_error_by_case": (
        "Las barras son errores absolutos por caso. "
        "La línea horizontal de trazo y punto indica la media observada."
    ),
    "model_2_score_by_case": (
        "La línea de trazo y punto indica el score medio. "
        "Las líneas punteadas señalan los cortes descriptivos 0,40 y 0,70."
    ),
    "model_1_ood_by_case": (
        "La línea punteada señala 25 % fuera de dominio; "
        "la línea de trazo y punto es la media de estos casos."
    ),
    "model_2_ood_by_case": (
        "La línea punteada señala 25 % fuera de dominio; "
        "la línea de trazo y punto es la media de estos casos."
    ),
}


def build_figures(context, styles) -> list:
    story: list = [
        PageBreak(),
        paragraph("7. Graficas", styles["LS_H1"]),
    ]
    for figure in context.figures:
        if not figure.is_file() or figure.stem == "sentinel_inputs":
            continue
        block = [
            paragraph(
                figure.stem.replace("_", " ").title(),
                styles["LS_H2"],
            ),
            Image(
                str(figure),
                width=16.0 * cm,
                height=10.0 * cm,
                kind="proportional",
            ),
        ]
        description = FIGURE_EXPLANATIONS.get(figure.stem)
        if description:
            block.append(paragraph(description, styles["LS_Small"]))
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))
    return story
