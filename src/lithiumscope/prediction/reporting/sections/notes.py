from __future__ import annotations

from reportlab.platypus import PageBreak

from lithiumscope.prediction.reporting.formatting import number
from lithiumscope.prediction.reporting.primitives import paragraph


def build_notes(context, styles) -> list:
    story: list = [
        PageBreak(),
        paragraph(
            "10. Notas de interpretacion",
            styles["LS_H1"],
        ),
        paragraph(
            "Modelo 1 responde a una estimacion de Li_icpms para "
            "una muestra compatible con su dominio. Modelo 2 "
            "responde a prioridad relativa de revision "
            "espacial/espectral. La salida de un modelo no se usa "
            "como feature de entrada del otro.",
            styles["LS_Body"],
        ),
        paragraph(
            "En una demostracion externa pequena, ROC-AUC y Average "
            "Precision pueden reflejar capacidad de ordenamiento "
            "incluso cuando el umbral operativo actual no identifica "
            "positivos. Por eso las metricas de ranking y las "
            "metricas binarias se reportan separadamente.",
            styles["LS_Body"],
        ),
        paragraph(
            "La discrepancia entre modelos no es un error por si "
            "misma. Puede reflejar diferencias entre evidencia "
            "geoquimica de una muestra y patrones "
            "espaciales/espectrales del sector, y debe conservarse "
            "para revision cientifica.",
            styles["LS_Body"],
        ),
    ]

    if (
        context.lithium_threshold_ppm is not None
        or context.model_2_classification_threshold
        is not None
    ):
        story.append(
            paragraph(
                "Umbrales de la ejecucion: "
                f"Li de referencia="
                f"{number(context.lithium_threshold_ppm, 3)} ppm; "
                f"score operativo M2="
                f"{number(context.model_2_classification_threshold, 3)}. "
                "Estos umbrales apoyan evaluacion y concordancia; "
                "no convierten el score M2 en una probabilidad "
                "geologica.",
                styles["LS_Body"],
            )
        )
    return story
