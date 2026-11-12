from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak

from lithiumscope.prediction.reporting.formatting import number
from lithiumscope.prediction.reporting.sections.performance_concordance import performance_concordance
from lithiumscope.prediction.reporting.primitives import (
    paragraph,
    table,
)


def _metric_rows(metrics: dict) -> list[list]:
    if not metrics:
        return [
            ["Metrica", "Valor"],
            ["Sin metricas disponibles", "N/D"],
        ]
    return [["Metrica", "Valor"]] + [
        [key, number(value, 4)]
        for key, value in metrics.items()
    ]


def build_performance(context, styles) -> list:
    story: list = [
        PageBreak(),
        paragraph(
            "4. Metricas externas",
            styles["LS_H1"],
        ),
        paragraph(
            "Modelo 1",
            styles["LS_H2"],
        ),
        table(
            _metric_rows(context.model_1_metrics),
            widths=[8.0 * cm, 7.0 * cm],
            font_size=8.5,
        ),
        paragraph(
            "Modelo 2",
            styles["LS_H2"],
        ),
        table(
            _metric_rows(context.model_2_metrics),
            widths=[8.0 * cm, 7.0 * cm],
            font_size=8.5,
        ),
        paragraph(
            "Como leer estas metricas",
            styles["LS_H2"],
        ),
        paragraph(
            "Modelo 1: RMSE penaliza mas los errores grandes; MAE "
            "resume el error absoluto medio; R2 compara la varianza "
            "explicada frente a una referencia constante. El Li real "
            "se usa aqui solo como ground truth para evaluacion "
            "externa.",
            styles["LS_Small"],
        ),
        paragraph(
            "Modelo 2: ROC-AUC y Average Precision describen "
            "capacidad de ordenamiento. Balanced Accuracy, precision, "
            "recall y F1 dependen del umbral operativo. El score "
            "representa prioridad exploratoria relativa: no es "
            "concentracion de Li ni probabilidad de yacimiento.",
            styles["LS_Small"],
        ),
    ]

    threshold = context.model_2_classification_threshold
    if threshold is not None:
        scores = pd.to_numeric(
            context.model_2_predictions.get(
                "prospectivity_score",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
        positive_count = int(
            (scores >= float(threshold)).sum()
        )
        story.append(
            paragraph(
                f"Con el umbral operativo actual "
                f"({threshold:.3f}), Modelo 2 clasifico "
                f"{positive_count}/{len(scores)} casos como "
                "positivos.",
                styles["LS_Small"],
            )
        )
        if not scores.empty and positive_count == 0:
            story.append(
                paragraph(
                    "No se predijo ningun positivo con el umbral "
                    "operativo actual. Esto no anula una posible "
                    "capacidad de ranking reflejada por ROC-AUC o "
                    "Average Precision; ambas lecturas se reportan "
                    "por separado y el score no se interpreta como "
                    "probabilidad.",
                    styles["LS_Warning"],
                )
            )

    story.extend(
        [
            PageBreak(),
            paragraph(
                "5. Validacion de entrenamiento vs. evaluacion actual",
                styles["LS_H1"],
            ),
            paragraph(
                "El delta se calcula como evaluacion externa menos "
                "CV/OOF de entrenamiento. Debe interpretarse junto con "
                "tamaño muestral, dominio y aplicabilidad; por si "
                "solo no establece la causa de una diferencia de "
                "rendimiento.",
                styles["LS_Small"],
            ),
        ]
    )

    comparison = context.training_vs_external
    if comparison.empty:
        story.append(
            paragraph(
                "Sin comparacion disponible.",
                styles["LS_Body"],
            )
        )
    else:
        rows = [
            [
                "Modelo",
                "Metrica",
                "CV/OOF de entrenamiento",
                "Externo",
                "Delta",
            ]
        ]
        for _, row in comparison.iterrows():
            rows.append(
                [
                    row.get("model_group"),
                    row.get("metric"),
                    number(
                        row.get("training_oof"),
                        4,
                    ),
                    number(
                        row.get("external_demo"),
                        4,
                    ),
                    number(
                        row.get(
                            "delta_external_minus_training"
                        ),
                        4,
                    ),
                ]
            )
        story.append(
            table(
                rows,
                widths=[
                    2.5 * cm,
                    3.0 * cm,
                    3.5 * cm,
                    3.5 * cm,
                    3.5 * cm,
                ],
                font_size=7.5,
            )
        )

    story.extend(performance_concordance(context, styles))
    return story
