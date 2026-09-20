from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from lithiumscope.prediction.reporting.formatting import (
    fraction,
    number,
)
from lithiumscope.prediction.reporting.primitives import (
    paragraph,
    table,
)


def case_lookup(
    frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    if frame.empty or "case_id" not in frame.columns:
        return {}
    return {
        str(row["case_id"]): row
        for _, row in frame.iterrows()
    }


def case_story(
    *,
    case_id: str,
    model_1_row: pd.Series | None,
    model_2_row: pd.Series | None,
    concordance_row: pd.Series | None,
    styles,
) -> list:
    story: list = [
        paragraph(
            f"Caso {case_id}",
            styles["LS_H2"],
        ),
    ]

    li_real = (
        model_1_row.get("Li_icpms")
        if model_1_row is not None
        else (
            model_2_row.get("Li_icpms")
            if model_2_row is not None
            else None
        )
    )
    li_pred = (
        model_1_row.get("Li_icpms_predicted")
        if model_1_row is not None
        else None
    )
    interval_low = (
        model_1_row.get("Li_icpms_interval_low_q90")
        if model_1_row is not None
        else None
    )
    interval_high = (
        model_1_row.get("Li_icpms_interval_high_q90")
        if model_1_row is not None
        else None
    )
    score = (
        model_2_row.get("prospectivity_score")
        if model_2_row is not None
        else None
    )

    rows = [
        ["Dato", "Resultado"],
        [
            "Li real (ground truth; no predictor)",
            number(li_real, 3),
        ],
        [
            "Li estimado por Modelo 1",
            number(li_pred, 3),
        ],
        [
            "Intervalo empirico q90 M1",
            (
                f"{number(interval_low, 3)} a "
                f"{number(interval_high, 3)} ppm"
                if (
                    interval_low is not None
                    and interval_high is not None
                )
                else "N/D"
            ),
        ],
        [
            "Fraccion OOD M1",
            (
                fraction(
                    model_1_row.get(
                        "out_of_training_range_fraction"
                    )
                )
                if model_1_row is not None
                else "N/D"
            ),
        ],
        [
            "Aplicabilidad M1",
            (
                str(
                    model_1_row.get(
                        "applicability_warning",
                        "N/D",
                    )
                )
                if model_1_row is not None
                else "N/D"
            ),
        ],
        [
            "Score de prioridad M2",
            number(score, 3),
        ],
        [
            "Prioridad M2",
            (
                str(
                    model_2_row.get(
                        "priority",
                        "N/D",
                    )
                ).upper()
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Fraccion OOD M2",
            (
                fraction(
                    model_2_row.get(
                        "out_of_training_range_fraction"
                    )
                )
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Aplicabilidad M2",
            (
                str(
                    model_2_row.get(
                        "applicability_warning",
                        "N/D",
                    )
                )
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Escena Sentinel-2",
            (
                str(
                    model_2_row.get(
                        "sentinel_scene_id",
                        "N/D",
                    )
                )
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Nubosidad Sentinel-2",
            (
                number(
                    model_2_row.get(
                        "sentinel_cloud_cover"
                    ),
                    2,
                )
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Fecha Sentinel-2",
            (
                str(
                    model_2_row.get(
                        "sentinel_datetime",
                        "N/D",
                    )
                )
                if model_2_row is not None
                else "N/D"
            ),
        ],
    ]
    story.append(
        table(
            rows,
            widths=[6.0 * cm, 10.0 * cm],
            font_size=8.5,
        )
    )
    story.append(Spacer(1, 5))

    story.append(
        paragraph(
            "Datos usados. El Modelo 1 recibe solamente variables "
            "predictoras compatibles con su esquema; Li_icpms real "
            "se conserva como ground truth para evaluacion y nunca "
            "se entrega al estimador. El Modelo 2 recibe "
            "caracteristicas espaciales/espectrales derivadas del "
            "parche multibanda Sentinel-2 del caso. La salida del "
            "Modelo 1 no alimenta al Modelo 2.",
            styles["LS_Small"],
        )
    )

    if model_1_row is not None:
        story.append(
            paragraph(
                "Lectura del Modelo 1",
                styles["LS_H2"],
            )
        )
        story.append(
            paragraph(
                model_1_row.get(
                    "scientific_interpretation",
                    "Sin interpretacion disponible.",
                ),
                styles["LS_Body"],
            )
        )
    if model_2_row is not None:
        story.append(
            paragraph(
                "Lectura del Modelo 2",
                styles["LS_H2"],
            )
        )
        story.append(
            paragraph(
                model_2_row.get(
                    "scientific_interpretation",
                    "Sin interpretacion disponible.",
                ),
                styles["LS_Body"],
            )
        )
    if concordance_row is not None:
        story.append(
            paragraph(
                "Lectura integrada",
                styles["LS_H2"],
            )
        )
        story.append(
            paragraph(
                concordance_row.get(
                    "scientific_interpretation",
                    "Sin interpretacion integrada disponible.",
                ),
                styles["LS_Body"],
            )
        )

    story.append(Spacer(1, 8))
    return story
