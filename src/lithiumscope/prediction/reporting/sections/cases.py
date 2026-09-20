from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Spacer

from lithiumscope.prediction.reporting.case_cards import (
    case_lookup,
    case_story,
)
from lithiumscope.prediction.reporting.formatting import number
from lithiumscope.prediction.reporting.primitives import (
    paragraph,
    table,
)


def build_cases(context, styles) -> list:
    story: list = [
        PageBreak(),
        paragraph(
            "8. Resumen de casos",
            styles["LS_H1"],
        ),
    ]

    paired = context.paired
    concordance_lookup = case_lookup(
        context.concordance
    )
    if paired.empty:
        story.append(
            paragraph(
                "No existen casos emparejados.",
                styles["LS_Body"],
            )
        )
    else:
        rows = [
            [
                "Caso",
                "Li real",
                "Li M1",
                "Error abs.",
                "Score M2",
                "Prioridad",
                "Concordancia",
            ]
        ]
        for _, row in paired.iterrows():
            case_id = str(row["case_id"])
            real = pd.to_numeric(
                pd.Series(
                    [row.get("Li_icpms")]
                ),
                errors="coerce",
            ).iloc[0]
            predicted = pd.to_numeric(
                pd.Series(
                    [row.get("Li_icpms_predicted")]
                ),
                errors="coerce",
            ).iloc[0]
            error = (
                abs(real - predicted)
                if (
                    pd.notna(real)
                    and pd.notna(predicted)
                )
                else None
            )
            concordance_row = (
                concordance_lookup.get(case_id)
            )
            rows.append(
                [
                    case_id,
                    number(real, 2),
                    number(predicted, 2),
                    number(error, 2),
                    number(
                        row.get(
                            "prospectivity_score"
                        ),
                        3,
                    ),
                    str(
                        row.get(
                            "priority",
                            "N/D",
                        )
                    ).upper(),
                    (
                        str(
                            concordance_row.get(
                                "concordance",
                                "N/D",
                            )
                        )
                        if concordance_row is not None
                        else "N/D"
                    ),
                ]
            )
        story.append(
            table(
                rows,
                widths=[
                    2.4 * cm,
                    2.0 * cm,
                    2.0 * cm,
                    2.0 * cm,
                    2.1 * cm,
                    2.0 * cm,
                    3.5 * cm,
                ],
                font_size=6.8,
            )
        )

    story.extend(
        [
            PageBreak(),
            paragraph(
                "9. Interpretacion caso por caso",
                styles["LS_H1"],
            ),
        ]
    )
    model_1_lookup = case_lookup(
        context.model_1_predictions
    )
    model_2_lookup = case_lookup(
        context.model_2_predictions
    )
    concordance_lookup = case_lookup(
        context.concordance
    )
    case_ids = list(
        dict.fromkeys(
            [
                *model_1_lookup.keys(),
                *model_2_lookup.keys(),
            ]
        )
    )
    for index, case_id in enumerate(case_ids):
        story.extend(
            case_story(
                case_id=case_id,
                model_1_row=model_1_lookup.get(
                    case_id
                ),
                model_2_row=model_2_lookup.get(
                    case_id
                ),
                concordance_row=(
                    concordance_lookup.get(
                        case_id
                    )
                ),
                styles=styles,
            )
        )
        if index != len(case_ids) - 1:
            story.append(Spacer(1, 4))
    return story
