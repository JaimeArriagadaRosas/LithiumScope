"""Correlations, model concordance and independence in the original PDF."""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import PageBreak

from lithiumscope.prediction.reporting.formatting import number
from lithiumscope.prediction.reporting.primitives import paragraph, table


def performance_concordance(context, styles) -> list:
    story: list = []
    story.extend(
        [
            PageBreak(),
            paragraph(
                "6. Correlaciones y concordancia",
                styles["LS_H1"],
            ),
        ]
    )

    correlations = context.correlations
    if correlations.empty:
        story.append(
            paragraph(
                "Sin correlaciones disponibles.",
                styles["LS_Body"],
            )
        )
    else:
        rows = [
            [
                "Relacion",
                "n",
                "Pearson",
                "Spearman",
            ]
        ]
        for _, row in correlations.iterrows():
            rows.append(
                [
                    row.get("relationship"),
                    row.get("n"),
                    number(
                        row.get("pearson"),
                        3,
                    ),
                    number(
                        row.get("spearman"),
                        3,
                    ),
                ]
            )
        story.append(
            table(
                rows,
                widths=[
                    9.0 * cm,
                    1.5 * cm,
                    3.0 * cm,
                    3.0 * cm,
                ],
                font_size=7.5,
            )
        )
        story.append(
            paragraph(
                "Estas asociaciones describen relacion estadistica "
                "dentro de esta muestra y no establecen causalidad.",
                styles["LS_Small"],
            )
        )

    concordance = context.concordance
    if (
        not concordance.empty
        and "concordance" in concordance.columns
    ):
        counts = concordance["concordance"].value_counts()
        story.append(
            paragraph(
                "Distribucion de concordancia",
                styles["LS_H2"],
            )
        )
        story.append(
            table(
                [
                    ["Categoria", "Casos"],
                    [
                        "Concordante alta",
                        int(
                            counts.get(
                                "concordante_alta",
                                0,
                            )
                        ),
                    ],
                    [
                        "Concordante baja",
                        int(
                            counts.get(
                                "concordante_baja",
                                0,
                            )
                        ),
                    ],
                    [
                        "Divergente M1 alto",
                        int(
                            counts.get(
                                "divergente_m1_alto",
                                0,
                            )
                        ),
                    ],
                    [
                        "Divergente M2 alto",
                        int(
                            counts.get(
                                "divergente_m2_alto",
                                0,
                            )
                        ),
                    ],
                ],
                widths=[10.5 * cm, 5.5 * cm],
                font_size=8,
            )
        )
        story.append(
            paragraph(
                "Una discrepancia entre modelos es un resultado "
                "cientifico valido: cada modelo responde una pregunta "
                "distinta y ninguno debe forzarse a confirmar al otro.",
                styles["LS_Small"],
            )
        )

    if context.overlap_audit:
        audit = context.overlap_audit
        story.append(
            paragraph(
                "Auditoria de independencia",
                styles["LS_H2"],
            )
        )
        story.append(
            table(
                [
                    ["Campo", "Resultado"],
                    [
                        "Estado",
                        audit.get(
                            "status",
                            "N/D",
                        ),
                    ],
                    [
                        "Coincidencias de identificador",
                        audit.get(
                            "sample_id_matches",
                            "N/D",
                        ),
                    ],
                    [
                        "Coincidencias por coordenadas",
                        audit.get(
                            "coordinate_matches",
                            "N/D",
                        ),
                    ],
                ],
                widths=[5.5 * cm, 10.5 * cm],
                font_size=8,
            )
        )


    return story
