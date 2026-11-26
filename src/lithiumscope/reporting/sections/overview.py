from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from lithiumscope.reporting.primitives import (
    paragraph,
    table,
)


def build_overview(context, styles) -> list:
    story: list = [PageBreak()]

    traceability = table(
        [
            ["Campo", "Valor"],
            ["Run de prediccion", context.run_id],
            [
                "Commit Git",
                context.runtime.get("git_commit", "N/D"),
            ],
            [
                "Modelo 1 - algoritmo",
                context.model_1_identity.get("algorithm", "N/D"),
            ],
            [
                "Modelo 1 - run de entrenamiento",
                context.model_1_identity.get("run_id", "N/D"),
            ],
            [
                "Modelo 1 - SHA-256 del modelo",
                paragraph(
                    str(
                        context.model_1_identity.get(
                            "model_sha256",
                            "N/D",
                        )
                    ),
                    styles["LS_Small"],
                ),
            ],
            [
                "Modelo 1 - SHA-256 del dataset",
                paragraph(
                    str(
                        context.model_1_identity.get(
                            "dataset_sha256",
                            "N/D",
                        )
                    ),
                    styles["LS_Small"],
                ),
            ],
            [
                "Modelo 2 - algoritmo",
                context.model_2_identity.get("algorithm", "N/D"),
            ],
            [
                "Modelo 2 - run de entrenamiento",
                context.model_2_identity.get("run_id", "N/D"),
            ],
            [
                "Modelo 2 - SHA-256 del modelo",
                paragraph(
                    str(
                        context.model_2_identity.get(
                            "model_sha256",
                            "N/D",
                        )
                    ),
                    styles["LS_Small"],
                ),
            ],
            [
                "Modelo 2 - SHA-256 del dataset",
                paragraph(
                    str(
                        context.model_2_identity.get(
                            "dataset_sha256",
                            "N/D",
                        )
                    ),
                    styles["LS_Small"],
                ),
            ],
        ],
        widths=[5.4 * cm, 10.6 * cm],
        font_size=7.2,
    )
    story.append(
        KeepTogether(
            [
                paragraph(
                    "2. Trazabilidad de modelos",
                    styles["LS_H1"],
                ),
                traceability,
            ]
        )
    )

    story.append(PageBreak())

    source_block: list = [
        paragraph(
            "3. Procedencia y cobertura de datos",
            styles["LS_H1"],
        )
    ]
    if context.input_info:
        source = context.input_info.get(
            "demonstration",
            context.input_info,
        )
        source_rows = [["Campo", "Valor"]]
        for key in (
            "source_name",
            "source_repository",
            "source_commit",
            "source_path",
            "source_license",
            "sentinel_datetime",
            "pairing_rule",
        ):
            value = (
                source.get(key)
                if isinstance(source, dict)
                else None
            )
            if value:
                source_rows.append([key, value])
        if len(source_rows) > 1:
            source_block.append(
                table(
                    source_rows,
                    widths=[5.0 * cm, 11.0 * cm],
                    font_size=7.5,
                )
            )

    story.append(KeepTogether(source_block))

    if (
        context.model_1_diagnostics
        or context.model_2_diagnostics
    ):
        m1 = context.model_1_diagnostics or {}
        m2 = context.model_2_diagnostics or {}
        m1_ood = context.statistics["m1_ood"]
        m2_ood = context.statistics["m2_ood"]
        def warning_counts(group):
            frame = getattr(context, group + "_predictions", None)
            if frame is None or "applicability_warning" not in frame:
                return ("N/D", "N/D")
            status = frame["applicability_warning"].astype(str).str.upper()
            return (int(status.eq("OUT_OF_DOMAIN").sum()), int(status.ne("OK").sum()))
        m1_full, m1_not_ok = warning_counts("model_1")
        m2_full, m2_not_ok = warning_counts("model_2")
        diagnostics = [
            paragraph(
                "Diagnosticos de aplicabilidad",
                styles["LS_H2"],
            ),
            table(
                [
                    [
                        "Diagnostico",
                        "Modelo 1",
                        "Modelo 2",
                    ],
                    [
                        "Fuera de rango marginal (>25 %)",
                        m1_ood,
                        m2_ood,
                    ],
                    ["Fuera de dominio (diagnóstico final)", m1_full, m2_full],
                    ["Casos con diagnóstico distinto de OK", m1_not_ok, m2_not_ok],
                    [
                        "Casos fallidos",
                        m1.get("failed_cases", "N/D"),
                        m2.get("failed_cases", "N/D"),
                    ],
                    [
                        "Variables/features ausentes",
                        ", ".join(
                            m1.get(
                                "missing_expected_columns",
                                [],
                            )
                        )
                        or "ninguna",
                        ", ".join(
                            m2.get(
                                "missing_feature_counts",
                                {},
                            ).keys()
                        )
                        or "ninguna",
                    ],
                ],
                widths=[
                    5.0 * cm,
                    5.5 * cm,
                    5.5 * cm,
                ],
                font_size=7.5,
            ),
        ]
        generated = m1.get(
            "generated_during_preparation",
            [],
        )
        if generated:
            diagnostics.append(
                paragraph(
                    "Variables derivadas generadas por el pipeline "
                    "antes de predecir: "
                    + ", ".join(generated)
                    + ". No constituyen datos faltantes del usuario.",
                    styles["LS_Small"],
                )
            )
        story.append(KeepTogether(diagnostics))

    return story
