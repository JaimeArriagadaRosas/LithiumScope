from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from lithiumscope.prediction.reporting.primitives import (
    paragraph,
    table,
)


def build_overview(context, styles) -> list:
    story: list = [
        Spacer(1, 0.8 * cm),
        paragraph(
            context.title,
            styles["LS_Title"],
        ),
        paragraph(
            f"Ejecucion: {context.run_id}\n"
            f"Modo: {context.mode}\n"
            "Informe generado automaticamente a partir de "
            "artefactos reproducibles.",
            styles["LS_Subtitle"],
        ),
        paragraph(
            "Alcance: apoyo preliminar para analisis y priorizacion. "
            "No confirma un recurso, no sustituye ICP-MS y no "
            "reemplaza validacion geologica, mineralogica ni de "
            "terreno.",
            styles["LS_Warning"],
        ),
        paragraph(
            "1. Trazabilidad de modelos",
            styles["LS_H1"],
        ),
        table(
            [
                ["Campo", "Valor"],
                ["Run de prediccion", context.run_id],
                [
                    "Commit Git",
                    context.runtime.get(
                        "git_commit",
                        "N/D",
                    ),
                ],
                [
                    "Modelo 1 - algoritmo",
                    context.model_1_identity.get(
                        "algorithm",
                        "N/D",
                    ),
                ],
                [
                    "Modelo 1 - run de entrenamiento",
                    context.model_1_identity.get(
                        "run_id",
                        "N/D",
                    ),
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
                    context.model_2_identity.get(
                        "algorithm",
                        "N/D",
                    ),
                ],
                [
                    "Modelo 2 - run de entrenamiento",
                    context.model_2_identity.get(
                        "run_id",
                        "N/D",
                    ),
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
        ),
        paragraph(
            "2. Procedencia y cobertura de datos",
            styles["LS_H1"],
        ),
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
            story.append(
                table(
                    source_rows,
                    widths=[5.0 * cm, 11.0 * cm],
                    font_size=7.5,
                )
            )

    story.append(
        paragraph(
            "El Li real, cuando existe, se conserva como verdad de "
            "referencia para evaluar la demostracion. No se utiliza "
            "como variable de entrada para predecir Li ni para "
            "calcular el score espacial.",
            styles["LS_Warning"],
        )
    )

    if (
        context.model_1_diagnostics
        or context.model_2_diagnostics
    ):
        story.append(
            paragraph(
                "Diagnosticos de aplicabilidad",
                styles["LS_H2"],
            )
        )
        m1 = context.model_1_diagnostics or {}
        m2 = context.model_2_diagnostics or {}
        story.append(
            table(
                [
                    [
                        "Diagnostico",
                        "Modelo 1",
                        "Modelo 2",
                    ],
                    [
                        "Casos fuera de dominio / fallidos",
                        m1.get(
                            "out_of_domain_rows",
                            "N/D",
                        )
                        if context.model_1_diagnostics
                        else "N/D",
                        m2.get(
                            "failed_cases",
                            "N/D",
                        )
                        if context.model_2_diagnostics
                        else "N/D",
                    ],
                    [
                        "Variables/features ausentes",
                        (
                            ", ".join(
                                m1.get(
                                    "missing_expected_columns",
                                    [],
                                )
                            )
                            or "ninguna"
                        )
                        if context.model_1_diagnostics
                        else "N/D",
                        (
                            ", ".join(
                                m2.get(
                                    "missing_feature_counts",
                                    {},
                                ).keys()
                            )
                            or "ninguna"
                        )
                        if context.model_2_diagnostics
                        else "N/D",
                    ],
                ],
                widths=[
                    5.0 * cm,
                    5.5 * cm,
                    5.5 * cm,
                ],
                font_size=7.5,
            )
        )
        generated = m1.get(
            "generated_during_preparation",
            [],
        )
        if generated:
            story.append(
                paragraph(
                    "Variables derivadas generadas por el pipeline "
                    "antes de predecir: "
                    + ", ".join(generated)
                    + ". No constituyen datos faltantes del usuario.",
                    styles["LS_Small"],
                )
            )

    story.append(
        paragraph(
            "3. Interpretacion cientifica general",
            styles["LS_H1"],
        )
    )
    for item in context.interpretation.splitlines():
        if item.strip():
            story.append(
                paragraph(
                    item,
                    styles["LS_Body"],
                )
            )
        else:
            story.append(Spacer(1, 4))
    return story
