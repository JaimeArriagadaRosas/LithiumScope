from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Spacer

from lithiumscope.prediction.reporting.primitives import paragraph


def build_spatial(context, styles) -> list:
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
            "1. Resultado espacial e interpretacion general",
            styles["LS_H1"],
        ),
    ]

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

    if not context.maps:
        story.append(
            paragraph(
                "No fue posible generar mapas espaciales porque "
                "los casos no contienen coordenadas compatibles.",
                styles["LS_Warning"],
            )
        )
        return story

    labels = {
        "sample_locations": (
            "Mapa de muestras",
            "Ubicacion de los casos evaluados. Cuando existe Li real, "
            "el color representa su concentracion en ppm.",
        ),
        "model_results_map": (
            "Mapa de resultados de los modelos",
            "Comparacion espacial entre la concentracion estimada por "
            "Modelo 1 y el score de prioridad entregado por Modelo 2.",
        ),
    }

    for path in context.maps:
        if not path.is_file():
            continue
        title, description = labels.get(
            path.stem,
            (
                path.stem.replace("_", " ").title(),
                "",
            ),
        )
        block = [
            paragraph(
                title,
                styles["LS_H2"],
            ),
        ]
        if description:
            block.append(
                paragraph(
                    description,
                    styles["LS_Small"],
                )
            )
        block.extend(
            [
                Image(
                    str(path),
                    width=16.0 * cm,
                    height=10.2 * cm,
                    kind="proportional",
                ),
                Spacer(1, 8),
            ]
        )
        story.append(KeepTogether(block))

    return story
