from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Spacer

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

    interpretation_lines = context.interpretation.splitlines()
    if (
        interpretation_lines
        and interpretation_lines[0].strip().upper()
        == "INTERPRETACIÓN CIENTÍFICA INTEGRADA"
    ):
        interpretation_lines = interpretation_lines[1:]

    story.append(
        paragraph(
            "Interpretación científica integrada",
            styles["LS_H2"],
        )
    )

    satellite_preview = next(
        (
            path
            for path in context.figures
            if path.stem == "sentinel_inputs"
            and path.is_file()
        ),
        None,
    )
    if satellite_preview is not None:
        story.append(
            KeepTogether(
                [
                    Image(
                        str(satellite_preview),
                        width=15.8 * cm,
                        height=13.2 * cm,
                        kind="proportional",
                    ),
                    paragraph(
                        "Vista RGB de los parches Sentinel-2 reales "
                        "utilizados como entrada del Modelo 2. Cada "
                        "recuadro corresponde a un caso; el modelo "
                        "opera sobre el parche multibanda completo, "
                        "no solo sobre esta visualización RGB.",
                        styles["LS_Small"],
                    ),
                    Spacer(1, 8),
                ]
            )
        )

    for item in interpretation_lines:
        stripped = item.strip()
        if stripped == "Cobertura de la ejecución:":
            story.append(PageBreak())
        if stripped:
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
