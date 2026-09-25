from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Spacer

from lithiumscope.prediction.reporting.primitives import paragraph


def build_figures(context, styles) -> list:
    story: list = [
        PageBreak(),
        paragraph(
            "7. Graficas",
            styles["LS_H1"],
        )
    ]
    for figure in context.figures:
        if (
            not figure.is_file()
            or figure.stem == "sentinel_inputs"
        ):
            continue
        story.append(
            KeepTogether(
                [
                    paragraph(
                        figure.stem.replace(
                            "_",
                            " ",
                        ).title(),
                        styles["LS_H2"],
                    ),
                    Image(
                        str(figure),
                        width=16.0 * cm,
                        height=10.0 * cm,
                        kind="proportional",
                    ),
                    Spacer(1, 8),
                ]
            )
        )
    return story
