from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import LongTable, Paragraph, TableStyle

from lithiumscope.prediction.reporting.formatting import pdf_text


def paragraph(text, style):
    payload = pdf_text(text).replace("\n", "<br/>")
    return Paragraph(payload, style)


def table(
    rows: list[list],
    *,
    widths: list[float] | None = None,
    header: bool = True,
    font_size: float = 8,
):
    normalized = [
        [
            cell
            if hasattr(cell, "wrapOn")
            else Paragraph(
                pdf_text(cell),
                ParagraphStyle(
                    name=f"cell_{font_size}",
                    fontName="Helvetica",
                    fontSize=font_size,
                    leading=font_size + 2,
                ),
            )
            for cell in row
        ]
        for row in rows
    ]
    result = LongTable(
        normalized,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.35,
            colors.HexColor("#C9D2DB"),
        ),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header and rows:
        commands.extend(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#E8F0F6"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
            ]
        )
    result.setStyle(TableStyle(commands))
    return result
