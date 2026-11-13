"""One-page dynamic M1 -> M2 decision view for the existing ReportLab PDF."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Spacer, Table, TableStyle

from lithiumscope.prediction.reporting.charts import _rgb_preview
from lithiumscope.prediction.reporting.decision import calculate_decision
from lithiumscope.prediction.reporting.primitives import paragraph, table


def _fmt(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return "N/D"
    return f"{number:.{digits}f}" if np.isfinite(number) else "N/D"


def _image_or_note(path_value, styles):
    path = Path(str(path_value))
    if not path.is_file():
        return paragraph("Imagen original no disponible.", styles["LS_Small"])
    try:
        rgb = _rgb_preview(path)
        if rgb is None:
            return paragraph("RGB no disponible.", styles["LS_Small"])
        rgb_uint8 = np.uint8(np.clip(rgb, 0, 1) * 255)
        buffer = io.BytesIO()
        PILImage.fromarray(rgb_uint8, mode="RGB").resize(
            (192, 192), PILImage.Resampling.NEAREST
        ).save(buffer, format="PNG")
        buffer.seek(0)
        return Image(buffer, width=3.6 * cm, height=3.6 * cm)
    except (OSError, ValueError, RuntimeError):
        return paragraph("No fue posible visualizar el parche.", styles["LS_Small"])


def _image_cards(selected: pd.DataFrame, winner_id: str | None, styles):
    cards = []
    for _, row in selected.iterrows():
        case_id = str(row.case_id)
        winner = case_id == winner_id
        heading = paragraph(
            f"{case_id} - {'PRIORIDAD' if winner else 'CANDIDATO'}",
            styles["LS_Small"],
        )
        score = _fmt(row["_m2_score"], 3)
        status = "M2 aplicable" if bool(row["_m2_ok"]) else "M2 fuera de dominio"
        contents = [[heading], [_image_or_note(row.get("image"), styles)], [
            paragraph(f"Score M2: {score}", styles["LS_Small"])
        ], [paragraph(status, styles["LS_Small"])]]
        card = Table(contents, colWidths=[5.35 * cm], hAlign="CENTER")
        card.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 1.2 if winner else 0.5,
             colors.HexColor("#198063") if winner else colors.HexColor("#BAC6D1")),
            ("BACKGROUND", (0, 0), (-1, -1),
             colors.HexColor("#E8F5EE") if winner else colors.HexColor("#F3F6F8")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        cards.append(card)
    while len(cards) < 3:
        cards.append("")
    aligned = Table([cards[:3]], colWidths=[5.55 * cm] * 3, hAlign="CENTER")
    aligned.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return aligned


def build_decision_page(context, styles) -> list:
    """No static candidate IDs, positions, scores, images or winners."""
    decision = calculate_decision(context.paired)
    n_cases = len(decision.ranked)
    story = [
        paragraph("Decisión integrada: de candidatos a una prioridad", styles["LS_H1"]),
        paragraph(
            f"De {n_cases} casos emparejados, el informe interpreta las predicciones "
            "ya calculadas de ambos modelos. No reentrena, ajusta ni combina "
            "features entre estimadores.", styles["LS_Small"],
        ),
        paragraph("Etapa 1 - Preselección por Modelo 1", styles["LS_H2"]),
    ]
    if decision.ranked.empty:
        return story

    top_ids = set(decision.selected.case_id.astype(str)) if not decision.selected.empty else set()
    rows = [[
        "Candidato", "Li M1 (ppm)", "Cota M1 (ppm)",
        "Posición M1", "Score M2", "Resultado",
    ]]
    for _, item in decision.ranked.iterrows():
        case_id = str(item.case_id)
        if case_id == decision.winner_id:
            result = "Prioridad M2"
        elif case_id in top_ids:
            result = "Top 3 M1"
        elif not bool(item["_m1_ok"]):
            result = "M1 fuera dominio"
        else:
            result = "-"
        rows.append([
            case_id, _fmt(item["_m1_ppm"]),
            _fmt(item["_m1_rank_score"]),
            str(int(item["_m1_position"])),
            _fmt(item["_m2_score"], 3),
            result,
        ])
    comparison = table(
        rows, widths=[
            2.7 * cm, 2.7 * cm, 3.1 * cm, 2.3 * cm, 2.8 * cm, 3.0 * cm
        ], font_size=7.0,
    )
    for k, row in enumerate(decision.ranked.itertuples(index=False), start=1):
        name = str(row.case_id)
        if name == decision.winner_id:
            color = "#DCEFE3"
        elif name in top_ids:
            color = "#EAF2FA"
        else:
            continue
        comparison.setStyle(TableStyle([
            ("BACKGROUND", (0, k), (-1, k), colors.HexColor(color))
        ]))
    story.append(comparison)
    story.append(Spacer(1, 0.12 * cm))
    story.append(paragraph(
        f"Regla M1: {decision.m1_rule}. Se excluyen predicciones M1 fuera "
        "del dominio empírico antes de elegir tres. El Li observado NO interviene.",
        styles["LS_Small"],
    ))
    if decision.selected.empty:
        return story

    story.extend([
        paragraph("Etapa 2 - Comparación espectral de las tres imágenes", styles["LS_H2"]),
        _image_cards(decision.selected, decision.winner_id, styles),
        paragraph(
            "Las imágenes RGB son vistas ilustrativas; el score M2 procede "
            "del parche multibanda completo (seis bandas).",
            styles["LS_Small"],
        ),
        Spacer(1, 0.10 * cm),
    ])
    heading = (
        f"Prioridad {'provisional' if decision.status == 'PROVISIONAL' else 'relativa'}: "
        f"{decision.winner_id}"
        if decision.winner_id else "Sin ganador justificable"
    )
    story.append(paragraph(heading, styles["LS_H2"]))
    story.append(paragraph(
        f"Regla M2: {decision.m2_rule}. {decision.reason} "
        "Los scores son relativos; no son probabilidades de yacimiento "
        "ni representan costos reales de extracción.", styles["LS_Small"],
    ))
    if decision.retrospective:
        retrospective = decision.retrospective
        story.append(paragraph(
            "Comprobación retrospectiva (sin influir en la selección): "
            f"mayor Li observado entre los tres = "
            f"{retrospective['best_observed_in_top3']}; "
            f"acierto del primer puesto M1 = "
            f"{'sí' if retrospective['m1_first_matches_observed'] else 'no'}; "
            f"acierto tras priorizar con M2 = "
            f"{'sí' if retrospective['m2_winner_matches_observed'] else 'no'}. "
            "No se atribuye ganancia al M2 sin comparar ambos métodos en "
            "más conjuntos independientes. La independencia "
            "geográfica de la demostración no está certificada.",
            styles["LS_Small"],
        ))
    return story
