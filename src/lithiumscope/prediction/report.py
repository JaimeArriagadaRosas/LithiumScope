from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    destination: Path,
) -> Path | None:
    if x not in frame.columns or y not in frame.columns:
        return None
    values = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 2:
        return None
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(values[x], values[y])
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def _case_bar(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    destination: Path,
    horizontal_lines: tuple[float, ...] = (),
) -> Path | None:
    if "case_id" not in frame.columns or value_column not in frame.columns:
        return None
    values = frame[["case_id", value_column]].copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values.dropna()
    if values.empty:
        return None

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(values["case_id"].astype(str), values[value_column])
    for threshold in horizontal_lines:
        axis.axhline(threshold, linestyle="--", linewidth=1)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_integrated_figures(
    paired: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    work = paired.copy()
    if {"Li_icpms", "Li_icpms_predicted"} <= set(work.columns):
        work["model_1_absolute_error"] = (
            pd.to_numeric(work["Li_icpms"], errors="coerce")
            - pd.to_numeric(work["Li_icpms_predicted"], errors="coerce")
        ).abs()

    outputs = [
        _scatter(
            work,
            "Li_icpms",
            "Li_icpms_predicted",
            "Li real vs. prediccion Modelo 1",
            figure_dir / "li_real_vs_model_1.png",
        ),
        _case_bar(
            work,
            value_column="model_1_absolute_error",
            title="Error absoluto del Modelo 1 por caso",
            ylabel="Error absoluto (ppm)",
            destination=figure_dir / "model_1_absolute_error_by_case.png",
        ),
        _scatter(
            work,
            "Li_icpms",
            "prospectivity_score",
            "Li real vs. score Modelo 2",
            figure_dir / "li_real_vs_model_2_score.png",
        ),
        _scatter(
            work,
            "Li_icpms_predicted",
            "prospectivity_score",
            "Prediccion Modelo 1 vs. score Modelo 2",
            figure_dir / "model_1_vs_model_2_score.png",
        ),
        _case_bar(
            work,
            value_column="prospectivity_score",
            title="Score de prioridad del Modelo 2 por caso",
            ylabel="Score de prioridad",
            destination=figure_dir / "model_2_score_by_case.png",
            horizontal_lines=(0.4, 0.7),
        ),
    ]
    return [path for path in outputs if path is not None]


def write_workbook(
    path: Path,
    *,
    model_1: pd.DataFrame,
    model_2: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        model_1.to_excel(writer, sheet_name="modelo_1", index=False)
        model_2.to_excel(writer, sheet_name="modelo_2", index=False)
        paired.to_excel(writer, sheet_name="casos_emparejados", index=False)
        correlations.to_excel(writer, sheet_name="correlaciones", index=False)
        concordance.to_excel(writer, sheet_name="concordancia", index=False)
        training_vs_external.to_excel(
            writer,
            sheet_name="training_vs_external",
            index=False,
        )
    return path


def _pdf_text(value) -> str:
    if value is None:
        return "N/D"
    text = str(value)
    replacements = {
        "↔": "<->",
        "→": "->",
        "—": "-",
        "–": "-",
        "²": "2",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return escape(text)


def _number(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/D"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="LS_Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=colors.HexColor("#17324D"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LS_Subtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5968"),
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LS_H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LS_H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#274F6F"),
            spaceBefore=7,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LS_Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LS_Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LS_Warning",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#7A3E00"),
            backColor=colors.HexColor("#FFF3D6"),
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def _paragraph(text, style):
    return Paragraph(_pdf_text(text), style)


def _table(
    rows: list[list],
    *,
    widths: list[float] | None = None,
    header: bool = True,
    font_size: float = 8,
) -> Table:
    normalized = [
        [
            cell
            if hasattr(cell, "wrapOn")
            else Paragraph(_pdf_text(cell), ParagraphStyle(
                name=f"cell_{font_size}",
                fontName="Helvetica",
                fontSize=font_size,
                leading=font_size + 2,
            ))
            for cell in row
        ]
        for row in rows
    ]
    table = LongTable(
        normalized,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D2DB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header and rows:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0F6")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#697785"))
    canvas.drawString(1.6 * cm, 0.8 * cm, "LithiumScope - informe reproducible")
    canvas.drawRightString(
        A4[0] - 1.6 * cm,
        0.8 * cm,
        f"Pagina {document.page}",
    )
    canvas.restoreState()


def _metric_rows(metrics: dict) -> list[list]:
    if not metrics:
        return [["Metrica", "Valor"], ["Sin metricas disponibles", "N/D"]]
    return [["Metrica", "Valor"]] + [
        [key, _number(value, 4)]
        for key, value in metrics.items()
    ]


def _case_lookup(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "case_id" not in frame.columns:
        return {}
    return {
        str(row["case_id"]): row
        for _, row in frame.iterrows()
    }


def _case_story(
    *,
    case_id: str,
    model_1_row: pd.Series | None,
    model_2_row: pd.Series | None,
    concordance_row: pd.Series | None,
    styles,
) -> list:
    story: list = [
        _paragraph(f"Caso {case_id}", styles["LS_H2"]),
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
        ["Li real reservado para evaluacion", _number(li_real, 3)],
        ["Li estimado por Modelo 1", _number(li_pred, 3)],
        [
            "Intervalo empirico q90 M1",
            (
                f"{_number(interval_low, 3)} a {_number(interval_high, 3)} ppm"
                if interval_low is not None and interval_high is not None
                else "N/D"
            ),
        ],
        [
            "Aplicabilidad M1",
            (
                str(model_1_row.get("applicability_warning", "N/D"))
                if model_1_row is not None
                else "N/D"
            ),
        ],
        ["Score de prioridad M2", _number(score, 3)],
        [
            "Prioridad M2",
            (
                str(model_2_row.get("priority", "N/D")).upper()
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Aplicabilidad M2",
            (
                str(model_2_row.get("applicability_warning", "N/D"))
                if model_2_row is not None
                else "N/D"
            ),
        ],
        [
            "Escena Sentinel-2",
            (
                str(model_2_row.get("sentinel_scene_id", "N/D"))
                if model_2_row is not None
                else "N/D"
            ),
        ],
    ]
    story.append(_table(rows, widths=[6.0 * cm, 10.0 * cm], font_size=8.5))
    story.append(Spacer(1, 5))

    story.append(
        _paragraph(
            "Datos solicitados. El Modelo 1 utiliza las variables geoquimicas, "
            "geograficas y contextuales compatibles con su esquema; el Li real se "
            "conserva solamente como verdad de referencia para esta demostracion y no "
            "se entrega como predictor. El Modelo 2 analiza el parche multibanda "
            "Sentinel-2 asociado a las coordenadas del mismo caso.",
            styles["LS_Small"],
        )
    )

    if model_1_row is not None:
        story.append(_paragraph("Lectura del Modelo 1", styles["LS_H2"]))
        story.append(
            _paragraph(
                model_1_row.get(
                    "scientific_interpretation",
                    "Sin interpretacion disponible.",
                ),
                styles["LS_Body"],
            )
        )
    if model_2_row is not None:
        story.append(_paragraph("Lectura del Modelo 2", styles["LS_H2"]))
        story.append(
            _paragraph(
                model_2_row.get(
                    "scientific_interpretation",
                    "Sin interpretacion disponible.",
                ),
                styles["LS_Body"],
            )
        )
    if concordance_row is not None:
        story.append(_paragraph("Lectura integrada", styles["LS_H2"]))
        story.append(
            _paragraph(
                concordance_row.get(
                    "scientific_interpretation",
                    "Sin interpretacion integrada disponible.",
                ),
                styles["LS_Body"],
            )
        )
    story.append(Spacer(1, 8))
    return story


def write_pdf_report(
    path: Path,
    *,
    title: str,
    run_id: str,
    mode: str,
    interpretation: str,
    model_1_identity: dict,
    model_2_identity: dict,
    model_1_metrics: dict,
    model_2_metrics: dict,
    model_1_predictions: pd.DataFrame,
    model_2_predictions: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
    overlap_audit: dict | None,
    input_info: dict | None,
    model_1_diagnostics: dict | None,
    model_2_diagnostics: dict | None,
    figures: list[Path],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.6 * cm,
        title=title,
        author="LithiumScope",
        subject="Prediccion integrada y demostracion cientifica reproducible",
    )

    story: list = [
        Spacer(1, 0.8 * cm),
        _paragraph(title, styles["LS_Title"]),
        _paragraph(
            f"Ejecucion: {run_id}<br/>Modo: {mode}<br/>"
            "Informe generado automaticamente a partir de artefactos reproducibles.",
            styles["LS_Subtitle"],
        ),
        _paragraph(
            "Alcance: apoyo preliminar para analisis y priorizacion. "
            "No confirma un recurso, no sustituye ICP-MS y no reemplaza "
            "validacion geologica, mineralogica ni de terreno.",
            styles["LS_Warning"],
        ),
        _paragraph("1. Trazabilidad de modelos", styles["LS_H1"]),
        _table(
            [
                ["Modelo", "Algoritmo", "Run de entrenamiento", "SHA-256"],
                [
                    "Modelo 1",
                    model_1_identity.get("algorithm", "N/D"),
                    model_1_identity.get("run_id", "N/D"),
                    str(model_1_identity.get("model_sha256", "N/D"))[:20] + "...",
                ],
                [
                    "Modelo 2",
                    model_2_identity.get("algorithm", "N/D"),
                    model_2_identity.get("run_id", "N/D"),
                    str(model_2_identity.get("model_sha256", "N/D"))[:20] + "...",
                ],
            ],
            widths=[2.2 * cm, 3.0 * cm, 5.6 * cm, 5.0 * cm],
            font_size=7.5,
        ),
        _paragraph("2. Procedencia y cobertura de datos", styles["LS_H1"]),
    ]

    if input_info:
        source = input_info.get("demonstration", input_info)
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
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                source_rows.append([key, value])
        if len(source_rows) > 1:
            story.append(
                _table(
                    source_rows,
                    widths=[5.0 * cm, 11.0 * cm],
                    font_size=7.5,
                )
            )

    story.append(
        _paragraph(
            "El Li real, cuando existe, se conserva como verdad de referencia para "
            "evaluar la demostracion. No se utiliza como variable de entrada para "
            "predecir Li ni para calcular el score espacial.",
            styles["LS_Warning"],
        )
    )

    if model_1_diagnostics or model_2_diagnostics:
        story.append(_paragraph("Diagnosticos de aplicabilidad", styles["LS_H2"]))
        diag_rows = [["Diagnostico", "Modelo 1", "Modelo 2"]]
        diag_rows.append(
            [
                "Casos fuera de dominio / fallidos",
                (
                    model_1_diagnostics.get("out_of_domain_rows", "N/D")
                    if model_1_diagnostics
                    else "N/D"
                ),
                (
                    model_2_diagnostics.get("failed_cases", "N/D")
                    if model_2_diagnostics
                    else "N/D"
                ),
            ]
        )
        diag_rows.append(
            [
                "Variables/features ausentes",
                (
                    ", ".join(model_1_diagnostics.get("missing_expected_columns", []))
                    or "ninguna"
                    if model_1_diagnostics
                    else "N/D"
                ),
                (
                    ", ".join(model_2_diagnostics.get("missing_feature_counts", {}).keys())
                    or "ninguna"
                    if model_2_diagnostics
                    else "N/D"
                ),
            ]
        )
        story.append(
            _table(
                diag_rows,
                widths=[5.0 * cm, 5.5 * cm, 5.5 * cm],
                font_size=7.5,
            )
        )
        if model_1_diagnostics:
            generated = model_1_diagnostics.get(
                "generated_during_preparation",
                [],
            )
            if generated:
                story.append(
                    _paragraph(
                        "Variables derivadas generadas por el pipeline antes de predecir: "
                        + ", ".join(generated)
                        + ". No constituyen datos faltantes del usuario.",
                        styles["LS_Small"],
                    )
                )

    story.append(_paragraph("3. Interpretacion cientifica general", styles["LS_H1"]))

    for paragraph in interpretation.splitlines():
        if paragraph.strip():
            story.append(_paragraph(paragraph, styles["LS_Body"]))
        else:
            story.append(Spacer(1, 4))

    story.extend(
        [
            _paragraph("4. Metricas externas", styles["LS_H1"]),
            _paragraph("Modelo 1", styles["LS_H2"]),
            _table(
                _metric_rows(model_1_metrics),
                widths=[8.0 * cm, 7.0 * cm],
                font_size=8.5,
            ),
            _paragraph("Modelo 2", styles["LS_H2"]),
            _table(
                _metric_rows(model_2_metrics),
                widths=[8.0 * cm, 7.0 * cm],
                font_size=8.5,
            ),
            _paragraph("5. Entrenamiento vs. evaluacion actual", styles["LS_H1"]),
        ]
    )

    if training_vs_external.empty:
        story.append(_paragraph("Sin comparacion disponible.", styles["LS_Body"]))
    else:
        rows = [["Modelo", "Metrica", "Entrenamiento", "Externo", "Delta"]]
        for _, row in training_vs_external.iterrows():
            rows.append(
                [
                    row.get("model_group"),
                    row.get("metric"),
                    _number(row.get("training_oof"), 4),
                    _number(row.get("external_demo"), 4),
                    _number(row.get("delta_external_minus_training"), 4),
                ]
            )
        story.append(
            _table(
                rows,
                widths=[2.5 * cm, 3.0 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm],
                font_size=7.5,
            )
        )

    story.append(_paragraph("6. Correlaciones y concordancia", styles["LS_H1"]))
    if correlations.empty:
        story.append(_paragraph("Sin correlaciones disponibles.", styles["LS_Body"]))
    else:
        rows = [["Relacion", "n", "Pearson", "Spearman"]]
        for _, row in correlations.iterrows():
            rows.append(
                [
                    row.get("relationship"),
                    row.get("n"),
                    _number(row.get("pearson"), 3),
                    _number(row.get("spearman"), 3),
                ]
            )
        story.append(
            _table(
                rows,
                widths=[9.0 * cm, 1.5 * cm, 3.0 * cm, 3.0 * cm],
                font_size=7.5,
            )
        )
        story.append(
            _paragraph(
                "Estas asociaciones describen relacion estadistica dentro de esta "
                "muestra y no establecen causalidad.",
                styles["LS_Small"],
            )
        )

    if overlap_audit:
        story.append(_paragraph("Auditoria de independencia", styles["LS_H2"]))
        story.append(
            _table(
                [
                    ["Campo", "Resultado"],
                    ["Estado", overlap_audit.get("status", "N/D")],
                    [
                        "Coincidencias de identificador",
                        overlap_audit.get("sample_id_matches", "N/D"),
                    ],
                    [
                        "Coincidencias por coordenadas",
                        overlap_audit.get("coordinate_matches", "N/D"),
                    ],
                ],
                widths=[5.5 * cm, 10.5 * cm],
                font_size=8,
            )
        )

    story.append(_paragraph("7. Graficas", styles["LS_H1"]))
    for figure in figures:
        if not figure.is_file():
            continue
        story.append(
            KeepTogether(
                [
                    _paragraph(
                        figure.stem.replace("_", " ").title(),
                        styles["LS_H2"],
                    ),
                    Image(str(figure), width=16.0 * cm, height=10.0 * cm, kind="proportional"),
                    Spacer(1, 8),
                ]
            )
        )

    story.append(PageBreak())
    story.append(_paragraph("8. Resumen de casos", styles["LS_H1"]))
    if paired.empty:
        story.append(_paragraph("No existen casos emparejados.", styles["LS_Body"]))
    else:
        summary_rows = [
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
        concordance_lookup = _case_lookup(concordance)
        for _, row in paired.iterrows():
            case_id = str(row["case_id"])
            real = pd.to_numeric(pd.Series([row.get("Li_icpms")]), errors="coerce").iloc[0]
            pred = pd.to_numeric(pd.Series([row.get("Li_icpms_predicted")]), errors="coerce").iloc[0]
            error = abs(real - pred) if pd.notna(real) and pd.notna(pred) else None
            c_row = concordance_lookup.get(case_id)
            summary_rows.append(
                [
                    case_id,
                    _number(real, 2),
                    _number(pred, 2),
                    _number(error, 2),
                    _number(row.get("prospectivity_score"), 3),
                    str(row.get("priority", "N/D")).upper(),
                    (
                        str(c_row.get("concordance", "N/D"))
                        if c_row is not None
                        else "N/D"
                    ),
                ]
            )
        story.append(
            _table(
                summary_rows,
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

    story.append(PageBreak())
    story.append(_paragraph("9. Interpretacion caso por caso", styles["LS_H1"]))
    model_1_lookup = _case_lookup(model_1_predictions)
    model_2_lookup = _case_lookup(model_2_predictions)
    concordance_lookup = _case_lookup(concordance)
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
            _case_story(
                case_id=case_id,
                model_1_row=model_1_lookup.get(case_id),
                model_2_row=model_2_lookup.get(case_id),
                concordance_row=concordance_lookup.get(case_id),
                styles=styles,
            )
        )
        if index != len(case_ids) - 1:
            story.append(Spacer(1, 4))

    story.extend(
        [
            PageBreak(),
            _paragraph("10. Notas de interpretacion", styles["LS_H1"]),
            _paragraph(
                "Modelo 1 responde a una estimacion de Li_icpms para una muestra "
                "compatible con su dominio. Modelo 2 responde a prioridad relativa "
                "de revision espacial/espectral. La salida de un modelo no se usa "
                "como feature de entrada del otro.",
                styles["LS_Body"],
            ),
            _paragraph(
                "En una demostracion externa pequena, ROC-AUC y Average Precision "
                "pueden reflejar capacidad de ordenamiento incluso cuando el umbral "
                "operativo actual no identifica positivos. Por eso las metricas de "
                "ranking y las metricas binarias se reportan separadamente.",
                styles["LS_Body"],
            ),
        ]
    )

    document.build(
        story,
        onFirstPage=_page_footer,
        onLaterPages=_page_footer,
    )
    return path
