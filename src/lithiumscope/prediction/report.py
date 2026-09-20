from __future__ import annotations

from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _scatter(frame: pd.DataFrame, x: str, y: str, title: str, destination: Path) -> Path | None:
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


def save_integrated_figures(paired: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _scatter(
            paired,
            "Li_icpms",
            "Li_icpms_predicted",
            "Li real vs. predicción Modelo 1",
            figure_dir / "li_real_vs_model_1.png",
        ),
        _scatter(
            paired,
            "Li_icpms",
            "prospectivity_score",
            "Li real vs. score Modelo 2",
            figure_dir / "li_real_vs_model_2_score.png",
        ),
        _scatter(
            paired,
            "Li_icpms_predicted",
            "prospectivity_score",
            "Predicción Modelo 1 vs. score Modelo 2",
            figure_dir / "model_1_vs_model_2_score.png",
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
        training_vs_external.to_excel(writer, sheet_name="training_vs_external", index=False)
    return path


def write_html_report(
    path: Path,
    *,
    title: str,
    interpretation: str,
    model_1_metrics: dict,
    model_2_metrics: dict,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
    figures: list[Path],
) -> Path:
    def frame_html(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<p>Sin datos disponibles.</p>"
        return frame.to_html(index=False, border=0, classes="table")

    figure_html = "".join(
        f'<figure><img src="{escape(str(figure.relative_to(path.parent)).replace(chr(92), "/"))}" '
        f'alt="{escape(figure.stem)}"></figure>'
        for figure in figures
    )
    interpretation_html = "".join(
        f"<p>{escape(line)}</p>" if line else "<br>"
        for line in interpretation.splitlines()
    )
    payload = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
h1, h2 {{ margin-top: 1.5rem; }}
.table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
.table th, .table td {{ border: 1px solid #ddd; padding: 0.45rem; text-align: left; }}
figure img {{ max-width: 100%; height: auto; }}
code {{ background: #f3f3f3; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<h2>Interpretación científica</h2>
{interpretation_html}
<h2>Métricas externas — Modelo 1</h2>
<pre>{escape(json.dumps(model_1_metrics, indent=2, ensure_ascii=False))}</pre>
<h2>Métricas externas — Modelo 2</h2>
<pre>{escape(json.dumps(model_2_metrics, indent=2, ensure_ascii=False))}</pre>
<h2>Correlaciones</h2>
{frame_html(correlations)}
<h2>Concordancia entre modelos</h2>
{frame_html(concordance)}
<h2>Entrenamiento vs. evaluación actual</h2>
{frame_html(training_vs_external)}
<h2>Figuras</h2>
{figure_html or "<p>Sin figuras suficientes.</p>"}
</body>
</html>
"""
    path.write_text(payload, encoding="utf-8")
    return path
