from __future__ import annotations

from html import escape
from pathlib import Path
import webbrowser

import pandas as pd


def _table_section(path: Path) -> str:
    try:
        frame = pd.read_csv(path)
        preview = frame.head(100).to_html(index=False, classes="dataframe", border=0)
        suffix = "" if len(frame) <= 100 else f"<p>Vista previa: 100 de {len(frame)} filas.</p>"
        return f"<section><h3>{escape(path.stem)}</h3>{suffix}{preview}</section>"
    except Exception as exc:
        return f"<section><h3>{escape(path.name)}</h3><p>No se pudo previsualizar: {escape(str(exc))}</p></section>"


def build_dashboard(run_dir: Path, title: str | None = None) -> Path:
    title = title or f"LithiumScope — {run_dir.name}"
    table_paths = sorted((run_dir / "tables").glob("*.csv"))
    figure_paths = sorted((run_dir / "figures").rglob("*.png"))
    export_paths = sorted((run_dir / "exports").glob("*"))

    tables = "\n".join(_table_section(path) for path in table_paths)
    figures = "\n".join(
        f'<figure><img src="{escape(path.relative_to(run_dir).as_posix())}" alt="{escape(path.stem)}">'
        f'<figcaption>{escape(path.stem)}</figcaption></figure>'
        for path in figure_paths
    )
    exports = "\n".join(
        f'<li><a href="{escape(path.relative_to(run_dir).as_posix())}">{escape(path.name)}</a></li>'
        for path in export_paths
    )

    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#17202a}}header{{padding:28px 5%;background:#17202a;color:white}}
main{{padding:24px 5%;max-width:1500px;margin:auto}}section{{background:white;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 1px 5px #0002;overflow:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}figure{{background:white;border-radius:12px;padding:14px;margin:0;box-shadow:0 1px 5px #0002}}img{{max-width:100%;height:auto;display:block;margin:auto}}figcaption{{padding-top:8px;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#eef2f5}}a{{color:#0b66c3}}
</style></head><body><header><h1>{escape(title)}</h1><p>Centro local de métricas, tablas, gráficos y exportaciones.</p></header><main>
<section><h2>Exportaciones</h2><ul>{exports or '<li>Sin exportaciones.</li>'}</ul></section>
<h2>Gráficos</h2><div class="grid">{figures or '<p>Sin gráficos.</p>'}</div>
<h2>Tablas</h2>{tables or '<section><p>Sin tablas.</p></section>'}
</main></body></html>"""
    destination = run_dir / "dashboard.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def open_dashboard(run_dir: Path) -> Path:
    dashboard = build_dashboard(run_dir)
    webbrowser.open(dashboard.resolve().as_uri())
    return dashboard
