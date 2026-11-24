"""Render descriptive EDA figures from calculated evidence."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _figure():
    fig = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def save_figures(directory: Path, frame: pd.DataFrame, target: str,
                 profiles: pd.DataFrame, correlations: pd.DataFrame) -> None:
    numeric = pd.to_numeric(frame[target], errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    fig, ax = _figure()
    ax.hist(numeric, bins=30)
    ax.set(xlabel="Litio (ppm)", ylabel="Muestras", title="Distribución de concentraciones")
    fig.savefig(directory / "lithium_distribution.png", dpi=140)
    if len(numeric) and (numeric >= 0).all():
        fig, ax = _figure()
        ax.hist(np.log1p(numeric), bins=30)
        ax.set(xlabel="log(1 + Li ppm)", ylabel="Muestras", title="Distribución logarítmica descriptiva")
        fig.savefig(directory / "lithium_distribution_log.png", dpi=140)

    if not profiles.empty:
        selected = profiles.groupby("feature")[["missing", "rows"]].sum()
        missing = (selected["missing"] / selected["rows"]).nlargest(20).sort_values()
        fig, ax = _figure()
        ax.barh(missing.index, missing.values * 100)
        ax.set(xlabel="Valores faltantes (%)", title="Disponibilidad de características")
        fig.savefig(directory / "feature_missingness.png", dpi=140)

    if target in correlations:
        values = correlations[target].drop(labels=[target], errors="ignore").dropna()
        selected = values.reindex(values.abs().nlargest(15).index).sort_values()
        if len(selected):
            fig, ax = _figure()
            ax.barh(selected.index, selected.values)
            ax.set(xlabel="Correlación de Spearman", xlim=(-1, 1),
                   title="Asociaciones descriptivas con Li")
            fig.savefig(directory / "target_correlations.png", dpi=140)

    longitude = next((name for name in ("Longitude", "longitude") if name in frame), None)
    latitude = next((name for name in ("Latitude", "latitude") if name in frame), None)
    if longitude and latitude:
        lon = pd.to_numeric(frame[longitude], errors="coerce")
        lat = pd.to_numeric(frame[latitude], errors="coerce")
        values = pd.to_numeric(frame[target], errors="coerce")
        valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
        if valid.any():
            fig, ax = _figure()
            scatter = ax.scatter(lon[valid], lat[valid], c=values[valid], s=10)
            fig.colorbar(scatter, ax=ax, label="Li (ppm)")
            ax.set(xlabel="Longitud", ylabel="Latitud", title="Ubicaciones observadas (no interpolación)")
            fig.savefig(directory / "sample_locations.png", dpi=140)
