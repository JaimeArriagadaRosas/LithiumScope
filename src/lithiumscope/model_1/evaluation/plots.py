from __future__ import annotations

from pathlib import Path

from lithiumscope.core.visualization import configure_headless_matplotlib

configure_headless_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_real_vs_predicted(y_true, y_pred, destination: Path, title: str = "Real vs. predicho") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.65)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Li_icpms real (ppm)")
    ax.set_ylabel("Li_icpms predicho (ppm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_residuals(y_true, y_pred, destination: Path, title: str = "Residuos") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.65)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Li predicho (ppm)")
    ax.set_ylabel("Residual (real - predicho)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_target_distribution(values, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30)
    ax.set_xlabel("Li_icpms (ppm)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de la variable objetivo")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_missingness(frame: pd.DataFrame, destination: Path, top_n: int = 25) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    missing = frame.isna().mean().sort_values(ascending=False).head(top_n) * 100.0
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(missing.index[::-1], missing.values[::-1])
    ax.set_xlabel("Datos faltantes (%)")
    ax.set_title(f"Top {top_n} variables con datos faltantes")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_pipeline_rows(audit: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(audit["stage"], audit["rows"], marker="o")
    ax.set_ylabel("Número de muestras")
    ax.set_title("Muestras disponibles después de cada etapa")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_correlation_heatmap(frame: pd.DataFrame, target: str, destination: Path, top_n: int = 18) -> Path | None:
    numeric = frame.select_dtypes(include=[np.number])
    if target not in numeric.columns or numeric.shape[1] < 2:
        return None
    correlation = numeric.corr(numeric_only=True)
    selected = correlation[target].abs().sort_values(ascending=False).head(top_n).index
    matrix = correlation.loc[selected, selected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected)), selected, rotation=70, ha="right")
    ax.set_yticks(range(len(selected)), selected)
    ax.set_title("Correlaciones numéricas — variables más relacionadas con Li")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Correlación")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_competition_chart(table: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("rmse_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ordered["label"], ordered["rmse_mean"], xerr=ordered["rmse_std"])
    ax.set_xlabel("RMSE outer CV (ppm, menor es mejor)")
    ax.set_title("Competencia Modelo 1")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
