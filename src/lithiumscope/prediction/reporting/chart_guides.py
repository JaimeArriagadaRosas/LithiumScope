"""Scientifically labeled visual guides for the report's charts.

Identity lines are used only between quantities sharing units; regression
lines are descriptive and fitted to the plotted run, not model predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


AXIS_LABELS = {
    "Li_icpms": "Li medido (ppm)",
    "Li_icpms_predicted": "Li estimado M1 (ppm)",
    "prospectivity_score": "Score espectral M2 (relativo)",
    "model_1_out_of_training_range_fraction": "Fracción OOD M1",
    "model_2_out_of_training_range_fraction": "Fracción OOD M2",
}


def _trend(ax, x: np.ndarray, y: np.ndarray) -> None:
    """OLS descriptive fit; decline to draw with insufficient variability."""
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite([slope, intercept]).all():
        return
    support = np.linspace(float(x.min()), float(x.max()), 100)
    fitted = slope * support + intercept
    corr = float(np.corrcoef(x, y)[0, 1])
    label = f"Tendencia lineal (r={corr:.2f}, n={len(x)})"
    ax.plot(support, fitted, linestyle="-", linewidth=1.7, label=label)


def scatter_guides(ax, values: pd.DataFrame, *, x: str, y: str) -> None:
    """Overlay only references justified by axes meaning and units."""
    xx = values[x].to_numpy(float)
    yy = values[y].to_numpy(float)
    ax.set_xlabel(AXIS_LABELS.get(x, x))
    ax.set_ylabel(AXIS_LABELS.get(y, y))

    lithium_identity = x == "Li_icpms" and y == "Li_icpms_predicted"
    ood_identity = (
        x == "model_1_out_of_training_range_fraction"
        and y == "model_2_out_of_training_range_fraction"
    )
    if lithium_identity or ood_identity:
        low = 0.0 if ood_identity else float(min(xx.min(), yy.min()))
        high = float(max(xx.max(), yy.max()))
        if ood_identity:
            high = max(0.30, high)
        if high <= low:
            high = low + 1.0
        ax.plot(
            [low, high], [low, high],
            linestyle="--", linewidth=1.4,
            label="Identidad (y = x)",
        )
        if lithium_identity:
            margin = max(0.5, (high - low) * 0.05)
            ax.set_xlim(low - margin, high + margin)
            ax.set_ylim(low - margin, high + margin)
        else:
            ax.set_xlim(0.0, high * 1.08)
            ax.set_ylim(0.0, high * 1.08)
            ax.axvline(0.25, linestyle=":", linewidth=1.2,
                       label="Corte OOD (25 %)")
            ax.axhline(0.25, linestyle=":", linewidth=1.2)
    if not ood_identity:
        _trend(ax, xx, yy)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="best")


def bar_guides(
    ax,
    values: pd.Series,
    *,
    thresholds: tuple[float, ...] = (),
    is_ood: bool = False,
) -> None:
    """Actual observed mean plus only applicable operational/descriptive cuts."""
    numbers = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(numbers):
        mean = float(np.mean(numbers))
        if np.isfinite(mean):
            ax.axhline(
                mean, linestyle="-.", linewidth=1.5,
                label=f"Media ({mean:.3f})",
            )
    if is_ood:
        ax.axhline(0.25, linestyle="--", linewidth=1.3,
                   label="Corte OOD (25 %)")
    else:
        for value in thresholds:
            if value in (0.4, 0.7):
                name = f"Corte descriptivo M2 ({value:.2f})"
            else:
                name = f"Referencia ({value:.2f})"
            ax.axhline(value, linestyle=":", linewidth=1.2, label=name)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="best")
