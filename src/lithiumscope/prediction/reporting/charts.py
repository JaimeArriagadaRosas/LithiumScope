from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    destination: Path,
) -> Path | None:
    if x not in frame.columns or y not in frame.columns:
        return None
    values = (
        frame[[x, y]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
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
    if (
        "case_id" not in frame.columns
        or value_column not in frame.columns
    ):
        return None
    values = frame[["case_id", value_column]].copy()
    values[value_column] = pd.to_numeric(
        values[value_column],
        errors="coerce",
    )
    values = values.dropna()
    if values.empty:
        return None

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        values["case_id"].astype(str),
        values[value_column],
    )
    for threshold in horizontal_lines:
        axis.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
        )
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
    if {
        "Li_icpms",
        "Li_icpms_predicted",
    } <= set(work.columns):
        work["model_1_absolute_error"] = (
            pd.to_numeric(
                work["Li_icpms"],
                errors="coerce",
            )
            - pd.to_numeric(
                work["Li_icpms_predicted"],
                errors="coerce",
            )
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
            destination=(
                figure_dir
                / "model_1_absolute_error_by_case.png"
            ),
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
            destination=(
                figure_dir
                / "model_2_score_by_case.png"
            ),
            horizontal_lines=(0.4, 0.7),
        ),
        _case_bar(
            work,
            value_column=(
                "model_1_out_of_training_range_fraction"
            ),
            title="Fraccion OOD del Modelo 1 por caso",
            ylabel=(
                "Fraccion fuera del dominio de entrenamiento"
            ),
            destination=(
                figure_dir
                / "model_1_ood_by_case.png"
            ),
        ),
        _case_bar(
            work,
            value_column=(
                "model_2_out_of_training_range_fraction"
            ),
            title="Fraccion OOD del Modelo 2 por caso",
            ylabel=(
                "Fraccion fuera del dominio de entrenamiento"
            ),
            destination=(
                figure_dir
                / "model_2_ood_by_case.png"
            ),
        ),
        _scatter(
            work,
            "model_1_out_of_training_range_fraction",
            "model_2_out_of_training_range_fraction",
            "OOD Modelo 1 vs. OOD Modelo 2",
            figure_dir / "model_1_vs_model_2_ood.png",
        ),
    ]
    return [
        path
        for path in outputs
        if path is not None
    ]
