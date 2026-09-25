from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery


def _rgb_preview(path: Path) -> np.ndarray | None:
    image = load_imagery(path)
    band_names = [
        str(value).strip().lower()
        for value in load_config("model_2")["imagery"]["bands"]
    ]
    required = ("red", "green", "blue")
    if any(name not in band_names for name in required):
        return None

    indices = [band_names.index(name) for name in required]
    if max(indices) >= image.shape[0]:
        return None

    rgb = np.moveaxis(image[indices, :, :], 0, -1)
    rendered = np.zeros(rgb.shape, dtype=np.float32)
    for channel in range(3):
        band = rgb[:, :, channel]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        rendered[:, :, channel] = np.clip(
            (band - low) / (high - low),
            0.0,
            1.0,
        )
    return rendered


def _satellite_preview_grid_shape(
    count: int,
) -> tuple[int, int]:
    columns = min(4, max(1, count))
    rows = (count + columns - 1) // columns
    return rows, columns


def save_satellite_input_preview(
    cases: pd.DataFrame,
    destination: Path,
) -> Path | None:
    required = {"case_id", "image_path"}
    if not required <= set(cases.columns):
        return None

    work = cases.copy()
    if "sentinel_status" in work.columns:
        work = work[
            work["sentinel_status"].astype(str) == "ready"
        ]
    work = work[
        work["image_path"].notna()
    ].head(12)
    if work.empty:
        return None

    previews: list[tuple[str, np.ndarray]] = []
    for _, row in work.iterrows():
        path = Path(str(row["image_path"]))
        if not path.is_file():
            continue
        preview = _rgb_preview(path)
        if preview is not None:
            previews.append(
                (str(row["case_id"]), preview)
            )

    if not previews:
        return None

    count = len(previews)
    rows, columns = _satellite_preview_grid_shape(count)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3.0, rows * 2.6),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, (case_id, preview) in zip(
        flat_axes,
        previews,
        strict=False,
    ):
        axis.imshow(preview)
        axis.set_title(str(case_id), fontsize=9)
        axis.axis("off")
    for axis in flat_axes[count:]:
        axis.axis("off")

    figure.suptitle(
        "Entradas Sentinel-2 utilizadas por Modelo 2",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


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
