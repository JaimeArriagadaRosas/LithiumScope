from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_COORDINATE_PAIRS = (
    ("Longitude", "Latitude"),
    ("Longitude (X)", "Latitude (Y)"),
    ("longitude", "latitude"),
    ("Logintude (X)", "Latitude (Y)"),
)


def _coordinate_pair(frame: pd.DataFrame) -> tuple[str, str] | None:
    for longitude, latitude in _COORDINATE_PAIRS:
        if longitude in frame.columns and latitude in frame.columns:
            return longitude, latitude
    return None


def _spatial_frame(frame: pd.DataFrame) -> pd.DataFrame:
    pair = _coordinate_pair(frame)
    if pair is None:
        return pd.DataFrame()

    longitude, latitude = pair
    columns = [longitude, latitude]
    for candidate in (
        "case_id",
        "Li_icpms",
        "Li_icpms_predicted",
        "prospectivity_score",
    ):
        if candidate in frame.columns:
            columns.append(candidate)

    result = frame[columns].copy()
    result = result.rename(
        columns={
            longitude: "longitude",
            latitude: "latitude",
        }
    )
    result["longitude"] = pd.to_numeric(
        result["longitude"],
        errors="coerce",
    )
    result["latitude"] = pd.to_numeric(
        result["latitude"],
        errors="coerce",
    )
    return result.dropna(
        subset=["longitude", "latitude"]
    )


def _configure_axis(axis, frame: pd.DataFrame) -> None:
    axis.set_xlabel("Longitud")
    axis.set_ylabel("Latitud")
    axis.grid(True, alpha=0.25)
    if not frame.empty:
        mean_latitude = float(frame["latitude"].mean())
        cosine = math.cos(math.radians(mean_latitude))
        if abs(cosine) > 1e-6:
            axis.set_aspect(
                1.0 / cosine,
                adjustable="datalim",
            )


def _annotate_cases(axis, frame: pd.DataFrame) -> None:
    if "case_id" not in frame.columns:
        return
    for _, row in frame.iterrows():
        axis.annotate(
            str(row["case_id"]),
            (row["longitude"], row["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def save_sample_location_map(
    frame: pd.DataFrame,
    destination: Path,
) -> Path | None:
    values = _spatial_frame(frame)
    if values.empty:
        return None

    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    color_column = (
        "Li_icpms"
        if "Li_icpms" in values.columns
        else None
    )
    if color_column is not None:
        colors = pd.to_numeric(
            values[color_column],
            errors="coerce",
        )
        points = axis.scatter(
            values["longitude"],
            values["latitude"],
            c=colors,
            s=58,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.35,
        )
        colorbar = figure.colorbar(points, ax=axis)
        colorbar.set_label("Li real (ppm)")
    else:
        axis.scatter(
            values["longitude"],
            values["latitude"],
            s=58,
            edgecolors="black",
            linewidths=0.35,
        )

    _annotate_cases(axis, values)
    _configure_axis(axis, values)
    axis.set_title("Ubicacion geografica de las muestras evaluadas")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def save_model_result_map(
    frame: pd.DataFrame,
    destination: Path,
) -> Path | None:
    values = _spatial_frame(frame)
    required = {
        "Li_icpms_predicted",
        "prospectivity_score",
    }
    if values.empty or not required <= set(values.columns):
        return None

    values["Li_icpms_predicted"] = pd.to_numeric(
        values["Li_icpms_predicted"],
        errors="coerce",
    )
    values["prospectivity_score"] = pd.to_numeric(
        values["prospectivity_score"],
        errors="coerce",
    )
    values = values.dropna(
        subset=[
            "Li_icpms_predicted",
            "prospectivity_score",
        ]
    )
    if values.empty:
        return None

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        sharex=True,
        sharey=True,
    )

    m1 = axes[0].scatter(
        values["longitude"],
        values["latitude"],
        c=values["Li_icpms_predicted"],
        s=58,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
    )
    figure.colorbar(
        m1,
        ax=axes[0],
        label="Li estimado M1 (ppm)",
    )
    axes[0].set_title("Modelo 1")

    m2 = axes[1].scatter(
        values["longitude"],
        values["latitude"],
        c=values["prospectivity_score"],
        s=58,
        cmap="plasma",
        vmin=0.0,
        vmax=max(
            1.0,
            float(values["prospectivity_score"].max()),
        ),
        edgecolors="black",
        linewidths=0.35,
    )
    figure.colorbar(
        m2,
        ax=axes[1],
        label="Score M2",
    )
    axes[1].set_title("Modelo 2")

    for axis in axes:
        _annotate_cases(axis, values)
        _configure_axis(axis, values)

    figure.suptitle(
        "Resultados espaciales de los modelos por muestra"
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def save_spatial_maps(
    paired: pd.DataFrame,
    destination_dir: Path,
) -> list[Path]:
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    outputs = (
        save_sample_location_map(
            paired,
            destination_dir / "sample_locations.png",
        ),
        save_model_result_map(
            paired,
            destination_dir / "model_results_map.png",
        ),
    )
    return [
        path
        for path in outputs
        if path is not None
    ]
