from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.datasets.readers.imagery import read_multiband_array
from lithiumscope.core.logging.session import get_logger

logger = get_logger("reporting.figures.satellite")


def _rgb_preview(path: Path, bands=None) -> np.ndarray | None:
    image = np.asarray(read_multiband_array(path), dtype=np.float32)
    if image.ndim != 3:
        return None
    band_names = [
        str(value).strip().lower()
        for value in (bands if bands is not None else load_config("model_2")["imagery"]["bands"])
    ]
    if bands is not None and (image.shape[0] != len(band_names) or len(set(band_names)) != len(band_names)):
        return None
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
            (np.where(np.isfinite(band), band, low) - low) / (high - low),
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
    *, bands=None,
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
        try:
            preview = _rgb_preview(path, bands)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Satellite preview omitted case=%s: %s", row["case_id"], exc)
            continue
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
    try:
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    logger.info("Satellite preview written: %s (%d cases)", destination, count)
    return destination
