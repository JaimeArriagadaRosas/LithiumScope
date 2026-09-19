from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.logger import get_logger
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("model_2.training_set")


def build_training_set(
    manifest: pd.DataFrame,
    lithium_column: str = "Li_icpms",
    image_path_column: str = "image_path",
    spatial_group_column: str = "spatial_group",
    *,
    band_names: tuple[str, ...] | list[str],
    normalize_per_band: bool = False,
) -> pd.DataFrame:
    if lithium_column not in manifest.columns:
        raise ValueError(f"Missing lithium column: {lithium_column}")
    if image_path_column not in manifest.columns:
        raise ValueError(f"Missing image path column: {image_path_column}")

    rows: list[dict] = []
    total = len(manifest)
    spinner = Spinner(
        f"Extrayendo características espectrales: 0/{total}"
    ).start()

    try:
        for position, (_, sample) in enumerate(
            manifest.iterrows(),
            start=1,
        ):
            path = Path(sample[image_path_column])
            image = preprocess_image(
                load_imagery(path),
                normalize_per_band=normalize_per_band,
            )
            features = extract_spectral_features(
                image,
                band_names=band_names,
            )
            features[lithium_column] = float(sample[lithium_column])
            if spatial_group_column in manifest.columns:
                features[spatial_group_column] = sample[spatial_group_column]
            rows.append(features)

            spinner.update(
                "Extrayendo características espectrales: "
                f"{position}/{total}"
            )
            if position % 25 == 0 or position == total:
                logger.info(
                    "Model 2 spectral extraction progress=%d/%d",
                    position,
                    total,
                )
    except Exception:
        spinner.fail(
            "Falló la extracción de características espectrales"
        )
        raise

    if not rows:
        spinner.fail("No había pares imagen/muestra utilizables")
        raise ValueError(
            "No image/sample pairs were available to build Model 2."
        )

    spinner.succeed(
        f"Características espectrales listas: {len(rows)} muestras"
    )
    return pd.DataFrame(rows)
