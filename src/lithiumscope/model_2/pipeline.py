from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.model_2.steps.step_07_build_training_set import build_training_set


def load_model_2_training_frame(path: Path | None = None) -> pd.DataFrame:
    cfg = load_config("model_2")
    training_cfg = cfg["training"]
    manifest_path = path or Path(training_cfg["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model 2 manifest not found: {manifest_path}. "
            "Debe contener una fila por muestra conocida y una columna image_path."
        )
    manifest = pd.read_csv(manifest_path)
    return build_training_set(
        manifest,
        lithium_column=str(training_cfg["lithium_column"]),
        spatial_group_column=str(training_cfg.get("spatial_group_column", "spatial_group")),
    )
