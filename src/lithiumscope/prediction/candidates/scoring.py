from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.model_2.prediction.predictor import (
    score_model_2_image,
)


def score_model_2_cases(
    cases: pd.DataFrame,
    *,
    bundle: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in cases.iterrows():
        if str(
            row.get(
                "sentinel_status",
                "",
            )
        ) != "ready":
            continue

        payload, _ = score_model_2_image(
            Path(
                str(
                    row["image_path"]
                )
            ),
            bundle=bundle,
        )
        rows.append(
            {
                "case_id": str(
                    row["case_id"]
                ),
                "Li_icpms": row.get(
                    "Li_icpms"
                ),
                **payload,
                "sentinel_scene_id": row.get(
                    "sentinel_scene_id"
                ),
                "sentinel_cloud_cover": row.get(
                    "sentinel_cloud_cover"
                ),
                "sentinel_datetime": row.get(
                    "sentinel_datetime"
                ),
            }
        )
    return pd.DataFrame(rows)
