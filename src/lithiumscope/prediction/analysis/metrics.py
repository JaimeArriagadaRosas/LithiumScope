from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.core.scientific.classification_metrics import classification_metrics


def model_1_external_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if "Li_icpms" not in frame.columns or "Li_icpms_predicted" not in frame.columns:
        return {}
    values = pd.DataFrame(
        {
            "truth": pd.to_numeric(frame["Li_icpms"], errors="coerce"),
            "prediction": pd.to_numeric(frame["Li_icpms_predicted"], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return {}
    return regression_metrics(values["truth"], values["prediction"]).to_dict()


def model_2_external_metrics(
    frame: pd.DataFrame,
    threshold_ppm: float,
    probability_threshold: float = 0.5,
) -> dict[str, float]:
    if "Li_icpms" not in frame.columns or "prospectivity_score" not in frame.columns:
        return {}
    values = pd.DataFrame(
        {
            "truth": pd.to_numeric(frame["Li_icpms"], errors="coerce"),
            "score": pd.to_numeric(frame["prospectivity_score"], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return {}
    y_true = (values["truth"].to_numpy(dtype=float) >= float(threshold_ppm)).astype(int)
    scores = values["score"].to_numpy(dtype=float)
    return classification_metrics(y_true, scores, threshold=float(probability_threshold), allow_single_class=True)


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
