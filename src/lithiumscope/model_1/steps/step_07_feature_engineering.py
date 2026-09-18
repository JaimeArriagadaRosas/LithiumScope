from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.feature_engineering")
_EPSILON = 1e-6


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def add_geochemical_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    na2o = _numeric(result, "Na2O")
    k2o = _numeric(result, "K2O")
    mgo = _numeric(result, "MgO")
    fe2o3 = _numeric(result, "Fe2O3")
    al2o3 = _numeric(result, "Al2O3")
    cao = _numeric(result, "CaO")

    result["Alkali_Sum"] = na2o + k2o
    result["Mg_Number"] = mgo / (mgo + fe2o3 + _EPSILON)
    result["A_CNK_proxy"] = al2o3 / (cao + na2o + k2o + _EPSILON)
    result["K_Mg_ratio"] = k2o / (mgo + _EPSILON)

    logger.info("Added derived features: Alkali_Sum, Mg_Number, A_CNK_proxy, K_Mg_ratio")
    return result
