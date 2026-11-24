"""Preserve historical formulas; new preparation rejects invalid operands."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.logging.session import get_logger

logger = get_logger("model_1.feature_engineering")
_EPSILON = 1e-6


def _numeric(frame: pd.DataFrame, column: str, *, strict: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.ge(0)) if strict else values


def _ratio(numerator: pd.Series, denominator: pd.Series, *, strict: bool) -> pd.Series:
    if strict:
        denominator = denominator.where(np.isfinite(denominator) & denominator.gt(_EPSILON))
        values = numerator / denominator
        return values.where(np.isfinite(values))
    return numerator / (denominator + _EPSILON)


def add_geochemical_features(frame: pd.DataFrame, *, version: int = 1) -> pd.DataFrame:
    if version not in {1, 2}:
        raise ValueError("Unknown derived feature contract")
    strict = version == 2
    result = frame.copy()
    na2o, k2o, mgo, fe2o3, al2o3, cao = (
        _numeric(result, name, strict=strict) for name in ("Na2O", "K2O", "MgO", "Fe2O3", "Al2O3", "CaO")
    )
    result["Alkali_Sum"] = na2o + k2o
    result["Mg_Number"] = _ratio(mgo, mgo + fe2o3, strict=strict)
    result["A_CNK_proxy"] = _ratio(al2o3, cao + na2o + k2o, strict=strict)
    result["K_Mg_ratio"] = _ratio(k2o, mgo, strict=strict)
    names = ["Alkali_Sum", "Mg_Number", "A_CNK_proxy", "K_Mg_ratio"]
    if strict:
        for name in names:
            result[name] = result[name].where(np.isfinite(result[name]))
    logger.info("Derived feature contract=%d; missing=%s", version, result[names].isna().sum().to_dict())
    return result
