from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    rmse: float
    mae: float
    median_ae: float
    explained_variance: float
    bias: float = 0.0
    p90_ae: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    if y_true.ndim!=1 or y_true.shape!=y_pred.shape or not y_true.size or not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Regression requires aligned nonempty finite one-dimensional observations")
    has_variability=len(y_true)>1 and np.ptp(y_true)>0
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred,force_finite=False)) if has_variability else float("nan"),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        median_ae=float(median_absolute_error(y_true, y_pred)),
        explained_variance=float(explained_variance_score(y_true, y_pred,force_finite=False)) if has_variability else float("nan"),
        bias=float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        p90_ae=float(np.quantile(np.abs(np.asarray(y_true) - np.asarray(y_pred)), 0.90)),
    )
