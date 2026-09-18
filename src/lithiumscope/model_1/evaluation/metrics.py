from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    rmse: float
    mae: float
    def to_dict(self) -> dict[str,float]: return asdict(self)

def regression_metrics(y_true,y_pred)->RegressionMetrics:
    return RegressionMetrics(r2=float(r2_score(y_true,y_pred)),rmse=float(np.sqrt(mean_squared_error(y_true,y_pred))),mae=float(mean_absolute_error(y_true,y_pred)))
