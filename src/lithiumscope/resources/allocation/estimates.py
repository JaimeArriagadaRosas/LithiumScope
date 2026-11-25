"""Conservative planning estimates, not measured estimator peak-memory guarantees."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from lithiumscope.resources.allocation.policy import Workload


@dataclass(frozen=True)
class FitDemand:
    algorithm: str
    rows: int
    raw_features: int
    encoded_features_upper_bound: int
    matrix_bytes: int
    working_memory_estimate_bytes: int
    planned_inner_fits_upper_bound: int
    planned_outer_fits: int
    execution_policy: str = "sequential_outer_and_trials_bounded_native_threads"

    def workload(self):
        # Canonical runners are sequential. Never count hypothetical fold workers as active.
        return Workload(tasks=1, memory_per_task_bytes=self.working_memory_estimate_bytes)

    def to_dict(self):
        return asdict(self)


def estimate_fit_demand(x, algorithm, *, folds, inner_folds, trials, optimize,
                        categorical=(), seed=None):
    del seed
    rows, columns = x.shape
    if not rows or not columns or any(type(v) is not int or v < 1 for v in (folds,inner_folds,trials)):
        raise ValueError("Invalid fit workload shape or counts")
    if not isinstance(optimize,bool):
        raise ValueError("Optimization flag must be explicit")
    categories = tuple(categorical)
    if len(set(categories)) != len(categories) or not set(categories) <= set(x.columns):
        raise ValueError("Invalid categorical workload schema")
    # Bound one-hot levels using observed counts plus missing/unknown columns.
    encoded = columns-len(categories)+sum(int(x[c].nunique(dropna=True))+2 for c in categories)
    matrix = max(rows*encoded*8, int(x.memory_usage(index=True,deep=True).sum()))
    # Include copies, encodings and representative algorithm workspace. The
    # estimate is a planning floor, not a proof about a fitted tree or GPU.
    factors = {"random_forest":12,"extra_trees":12,"random_forest_spatial_calibrated":16,
               "xgboost":12,"catboost":16,"hist_gradient_boosting":10,"tabnet":20,
               "svm":8,"svm_rbf":8}
    if algorithm not in factors:
        raise ValueError("Unknown workload family")
    estimate = max(32*1024**2, math.ceil(matrix*factors[algorithm]))
    if algorithm in {"svm","svm_rbf"}:
        # Conservative upper bound for kernel working set before cache policy.
        estimate += min(rows*rows*8,200*1024**2)
    return FitDemand(algorithm,rows,columns,encoded,matrix,estimate,
        (folds+1)*inner_folds*trials if optimize else 0,folds)
