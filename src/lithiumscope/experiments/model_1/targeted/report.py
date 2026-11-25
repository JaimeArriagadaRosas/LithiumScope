"""Descriptive paired evidence, not a release-selection gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.experiments.shared.paired_comparison import paired_group_intervals
from lithiumscope.results.atomic_exports import write_table


def fold_metrics(truth, scores, train_truth, train_scores):
    result = regression_metrics(truth, scores).to_dict()
    train = regression_metrics(train_truth, train_scores).to_dict()
    high = truth >= np.quantile(train_truth, .75)
    above = truth > np.max(train_truth)
    error = scores-truth
    return {**result, **{"train_"+k:v for k,v in train.items()},
            "high_mae":float(np.abs(error[high]).mean()) if high.any() else np.nan,
            "high_bias":float(error[high].mean()) if high.any() else np.nan,
            "above_train_max_count":int(above.sum()), "negative_predictions":int((scores<0).sum())}


def summarize(destination, folds, predictions, expected, reference):
    rows, intervals = [], []
    for (algorithm, policy), table in predictions.groupby(["algorithm","policy"], sort=False):
        if len(table) != expected or table.position.duplicated().any():
            continue
        subset = folds[(folds.algorithm == algorithm) & (folds.policy == policy)]
        global_metrics = regression_metrics(table.truth, table.prediction).to_dict()
        rows.append({"algorithm":algorithm, "policy":policy, "rows":len(table),
                     "mean_fold_rmse":float(subset.rmse.mean()), "mean_fold_mae":float(subset.mae.mean()),
                     "mean_fold_r2":float(subset.r2.mean()), "worst_fold_rmse":float(subset.rmse.max()),
                     "mean_train_r2":float(subset.train_r2.mean()), "high_mae":float(subset.high_mae.mean()),
                     "high_bias":float(subset.high_bias.mean()),
                     "negative_predictions":int(subset.negative_predictions.sum()),
                     **{"global_"+k:v for k,v in global_metrics.items()}})
        for item in paired_group_intervals(reference[algorithm], table, kind="regression", repeats=500):
            intervals.append({"algorithm":algorithm,"policy":policy,**item})
    write_table(destination/"summary.csv", pd.DataFrame(rows))
    write_table(destination/"paired_intervals.csv", pd.DataFrame(intervals))
    return rows
