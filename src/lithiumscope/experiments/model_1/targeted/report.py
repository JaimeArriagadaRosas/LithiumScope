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
    rows, intervals, strata, reference_rows = [], [], [], []
    for algorithm, table in reference.items():
        metric = regression_metrics(table.truth, table.prediction).to_dict()
        by_fold = [regression_metrics(g.truth,g.prediction).to_dict() for _,g in table.groupby("fold")]
        reference_rows.append({"algorithm":algorithm,"rows":len(table),
                               "mean_fold_rmse":float(np.mean([v["rmse"] for v in by_fold])),
                               "mean_fold_mae":float(np.mean([v["mae"] for v in by_fold])),
                               "worst_fold_rmse":max(v["rmse"] for v in by_fold),
                               **{"global_"+k:v for k,v in metric.items()}})
    for (algorithm, policy), table in predictions.groupby(["algorithm","policy"], sort=False):
        if len(table) != expected or table.position.duplicated().any():
            continue
        subset = folds[(folds.algorithm == algorithm) & (folds.policy == policy)]
        global_metrics = regression_metrics(table.truth, table.prediction).to_dict()
        outside = table[table.actual_above_train_max]
        in_support = table[~table.actual_above_train_max]
        for dimension, grouped in (("source",table.groupby("source")),("group",table.groupby("group")),
                                    ("target_support",table.groupby("actual_above_train_max"))):
            for name, group in grouped:
                values = regression_metrics(group.truth,group.prediction).to_dict()
                strata.append({"algorithm":algorithm,"policy":policy,"dimension":dimension,
                               "stratum":name,"rows":len(group),**values})
        rows.append({"algorithm":algorithm, "policy":policy, "rows":len(table),
                     "mean_fold_rmse":float(subset.rmse.mean()), "mean_fold_mae":float(subset.mae.mean()),
                     "mean_fold_r2":float(subset.r2.mean()), "worst_fold_rmse":float(subset.rmse.max()),
                     "mean_train_r2":float(subset.train_r2.mean()),
                     "mean_train_test_r2_gap":float((subset.train_r2-subset.r2).mean()),
                     "above_train_max_rows":len(outside),
                     "above_train_max_rmse":regression_metrics(outside.truth,outside.prediction).rmse if len(outside) else None,
                     "within_train_max_rmse":regression_metrics(in_support.truth,in_support.prediction).rmse if len(in_support) else None, "high_mae":float(subset.high_mae.mean()),
                     "high_bias":float(subset.high_bias.mean()),
                     "negative_predictions":int(subset.negative_predictions.sum()),
                     **{"global_"+k:v for k,v in global_metrics.items()}})
        for item in paired_group_intervals(reference[algorithm], table, kind="regression", repeats=500):
            intervals.append({"algorithm":algorithm,"policy":policy,**item})
    write_table(destination/"summary.csv", pd.DataFrame(rows))
    write_table(destination/"reference_summary.csv",pd.DataFrame(reference_rows))
    write_table(destination/"stratified_metrics.csv",pd.DataFrame(strata))
    write_table(destination/"paired_intervals.csv", pd.DataFrame(intervals))
    return rows
