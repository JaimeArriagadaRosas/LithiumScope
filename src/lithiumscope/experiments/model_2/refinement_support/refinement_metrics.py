"""Ranking metrics with row, signature, and whole-group perspectives."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from lithiumscope.experiments.generalization.signature_policy import signature_weights


def ranking(y, scores, x, groups):
    y=np.asarray(y,int)
    scores=np.asarray(scores,float)
    if set(np.unique(y))!={0,1} or not np.isfinite(scores).all() or scores.shape!=y.shape:
        raise ValueError("Ranking requires aligned finite scores and both classes")
    weights=signature_weights(x)
    group_sizes=groups.map(groups.value_counts()).to_numpy(float)
    return {"roc_auc":float(roc_auc_score(y,scores)),
            "ap":float(average_precision_score(y,scores)),
            "signature_auc":float(roc_auc_score(y,scores,sample_weight=weights)),
            "signature_ap":float(average_precision_score(y,scores,sample_weight=weights)),
            "group_weighted_auc":float(roc_auc_score(y,scores,sample_weight=1/group_sizes)),
            "group_weighted_ap":float(average_precision_score(y,scores,sample_weight=1/group_sizes))}


def summarize(folds):
    metrics=("roc_auc","ap","signature_auc","signature_ap","group_weighted_auc","group_weighted_ap")
    report={name:float(folds[name].mean()) for name in metrics}
    report.update(worst_fold_auc=float(folds.roc_auc.min()),
        worst_fold_ap=float(folds.ap.min()),mean_train_auc=float(folds.train_roc_auc.mean()),
        train_test_auc_gap=float((folds.train_roc_auc-folds.roc_auc).mean()),
        auc_std_across_folds=float(folds.roc_auc.std(ddof=0)))
    return report
