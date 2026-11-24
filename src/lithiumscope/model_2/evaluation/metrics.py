"""Canonical binary metrics; invalid inputs cannot become favourable scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def validate_binary_inputs(y_true, probabilities, *, require_both: bool = True):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Binary labels and scores must have the same nonempty 1D shape")
    if not np.isfinite(y).all() or not np.isin(y,[0,1]).all():
        raise ValueError("Binary labels must be exactly 0 or 1")
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError("Scores must be finite within [0,1]")
    if require_both and np.unique(y).size != 2:
        raise ValueError("Both classes are required for a comparable classification evaluation")
    return y.astype(int),p


def reliability_table(y_true, probabilities, *, bins: int = 10) -> pd.DataFrame:
    y,p = validate_binary_inputs(y_true,probabilities,require_both=False)
    if not isinstance(bins,int) or bins<2:
        raise ValueError("Reliability bins must be an integer >=2")
    edges = np.linspace(0,1,bins+1)
    positions = np.minimum(np.searchsorted(edges,p,side="right")-1,bins-1)
    rows=[]
    for i in range(bins):
        mask=positions==i
        n=int(mask.sum())
        rows.append({"bin":i,"lower":float(edges[i]),"upper":float(edges[i+1]),"count":n,
                     "mean_score":float(p[mask].mean()) if n else np.nan,
                     "observed_positive_fraction":float(y[mask].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def classification_metrics(y_true, probabilities, threshold: float = 0.5,
                           *, allow_single_class: bool = False) -> dict[str,float]:
    y,p=validate_binary_inputs(y_true,probabilities,require_both=not allow_single_class)
    threshold=float(threshold)
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError("Operating threshold must be finite within [0,1]")
    predictions=(p>=threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,predictions,labels=[0,1]).ravel())
    sensitivity=tp/(tp+fn) if tp+fn else np.nan
    specificity=tn/(tn+fp) if tn+fp else np.nan
    both=bool(tp+fn and tn+fp)
    # With a single observed class, discrimination/BA/MCC/Kappa cannot support comparison.
    payload={"n":len(y),"threshold":threshold,"tn":tn,"fp":fp,"fn":fn,"tp":tp,
             "positive_support":tp+fn,"negative_support":tn+fp,"prevalence":float(y.mean()),
             "accuracy":float(accuracy_score(y,predictions)),
             "balanced_accuracy":float((sensitivity+specificity)/2) if both else float("nan"),
             "precision":float(precision_score(y,predictions,zero_division=0)),
             "recall":float(sensitivity),"sensitivity":float(sensitivity),
             "specificity":float(specificity),
             "f1":float(f1_score(y,predictions,zero_division=0)) if tp+fn else float("nan"),
             "false_positive_rate":float(1-specificity),"false_negative_rate":float(1-sensitivity),
             "roc_auc":float(roc_auc_score(y,p)) if both else float("nan"),
             "average_precision":float(average_precision_score(y,p)) if both else float("nan"),
             "brier_score":float(brier_score_loss(y,p)),
             "log_loss":float(log_loss(y,p,labels=[0,1])),
             "mcc":float(matthews_corrcoef(y,predictions)) if both else float("nan"),
             "kappa":float(cohen_kappa_score(y,predictions)) if both else float("nan")}
    # Count-weighted, fixed-width calibration error; descriptive, not calibration proof.
    rel=reliability_table(y,p)
    payload["ece_10"]=float(((rel.mean_score-rel.observed_positive_fraction).abs()*rel["count"]).sum()/len(y))
    return payload
