"""Exploratory M2 train-only threshold selection for spatially calibrated RF."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.experiments.analysis.run_vnext_m2 import (
    _calibration_splits, score_metrics,
)
from lithiumscope.model_2.training.scientific_validation import make_scientific_folds


def threshold_by_training_prevalence(
    train_scores: np.ndarray, train_labels: np.ndarray,
) -> float:
    """Uses ONLY fitted-training scores and outer-training class prevalence."""
    p=np.asarray(train_scores,dtype=float)
    y=np.asarray(train_labels,dtype=int)
    if p.shape!=y.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid training score / label inputs")
    if len(np.unique(y))!=2:
        raise ValueError("Training labels must have both classes")
    return float(np.quantile(p, 1.0-float(y.mean())))


def run_threshold_experiment(
    root: Path = PROJECT_ROOT,
    destination: Path | None = None,
) -> dict:
    out=destination or (
        RESULTS_DIR/"scientific_simulations"/"vnext_20261106"/"stage3_m2"
    )
    out.mkdir(parents=True, exist_ok=True)
    data=pd.read_csv(root/"data/processed/model_2/sentinel_features.csv")
    bundle=joblib.load(
        root/"models/model_2/trained/training_20261105_223618_m0300/model.joblib"
    )
    x=data[list(bundle["features"])]
    groups=data.spatial_group.reset_index(drop=True)
    lithium=data.Li_icpms.reset_index(drop=True)
    folds=make_scientific_folds(lithium,groups,n_splits=5,quantile=.75)
    model=clone(bundle["estimator"])
    model.set_params(n_estimators=160,min_samples_leaf=4,max_depth=9,n_jobs=4)
    rows=[]
    for idx,fold in enumerate(folds,start=1):
        train=fold.train_idx
        test=fold.test_idx
        xtr=x.iloc[train].reset_index(drop=True)
        xte=x.iloc[test]
        ytr=np.asarray(fold.y_train,dtype=int)
        yte=np.asarray(fold.y_test,dtype=int)
        inner=_calibration_splits(
            xtr,ytr,groups.iloc[train].reset_index(drop=True)
        )
        if inner is None:
            rows.append({"fold":idx,"status":"NO_GROUPED_CALIBRATION"})
            continue
        calibrated=CalibratedClassifierCV(
            estimator=clone(model),method="sigmoid",cv=inner
        )
        calibrated.fit(xtr,ytr)
        fit_scores=calibrated.predict_proba(xtr)[:,1]
        operating=threshold_by_training_prevalence(fit_scores,ytr)
        prob=calibrated.predict_proba(xte)[:,1]
        original=score_metrics(yte,prob)
        extra={
            "balanced_accuracy_train_prevalence":float(
                balanced_accuracy_score(yte,prob>=operating)
            ),
            "f1_train_prevalence":float(
                f1_score(yte,prob>=operating,zero_division=0)
            ),
            "recall_train_prevalence":float(
                recall_score(yte,prob>=operating,zero_division=0)
            ),
        }
        rows.append({
            "fold":idx,"status":"ok","n_test":len(test),
            "label_threshold_ppm":fold.label_threshold_ppm,
            "operating_threshold_train_only":operating,
            "train_positive_rate":float(ytr.mean()),
            "balanced_accuracy_at_0_5":original["balanced_accuracy"],
            "roc_auc":original["roc_auc"],"ap":original["average_precision"],
            "brier":original["brier"],**extra,
        })
        print("M2_OPERATING_THRESHOLD",idx,
              "train_threshold",round(operating,4),
              "BA_0.5",round(original["balanced_accuracy"],3),
              "BA_train_rule",round(extra["balanced_accuracy_train_prevalence"],3),
              flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(out/"m2_calibrated_train_only_threshold.csv",index=False)
    good=result.loc[result.status=="ok"]
    summary={
        "scope":"development_only_not_independent",
        "n_folds":len(good),
        "mean_balanced_accuracy_at_0_5":float(
            good.balanced_accuracy_at_0_5.mean()
        ),
        "mean_balanced_accuracy_train_rule":float(
            good.balanced_accuracy_train_prevalence.mean()
        ),
        "mean_recall_train_rule":float(good.recall_train_prevalence.mean()),
        "mean_auc":float(good.roc_auc.mean()),
        "mean_brier":float(good.brier.mean()),
        "threshold_logic":"Train-only prevalence score quantile; no test labels",
        "warning":"Threshold fitted on train scores, can be optimistic, must verify independent later.",
    }
    (out/"m2_calibrated_threshold_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8'
    )
    return summary


if __name__=="__main__":
    print("CALIBRATED_THRESHOLD_RESULTS",run_threshold_experiment(),flush=True)
