"""Read, validate and score saved out-of-fold predictions without fitting."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
)


def get_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),
                         encoding="utf-8")
    temporary.replace(path)


def metrics(labels,probabilities):
    y=np.asarray(labels,int)
    score=np.asarray(probabilities,float)
    if len(y)==0 or len(y)!=len(score) or not np.isfinite(score).all():
        raise ValueError("Incomplete or nonfinite model predictions")
    if not np.isin(y,[0,1]).all():
        raise ValueError("Invalid binary labels")
    predicted=(score>=0.5).astype(int)
    return {
        "n":int(len(y)),"positives":int(y.sum()),
        "roc_auc":float(roc_auc_score(y,score)) if len(set(y))==2 else None,
        "average_precision":float(average_precision_score(y,score)) if y.sum() else None,
        "brier":float(brier_score_loss(y,score)),
        "balanced_accuracy_at_0p5":float(balanced_accuracy_score(y,predicted))
            if len(set(y))==2 else None,
        "precision_at_0p5":float(precision_score(y,predicted,zero_division=0)),
        "recall_at_0p5":float(recall_score(y,predicted,zero_division=0)),
        "f1_at_0p5":float(f1_score(y,predicted,zero_division=0)),
    }


def load_predictions(manifest,tables):
    if manifest.empty or "Li_icpms" not in manifest:
        raise ValueError("Expected a nonempty M2 manifest with Li_icpms")
    scores={}
    common_fold=None
    common_y=None
    for path in sorted(Path(tables).glob("oof_predictions_*.csv")):
        name=path.stem.removeprefix("oof_predictions_")
        df=pd.read_csv(path)
        required={"y_true","li_ppm","outer_fold","prospectivity_score"}
        if not required.issubset(df) or len(df)!=len(manifest):
            raise ValueError("Incomplete OOF for "+name)
        if "sample_id" in df and "sample_id" in manifest:
            if not np.array_equal(df.sample_id.astype(str),manifest.sample_id.astype(str)):
                raise ValueError("OOF sample identity differs for "+name)
        if not np.allclose(df.li_ppm,manifest.Li_icpms):
            raise ValueError("OOF geochemistry differs for "+name)
        fold_values=df.outer_fold.to_numpy(float)
        y_values=df.y_true.to_numpy(float)
        if not (np.isfinite(fold_values).all() and np.equal(fold_values,np.floor(fold_values)).all()
                and np.isin(y_values,[0,1]).all()):
            raise ValueError("Invalid OOF for "+name)
        folds=fold_values.astype(int)
        y=y_values.astype(int)
        probabilities=df.prospectivity_score.to_numpy(float)
        if not (np.array_equal(np.unique(folds),np.arange(1,int(folds.max())+1)) and
                np.isin(y,[0,1]).all() and np.isfinite(probabilities).all() and
                ((probabilities>=0)&(probabilities<=1)).all()):
            raise ValueError("Invalid OOF for "+name)
        if common_y is not None and (
            not np.array_equal(y,common_y) or
            not np.array_equal(folds,common_fold)):
            raise ValueError("Algorithms disagree on held-out fold labels or indices")
        common_y=y
        common_fold=folds
        scores[name]=probabilities
    if not scores:
        raise ValueError("No complete OOF predictions saved by competition")
    return scores,common_fold,common_y
