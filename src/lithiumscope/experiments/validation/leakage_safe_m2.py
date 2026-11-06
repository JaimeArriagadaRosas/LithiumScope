"""Leakage-controlled grouped M2 research experiment; original release stays immutable."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

DATA=Path(r"D:\Descargas\LithiumScope\data\processed\model_2")
OUT=Path("results/scientific_simulations/leakage_safe_m2")

def threshold(y,s):
    candidates=np.unique(np.quantile(s,np.linspace(.05,.95,61)))
    best=(0.5, float("-inf"))
    for t in candidates:
        value=balanced_accuracy_score(y,s>=t)
        if value>best[1]:best=(float(t),float(value))
    return best[0]

def evaluate():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(DATA/"training_manifest.csv")
    features=pd.read_csv(DATA/"sentinel_features.csv")
    if len(manifest)!=len(features):raise ValueError("Feature/manifest length mismatch")
    x=features.drop(columns=["Li_icpms","spatial_group"],errors="ignore").select_dtypes(include="number").replace([np.inf,-np.inf],np.nan)
    x=x.fillna(0)
    li=manifest.Li_icpms.to_numpy(dtype=float)
    groups=manifest.spatial_group.astype(str).to_numpy()
    # Global q used ONLY to stratify outer folds, not to label their test samples.
    strata=(li>=np.quantile(li,.75)).astype(int)
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    predictions=[];splits=[]
    for fold,(train,test) in enumerate(outer.split(x,strata,groups),1):
        assert not(set(groups[train])&set(groups[test]))
        q=float(np.quantile(li[train],.75))
        ytr=(li[train]>=q).astype(int)
        yte=(li[test]>=q).astype(int)
        if len(set(ytr))!=2 or len(set(yte))!=2: raise ValueError("Single-class grouped fold")
        splitter=GroupKFold(n_splits=3)
        inner=np.full(len(train),np.nan)
        for inner_train,inner_valid in splitter.split(x.iloc[train],ytr,groups[train]):
            estimator=CatBoostClassifier(iterations=240,depth=5,learning_rate=.05,
                loss_function="Logloss",verbose=False,allow_writing_files=False,
                random_seed=42,thread_count=4)
            estimator.fit(x.iloc[train[inner_train]],ytr[inner_train])
            inner[inner_valid]=estimator.predict_proba(x.iloc[train[inner_valid]])[:,1]
        assert np.isfinite(inner).all()
        operating=threshold(ytr,inner)
        model=CatBoostClassifier(iterations=240,depth=5,learning_rate=.05,
            loss_function="Logloss",verbose=False,allow_writing_files=False,
            random_seed=42,thread_count=4)
        model.fit(x.iloc[train],ytr)
        p=model.predict_proba(x.iloc[test])[:,1]
        for j,score in zip(test,p):
            predictions.append(dict(sample_id=manifest.iloc[j].sample_id,fold=fold,
                y_true=int(li[j]>=q),li_ppm=float(li[j]),probability=float(score),
                threshold_ppm=q,operating_threshold=operating,
                y_pred=int(score>=operating),spatial_group=groups[j]))
        splits.append(dict(fold=fold,n_train=len(train),n_test=len(test),
             threshold_ppm=q,operating_threshold=operating,
             grouped_overlap=0,inner_grouped_folds=3,
             roc_auc=float(roc_auc_score(yte,p)),
             average_precision=float(average_precision_score(yte,p)),
             balanced_accuracy=float(balanced_accuracy_score(yte,p>=operating))))
        pd.DataFrame(predictions).to_csv(OUT/"predictions_partial.csv",index=False)
        pd.DataFrame(splits).to_csv(OUT/"fold_metrics_partial.csv",index=False)
        print("FOLD",fold,"RESULT",splits[-1],flush=True)
    result=pd.DataFrame(predictions)
    result.to_csv(OUT/"predictions.csv",index=False)
    summary=pd.DataFrame(splits)
    summary.to_csv(OUT/"fold_metrics.csv",index=False)
    assert len(result)==len(manifest)
    metrics={"rows":len(result),"spatial_groups":len(set(groups)),"folds":len(splits),
      "mean_fold_roc_auc":float(summary.roc_auc.mean()),
      "mean_fold_average_precision":float(summary.average_precision.mean()),
      "mean_fold_balanced_accuracy":float(summary.balanced_accuracy.mean()),
      "dataset_sha256":hashlib.sha256((DATA/"training_manifest.csv").read_bytes()).hexdigest(),
      "features_sha256":hashlib.sha256((DATA/"sentinel_features.csv").read_bytes()).hexdigest(),
      "estimator":"CatBoost fixed 240 trees depth 5, no hyperparameter search",
      "outer":"StratifiedGroupKFold 5 grouped; global binary stratification proxy only",
      "label":"q75 from each outer train Li, never from outer test",
      "operating_threshold":"inner 3-group OOF chosen on outer training only",
      "not_comparable_to_historical":"Different fixed hyperparameters and fold-specific target definitions",
      "external_accuracy":None}
    (OUT/"experiment_manifest.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    print("FINAL",json.dumps(metrics,indent=2),flush=True)
if __name__=="__main__":evaluate()


