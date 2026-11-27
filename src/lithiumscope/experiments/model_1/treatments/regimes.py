"""Soft mixture of experts; routing at prediction time uses X only."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from lithiumscope.experiments.model_1.training_methods.fitting import _base_fit
from lithiumscope.experiments.model_1.treatments.policies import Arm
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema,build_preprocessor
from lithiumscope.resources.manager import estimator_thread_scope


def regime_fit(algorithm,arm,device,seed,params,x,y,groups,sources,numeric,categorical,validation):
    # Boundaries derive only from the supplied TRAIN; never the held-out Li target.
    thresholds=np.unique(y.quantile([.5,.9]).to_numpy()) if arm.regime=="quantiles" else np.array([100.,696.])
    labels=np.searchsorted(thresholds,y.to_numpy(float),side="right")
    classes=np.unique(labels)
    if len(classes)<2:
        raise ValueError("Not enough TRAIN concentration regimes")
    schema=FeatureSchema(list(numeric),list(categorical),2,"stable_v1")
    gate=Pipeline([("preprocess",build_preprocessor(schema,"random_forest")),
                   ("gate",RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=5,
                                                 random_state=seed,n_jobs=1))])
    with estimator_thread_scope():
        gate.fit(x,labels)
    global_predict,global_info=_base_fit(algorithm,Arm("control"),device,seed,params,
                                       x,y,groups,sources,numeric,categorical)
    experts={}
    counts=[]
    trace=list(global_info["trace"])
    for label in classes:
        ix=np.flatnonzero(labels==label)
        usable=len(ix)>=40 and groups.iloc[ix].nunique()>=5
        counts.append({"regime":int(label),"real_rows":len(ix),"groups":int(groups.iloc[ix].nunique()),
                       "fallback_to_global":not usable})
        if usable:
            experts[label],info=_base_fit(algorithm,Arm("identity_target"),device,seed,params,
                x.iloc[ix],y.iloc[ix],groups.iloc[ix],sources.iloc[ix],numeric,categorical)
            trace.extend(info["trace"])
        else:
            experts[label]=global_predict

    def predict(frame):
        probabilities=gate.predict_proba(frame)
        values=np.column_stack([experts[label](frame) for label in gate.classes_])
        result=np.maximum((probabilities*values).sum(axis=1),0.)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite mixture prediction")
        return result

    return predict(validation),predict(x),{"fit_rows":len(x),"real_fit_rows":len(x),"trace":trace,
        "calibration_fits":0,"extra_regime_fits":1+sum(not r["fallback_to_global"] for r in counts),
        "thresholds_train_only":thresholds.tolist(),"regimes":counts,"router":"soft_RF_X_only"}
