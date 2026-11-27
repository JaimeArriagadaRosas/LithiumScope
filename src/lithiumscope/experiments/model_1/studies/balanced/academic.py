"""Approximate OASIS protocol on our own larger, explicitly truncated M1 cohort."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.sources.diagnostic import prepare_diagnostic_source
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json
from lithiumscope.results.reference_benchmarks import OASIS_M1


def parameters(trial):
    return {"n_estimators":trial.suggest_int("n_estimators",300,1500),
            "max_depth":trial.suggest_int("max_depth",4,20),
            "min_samples_split":trial.suggest_int("min_samples_split",2,20),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,10),
            "max_features":trial.suggest_categorical("max_features",["sqrt","log2",None]),
            "bootstrap":trial.suggest_categorical("bootstrap",[True,False])}


def run_academic(*,trials=20,output_root=None):
    import optuna
    if not 1<=trials<=100:
        raise ValueError("Trial budget must be 1–100 per fold")
    source=prepare_diagnostic_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups))
    device=detect_device(prefer_gpu=False)
    root=output_root or RESULTS_DIR/"generalization"/"model_1_oasis_like"
    destination=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True)
    manifest={"state":"running","scope":"academic_approximation_own_truncated_cohort",
              "source_sha256":digest,"domain":domain,"rows":len(x),"trials_per_outer_fold":trials,
              "seed":42,"outer_folds":5,"inner_folds":5,"oasis_reference":OASIS_M1,
              "limitations":["Different source cohort and preparation implementation.",
                  "Target trimming defines a restricted evaluation population; not full-domain accuracy.",
                  "Random CV allows spatial-group overlap. No model publication or independent holdout.",
                  "Seeded Optuna differs from original unseeded search."],"promotion":False}
    write_legacy_evidence_json(destination/"manifest.json",manifest)
    metrics,rows,trials_rows=[],[],[]
    try:
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(x),1):
            inner=list(KFold(5,shuffle=True,random_state=42).split(tr))
            def objective(trial):
                params=parameters(trial)
                values=[]
                for a,b in inner:
                    model=build_pipeline("random_forest",device,42,prepared.schema,params)
                    with estimator_thread_scope():
                        model.fit(x.iloc[tr[a]],y.iloc[tr[a]])
                    scores=model.predict(x.iloc[tr[b]])
                    values.append(regression_metrics(y.iloc[tr[b]],scores).rmse)
                return float(np.mean(values))
            study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42+fold))
            study.optimize(objective,n_trials=trials)
            model=build_pipeline("random_forest",device,42,prepared.schema,study.best_params)
            with estimator_thread_scope():
                model.fit(x.iloc[tr],y.iloc[tr])
            pred=model.predict(x.iloc[te])
            score=regression_metrics(y.iloc[te],pred).to_dict()
            shared=set(groups.iloc[tr])&set(groups.iloc[te])
            metrics.append({"fold":fold,"train_rows":len(tr),"test_rows":len(te),
                            "groups_shared_train_test":len(shared),"best_inner_rmse":study.best_value,**score})
            rows.extend({"row_id":str(x.index[i]),"fold":fold,"group":str(groups.iloc[i]),
                         "truth":float(y.iloc[i]),"prediction":float(p)}
                        for i,p in zip(te,pred,strict=True))
            trials_rows.extend({"fold":fold,"trial":t.number,"value":t.value,**t.params} for t in study.trials)
            print(f"OASIS-like fold {fold}/5 | RMSE={score['rmse']:.4f} | R²={score['r2']:.4f}",flush=True)
            write_table(destination/"fold_metrics.csv",pd.DataFrame(metrics))
            write_table(destination/"predictions.csv",pd.DataFrame(rows))
            write_table(destination/"trials.csv",pd.DataFrame(trials_rows))
        if len(rows)!=len(x) or len({r["row_id"] for r in rows})!=len(x) or file_sha256(source)!=digest:
            raise ValueError("Academic OOF identity mismatch")
        table=pd.DataFrame(metrics)
        prediction=pd.DataFrame(rows)
        manifest["summary"]={"rmse_mean":float(table.rmse.mean()),"rmse_std":float(table.rmse.std(ddof=1)),
            "r2_mean":float(table.r2.mean()),"r2_std":float(table.r2.std(ddof=1)),
            **{"global_"+k:v for k,v in regression_metrics(prediction.truth,prediction.prediction).to_dict().items()}}
        manifest["state"]="complete"
    except BaseException:
        manifest["state"]="interrupted"
        raise
    finally:
        write_legacy_evidence_json(destination/"manifest.json",manifest)
    print(f"Evidencias: {destination}",flush=True)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=20)
    args=parser.parse_args()
    run_academic(trials=args.trials)


if __name__=="__main__":
    main()
