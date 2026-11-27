"""Compare persisted OOF evidence only when observation and protocol identity permit."""
from __future__ import annotations

import json

import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.experiments.integrated.compare_vnext_trained_runs import (
    load_m1_oof, pair_shared_m1, load_m2_oof, check_m2_identical_protocol,
    regression_metrics, classification_metrics, group_bootstrap_gain,
)
from lithiumscope.results.atomic_exports import write_table, write_json


def compare_saved(group, models, destination):
    if not {"release","candidate"}<=set(models):
        raise FileNotFoundError("Se requieren release y candidato verificados")
    old,new=(models[k][1] for k in ("release","candidate"))
    a,b=(RESULTS_DIR/group/"runs"/i["run_id"] for i in (old,new))
    if not a.is_dir() or not b.is_dir():
        raise FileNotFoundError("Faltan resultados OOF guardados de uno o ambos modelos; un ZIP de modelo puede no incluirlos")
    for directory, identity in ((a,old),(b,new)):
        run=json.loads((directory/"run.json").read_text(encoding="utf-8"))
        if (run.get("state")!="completed" or run.get("run_id")!=identity["run_id"]
                or not identity.get("dataset_sha256")
                or run.get("summary",{}).get("dataset_sha256")!=identity["dataset_sha256"]):
            raise ValueError("Saved run identity does not match selected model")
    if group=="model_1":
        def key(path,identity):
            prefix=str(identity["algorithm"])
            tables=sorted((path/"tables").glob("oof_predictions_"+prefix+"*.csv"))
            if len(tables)!=1:
                raise ValueError("No se puede identificar inequívocamente el OOF del modelo elegido")
            return tables[0].stem.removeprefix("oof_predictions_")
        x,y=load_m1_oof(a,key(a,old)),load_m1_oof(b,key(b,new))
        if old["dataset_sha256"]==new["dataset_sha256"]:
            from lithiumscope.experiments.shared.frozen_m1_comparison import pair_frozen_m1
            paired,details=pair_frozen_m1(x,y)
        else:
            paired,details=pair_shared_m1(x,y)
        truth=paired.y_true
        o,n=paired.prediction_old,paired.prediction_new
        groups=paired.spatial_group_old
        old_metrics,new_metrics=regression_metrics(truth,o),regression_metrics(truth,n)
        gains=[group_bootstrap_gain(truth,o,n,groups,metric=m) for m in ("rmse","mae","r2")]
    else:
        if old["dataset_sha256"]!=new["dataset_sha256"]:
            from lithiumscope.core.paths import PROJECT_ROOT
            from lithiumscope.experiments.shared.reconstructed_m2_comparison import compare_reconstructed
            return compare_reconstructed(old,new,a,b,destination,root=PROJECT_ROOT)
        from lithiumscope.datasets.management.sources.diagnostic import prepare_diagnostic_source
        x,y=load_m2_oof(a,old["algorithm"]),load_m2_oof(b,new["algorithm"])
        manifest=pd.read_csv(prepare_diagnostic_source("model_2"))
        check_m2_identical_protocol(a,b,x,y,manifest)
        truth=x.y_true
        o,n=x.prospectivity_score,y.prospectivity_score
        groups=manifest.spatial_group
        paired=pd.DataFrame({"group":groups,"truth":truth,"release":o,"candidate":n,"fold":x.outer_fold})
        details={"matched":len(paired),"protocol":"identical_saved_spatial_folds_and_labels"}
        old_metrics,new_metrics=classification_metrics(truth,o),classification_metrics(truth,n)
        gains=[group_bootstrap_gain(truth,o,n,groups,metric=m) for m in ("roc_auc","average_precision","brier")]
    destination.mkdir(parents=True,exist_ok=True)
    if group=="model_1":
        by_fold=[]
        for fold, part in paired.groupby("outer_fold_old"):
            by_fold.append({"fold":int(fold),
                **{"release_"+k:v for k,v in regression_metrics(part.y_true,part.prediction_old).items()},
                **{"candidate_"+k:v for k,v in regression_metrics(part.y_true,part.prediction_new).items()}})
        write_table(destination/"fold_metrics.csv",pd.DataFrame(by_fold))
    write_table(destination/"paired_oof.csv",paired)
    report={"scope":"exploratory_saved_oof_not_independent_holdout","release":old,"candidate":new,
            "comparison":details,"release_metrics":old_metrics,"candidate_metrics":new_metrics,"gains":gains}
    return write_json(destination/"comparison.json",report)
