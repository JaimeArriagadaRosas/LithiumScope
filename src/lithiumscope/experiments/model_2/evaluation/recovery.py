"""Recover the two previously incomplete M2 reports without retraining."""
from __future__ import annotations

from lithiumscope.core.reproducibility.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.results.atomic_exports import write_json


def recover_m2_reports(identity,destination):
    import pandas as pd
    from lithiumscope.model_2.evaluation.saved_oof import load_predictions
    from lithiumscope.model_2.diagnostics.prior_release_evidence import fixed_old_release_comparison
    from lithiumscope.model_2.diagnostics.science_evidence import science_report
    source=local_training_source("model_2")
    if file_sha256(source)!=identity.get("dataset_sha256"):
        raise ValueError("M2 recovery source differs from candidate dataset")
    tables=RESULTS_DIR/"model_2"/"runs"/identity["run_id"]/"tables"
    manifest=pd.read_csv(source)
    scores,folds,labels=load_predictions(manifest,tables)
    if identity["algorithm"] not in scores:
        raise ValueError("Missing selected candidate OOF")
    destination.mkdir(parents=True,exist_ok=True)
    status={"run_id":identity["run_id"],"dataset_sha256":identity["dataset_sha256"],
            "model_sha256":identity["sha256"],"training":False,"reports":{}}
    tasks={"old_release_comparison":lambda:fixed_old_release_comparison(PROJECT_ROOT,manifest,folds,scores,destination),
           "scientific_diagnostics":lambda:science_report(PROJECT_ROOT,manifest,folds,labels,scores,destination)}
    for key,task in tasks.items():
        try:
            status["reports"][key]={"state":"complete","result":task()}
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            status["reports"][key]={"state":"requires_recovery","reason":str(exc)}
    status["state"]="complete" if all(r["state"]=="complete" for r in status["reports"].values()) else "partial"
    write_json(destination/"recovery_status.json",status)
    return status
