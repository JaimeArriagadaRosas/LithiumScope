"""Read-only cohort and archived spatial protocol verification for M1 refinement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.experiments.generalization.prepared import load_snapshot
from lithiumscope.experiments.generalization.partitions import partitions
from lithiumscope.experiments.generalization.preflight import require_treatments


def load_protocol(run_id):
    if not run_id or Path(run_id).name != run_id or any(c in run_id for c in ("/","\\",":","..")):
        raise ValueError("Invalid previous M1 run identifier")
    prior = RESULTS_DIR/"generalization"/"model_1"/run_id
    manifest_path = prior/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete" or manifest.get("scope") != "development_only":
        raise ValueError("Previous M1 screen is not complete")
    key = manifest["prepared_identity"]
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid prepared identity")
    snapshot = DATA_DIR/"processed"/"model_1"/"prepared_versions"/key
    cfg = load_config("model_1")
    data = load_snapshot(snapshot,key)
    plan, plan_manifest = require_treatments(snapshot,cfg)
    if plan_manifest["identity"] != manifest["treatment_plan_identity"]:
        raise ValueError("Prior treatment plan identity changed")
    frozen = json.loads((plan/"partitions.json").read_text(encoding="utf-8"))
    current = partitions("model_1",data,cfg)
    if len(frozen) != len(current):
        raise ValueError("Outer partitions changed")
    for number, ((tr,te,inner), saved) in enumerate(zip(current,frozen,strict=True),1):
        if (saved["fold"] != number or tr.tolist() != saved["train"] or te.tolist() != saved["test"]
                or [{"train":a.tolist(),"validation":b.tolist()} for a,b in inner] != saved["inner"]):
            raise ValueError("Spatial protocol changed")
    x,y,groups,sources,eligible,_,_ = data
    if not eligible.all():
        raise ValueError("Unexpected M1 QA exclusion; refinement requires original cohort")
    pred_path = prior/"predictions.csv"
    predictions = pd.read_csv(pred_path,dtype={"row_id":str})
    if predictions.duplicated(["algorithm","mode","row_id"]).any():
        raise ValueError("Duplicated previous OOF rows")
    for algorithm in manifest["algorithms"]:
        for mode in ("reference","selected_inner"):
            rows = predictions.loc[(predictions.algorithm==algorithm)&(predictions["mode"]==mode)]
            if len(rows)!=len(x) or set(rows.row_id)!=set(x.index):
                raise ValueError("Previous OOF coverage incomplete")
            for number,(_,te,_) in enumerate(current,1):
                r=rows.loc[rows.fold==number].set_index("row_id")
                ids=x.index[te]
                if set(r.index)!=set(ids) or not np.array_equal(r.loc[ids,"actual"].to_numpy(float),y.iloc[te].to_numpy(float)):
                    raise ValueError("Previous OOF targets or folds changed")
    from lithiumscope.datasets.management.sources.diagnostic import prepare_diagnostic_source
    source=prepare_diagnostic_source("model_1")
    prepared=json.loads((snapshot/"manifest.json").read_text(encoding="utf-8"))
    if file_sha256(source)!=prepared["source_sha256"]:
        raise ValueError("Source dataset changed since archived M1 screen")
    evidence={"previous_run":run_id,"prepared_identity":key,"treatment_plan_identity":plan_manifest["identity"],
              "source_sha256":prepared["source_sha256"],"previous_manifest_sha256":file_sha256(manifest_path),
              "previous_predictions_sha256":file_sha256(pred_path)}
    return data,current,evidence
