"""Verify an archived M2 screen against current immutable data and partitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.reproducibility.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.experiments.generalization.prepared import identity, load_snapshot
from lithiumscope.experiments.generalization.partitions import partitions, training_target
from lithiumscope.experiments.generalization.preflight import require_treatments


def load_protocol(run_id):
    if not run_id or Path(run_id).name!=run_id or any(c in run_id for c in ("/","\\",":","..")):
        raise ValueError("Invalid previous M2 run identifier")
    prior=RESULTS_DIR/"generalization"/"model_2"/run_id
    manifest_path=prior/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state")!="complete" or manifest.get("scope")!="development_only":
        raise ValueError("Previous M2 screen is not complete")
    key=manifest["prepared_identity"]
    if len(key)!=64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid snapshot identity")
    snapshot=DATA_DIR/"processed"/"model_2"/"prepared_versions"/key
    cfg=load_config("model_2")
    data=load_snapshot(snapshot,key)
    from lithiumscope.datasets.management.sources.diagnostic import prepare_diagnostic_source
    source=prepare_diagnostic_source("model_2")
    if identity("model_2",source,cfg)!=key:
        raise ValueError("Current V4 scientific data/configuration differs from prepared snapshot")
    prepared=json.loads((snapshot/"manifest.json").read_text(encoding="utf-8"))
    if prepared["source_sha256"]!=file_sha256(source):
        raise ValueError("M2 source manifest changed")
    plan,record=require_treatments(snapshot,cfg)
    if record["identity"]!=manifest["treatment_plan_identity"]:
        raise ValueError("Previous treatment plan changed")
    frozen=json.loads((plan/"partitions.json").read_text(encoding="utf-8"))
    splits=partitions("model_2",data,cfg)
    if len(frozen)!=len(splits):
        raise ValueError("Outer partition count changed")
    for fold,((tr,te,inner),saved) in enumerate(zip(splits,frozen,strict=True),1):
        if (saved["fold"]!=fold or tr.tolist()!=saved["train"] or te.tolist()!=saved["test"]
                or [{"train":a.tolist(),"validation":b.tolist()} for a,b in inner]!=saved["inner"]):
            raise ValueError("Spatial partition identity changed")
    x,y,groups,_,_,_,_=data
    if x.shape!=(706,170):
        raise ValueError("Refinement requires the verified frozen 706 x 170 cohort")
    pred_path=prior/"predictions.csv"
    pred=pd.read_csv(pred_path,dtype={"row_id":str})
    if pred.duplicated(["algorithm","mode","row_id"]).any():
        raise ValueError("Duplicated previous OOF rows")
    for alg in manifest["algorithms"]:
        for mode in ("reference","selected_inner"):
            rows=pred.loc[(pred.algorithm==alg)&(pred["mode"]==mode)]
            if len(rows)!=len(x) or set(rows.row_id)!=set(x.index):
                raise ValueError("Previous OOF coverage incomplete")
            for fold,(tr,te,_) in enumerate(splits,1):
                _,_,cut=training_target("model_2",data,cfg,tr)
                z=rows.loc[rows.fold==fold].set_index("row_id")
                ids=x.index[te]
                if set(z.index)!=set(ids) or not np.array_equal(z.loc[ids,"actual"],(y.iloc[te]>=cut).astype(int)):
                    raise ValueError("Previous OOF labels or spatial folds changed")
    evidence={"previous_run":run_id,"prepared_identity":key,"treatment_plan_identity":record["identity"],
              "source_sha256":prepared["source_sha256"],"previous_manifest_sha256":file_sha256(manifest_path),
              "previous_predictions_sha256":file_sha256(pred_path)}
    return data,splits,cfg,evidence
