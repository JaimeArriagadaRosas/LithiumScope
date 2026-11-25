"""Compact local evidence export for manual EDA verification."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import _git_commit


def collect(snapshot, evidence, treatments, treatment_manifest):
    snapshot, evidence, treatments = map(Path,(snapshot,evidence,treatments))
    prepared=json.loads((snapshot/"manifest.json").read_text(encoding="utf-8"))
    eda=json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
    if prepared["source_sha256"] != eda["source_sha256"]:
        raise ValueError("EDA and prepared dataset source hashes differ")
    if file_sha256(Path(eda["source_path"])) != prepared["source_sha256"]:
        raise ValueError("Original dataset changed after preparation")
    def records(path, omit=()):
        frame=pd.read_csv(path)
        return json.loads(frame.drop(columns=list(omit),errors="ignore").to_json(orient="records"))
    table=records(treatments/"treatments.csv", ("feature_names",))
    fold_data=records(treatments/"fold_summary.csv")
    source_table=evidence/"tables"/"cohort_summary.csv"
    return {"prepared_path":str(snapshot), "eda_path":str(evidence), "treatments_path":str(treatments),
        "prepared_manifest":prepared, "treatment_manifest":treatment_manifest,
        "eda_manifest":eda, "source_unchanged":True,
        "fold_summary":fold_data,"treatments":table,
        "cohort_summary":records(source_table) if source_table.is_file() else []}


def write_verification(state, records, *, root=None):
    root=Path(root) if root is not None else RESULTS_DIR/"eda_verification"
    root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path=root/f"eda_verification_{stamp}_{uuid4().hex[:8]}.json"
    payload={"schema_version":1,"state":state,"git_commit":_git_commit(),
        "scope":"local_eda_and_train_only_data_preparation",
        "model_fits":0,"predictive_improvement_verified":False,
        "models":records}
    temporary=path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
