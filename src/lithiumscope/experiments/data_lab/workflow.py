"""Separate descriptive EDA and deterministic preparation / TRAIN-only preflight."""
from __future__ import annotations

import json
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.results.eda_artifacts import verify_eda


def analyze(group, source, *, include_candidates=True):
    if group=="model_1":
        from lithiumscope.model_1.eda.pipeline import run_eda
    elif group=="model_2":
        from lithiumscope.model_2.eda.pipeline import run_eda
    else:
        raise ValueError("Unknown model group")
    evidence=run_eda(source)
    records=[]
    if group=="model_1" and include_candidates:
        from lithiumscope.experiments.data_lab.candidate_eda import run_candidates
        records=run_candidates(source)
    state="complete" if all(r["state"]=="complete" for r in records) else "partial"
    return evidence,state,{"eda_path":str(evidence),
        "eda_manifest":json.loads((evidence/"manifest.json").read_text(encoding="utf-8")),
        "candidate_sources":records,"model_fits":0,"treatment_plans_created":False}


def existing_eda(group, source):
    """Find intact same-source EDA; actual treatment preparation validates its own signature."""
    digest=file_sha256(source)
    for path in sorted((RESULTS_DIR/group/"eda").glob("eda_*"),reverse=True):
        try:
            manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("source_sha256")==digest and manifest.get("effective_config")==load_config(group)
                    and verify_eda(path,manifest["signature"])):
                return path
        except (OSError,ValueError,KeyError,TypeError):
            continue
    raise FileNotFoundError("Ejecute primero el análisis exploratorio para esta fuente y configuración")


def prepare(group,source):
    from lithiumscope.experiments.generalization.prepared import prepare_version
    from lithiumscope.experiments.generalization.preflight import prepare_treatments, require_treatments, show_treatments
    evidence=existing_eda(group,source)
    snapshot,_=prepare_version(group,source,run_eda=False)
    cfg=load_config(group)
    try:
        directory,manifest=require_treatments(snapshot,cfg)
        print(f"[OK] Tratamientos ya preparados y verificados: {directory}")
    except FileNotFoundError:
        directory,manifest=prepare_treatments(group,snapshot,cfg)
    show_treatments(directory,manifest)
    from lithiumscope.cli.eda_verification import collect
    record=collect(snapshot,evidence,directory,manifest)
    return manifest["state"],record
