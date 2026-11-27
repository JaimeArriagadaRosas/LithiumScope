"""Separate descriptive EDA and deterministic preparation / TRAIN-only preflight."""
from __future__ import annotations

import json
from lithiumscope.core.config import load_config


def analyze(group, source, *, include_candidates=True):
    if group=="model_1":
        from lithiumscope.model_1.eda.pipeline import run_eda
    elif group=="model_2":
        from lithiumscope.model_2.eda.pipeline import run_eda
    else:
        raise ValueError("Unknown model group")
    from lithiumscope.results import eda_cache
    evidence=eda_cache.lookup(group,source)
    if evidence is None:
        key=eda_cache.key_for(group,source)
        evidence=run_eda(source)
        eda_cache.record(group,source,evidence,key)
    else:
        print(f"[OK] EDA descriptivo vigente, sin repetir análisis: {evidence}")
    records=[]
    if group=="model_1" and include_candidates:
        from lithiumscope.experiments.data_lab.candidate_eda import run_candidates
        records=run_candidates(source)
    state="complete" if all(r["state"]=="complete" for r in records) else "partial"
    return evidence,state,{"eda_path":str(evidence),
        "eda_manifest":json.loads((evidence/"manifest.json").read_text(encoding="utf-8")),
        "candidate_sources":records,"model_fits":0,"treatment_plans_created":False}


def existing_eda(group, source):
    """Require exact, intact descriptive evidence from the independent EDA stage."""
    from lithiumscope.results.eda_cache import lookup
    path=lookup(group,source)
    if path is not None:
        return path
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
