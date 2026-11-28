"""Separate descriptive EDA and deterministic preparation / TRAIN-only preflight."""
from __future__ import annotations

import json
from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger
from .source_studies.registry import run_candidate_eda, studies_for


def analyze(group, source, *, include_candidates=True, source_error=None, source_blocked=False, display_canonical=None):
    if group=="model_1":
        from lithiumscope.model_1.eda.pipeline import run_eda
    elif group=="model_2":
        from lithiumscope.model_2.eda.pipeline import run_eda
    else:
        raise ValueError("Unknown model group")
    from lithiumscope.results import eda_cache
    evidence = None
    canonical_state = "complete"
    manifest = {}
    if group == "model_1" and include_candidates:
        print("\nDataset canónico M1 | preparación y perfiles por fuente", flush=True)
    try:
        if source is None:
            raise (ValueError if source_blocked else FileNotFoundError)(source_error or "Canonical source unavailable")
        evidence=eda_cache.lookup(group,source)
        if evidence is None:
            key=eda_cache.key_for(group,source)
            evidence=run_eda(source)
            eda_cache.record(group,source,evidence,key)
        else:
            print(f"[OK] EDA descriptivo vigente, sin repetir análisis: {evidence}")
        manifest = json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        if group != "model_1" or not include_candidates:
            raise
        canonical_state = "missing" if isinstance(exc, FileNotFoundError) else "blocked"
        source_error = f"{type(exc).__name__}: {exc}"
        evidence = None
        print(f"[{canonical_state.upper()}] Dataset canónico: {source_error}", flush=True)
        get_logger("experiments.data_lab").exception("Canonical M1 EDA unavailable")
    displayed = display_canonical(evidence) if display_canonical and evidence is not None else None
    records, retired, inventory = run_candidate_eda(group,source,canonical_state=canonical_state,
        detailed=True) if include_candidates else ([], [], None)
    get_logger("experiments.data_lab").info("Descriptive EDA model=%s run=%s candidate_records=%d",
        group,evidence.name if evidence is not None else None,len(records))
    state="complete" if canonical_state == "complete" and all(
        r["state"] in ("complete", "reused") for r in records) else "partial"
    return evidence,state,{"eda_path":str(evidence) if evidence is not None else None,
        "eda_manifest":manifest,"canonical_displayed":displayed,"canonical_state":canonical_state,"canonical_error":source_error,
        "candidate_sources":records,"retired_sources":retired,
        "source_inventory":str(inventory) if inventory is not None else None,
        "available_source_studies":[s.key for s in studies_for(group)],"model_fits":0,"treatment_plans_created":False}


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
