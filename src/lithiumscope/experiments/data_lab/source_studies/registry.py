"""Discover source research independently of model training.

Automatic EDA is registered explicitly; other source studies have their own
contracts and must be invoked with their declared inputs.
"""
from dataclasses import dataclass
from importlib import import_module

@dataclass(frozen=True)
class SourceStudy:
    key: str
    models: tuple[str, ...]
    module: str
    entrypoint: str
    automatic_eda: bool = False

_PREFIX="lithiumscope.experiments.data_lab.source_studies."
STUDIES=(
    SourceStudy("georoc_candidate_eda",("model_1",),_PREFIX+"georoc.candidate_eda","profile"),
    SourceStudy("m1_source_eda",("model_1",),_PREFIX+"sequential","run_sources",True),
    SourceStudy("georoc_external_evidence",("model_2",),_PREFIX+"georoc.workflow","run"),
    SourceStudy("chen",("model_1",),_PREFIX+"chen","run"),
    SourceStudy("meixner",("model_1",),_PREFIX+"meixner","parse_a1"),
    SourceStudy("whole_rock",("model_1",),_PREFIX+"whole_rock","parse_tables"),
    SourceStudy("directed_intake",("model_1",),_PREFIX+"directed_intake","run"),
    SourceStudy("unab",("model_1","model_2"),_PREFIX+"unab","audit"),
)

def studies_for(group):
    if group not in {"model_1","model_2"}:
        raise ValueError("Unknown model group")
    return tuple(study for study in STUDIES if group in study.models)

def run_candidate_eda(group, source):
    records=[]
    for study in studies_for(group):
        if study.automatic_eda:
            result = getattr(import_module(study.module),study.entrypoint)(source)
            records.extend(result[0])
    return records
