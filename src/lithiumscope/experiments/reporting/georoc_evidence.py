"""Complete conservative source-quality review without inventing external validation.

Generates a per-citation audit, full evidence ledger, and final report from the
downloaded public GEOROC data. Does not alter, fit, or evaluate trained models.
"""
from __future__ import annotations
from pathlib import Path
from lithiumscope.experiments.georoc.protocol import EXPERIMENT_ROOT
import json,re,hashlib
from lithiumscope.experiments.georoc.provenance.bibliography import unambiguous_references
from lithiumscope.results.atomic_exports import write_table, write_json, write_text
import numpy as np
import pandas as pd

BASE=EXPERIMENT_ROOT
DOC=BASE/"GEOCHEMICAL_EXTERNAL_VALIDATION_FINAL.md"
def main(base=BASE, *, candidate_path=None, citations_path=None):
    base=Path(base)
    candidate_path=Path(candidate_path) if candidate_path is not None else base/"georoc_extended_evidence_screen.csv"
    citations_path=Path(citations_path) if citations_path is not None else base/"georoc_citations_2024.tab"
    d=pd.read_csv(candidate_path,low_memory=False)
    ref=pd.read_csv(citations_path,sep="\t",encoding="latin1",low_memory=False)
    ref, ambiguous=unambiguous_references(ref)
    cols=["citation_number","DOI","TITLE","AUTHORS","YEAR","FORMATTED CITATION"]
    ref=ref[cols]
    review=d.loc[d.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE")].copy()
    review["coordinate_key"] = list(zip(review.latitude, review.longitude))
    citations=review.groupby("citation_number",dropna=False).agg(
        samples=("sample_name","size"),country_count=("country","nunique"),
        countries=("country",lambda x:";".join(sorted(set(x.dropna().astype(str))))),
        minimum_distance_km=("nearest_training_km","min"),
        maximum_distance_km=("nearest_training_km","max"),
        median_li_ppm=("li_ppm","median"),
        distinct_coordinates=("coordinate_key","nunique"),
        example_sample=("sample_name","first"),
    ).reset_index().merge(ref,on="citation_number",how="left",validate="one_to_one")
    citations["primary_method_confirmed_for_li"]=False
    citations["training_source_independence_proven"]=False
    citations["surface_match_confirmed"]=False
    citations["eligibility"]="MANUAL_ARTICLE_AND_SAMPLE_VALIDATION_REQUIRED"
    write_table(base/"all_reviewable_citations.csv",citations)
    d["distance_gate_pass"]=d.nearest_training_km.ge(100)
    d["citation_metadata_present"]=d.resolved_citation.fillna(False)
    d["element_method_proven"]=False
    d["source_independence_proven"]=False
    d["surface_label_match_proven"]=False
    d["scientific_validation_approved"]=False
    d["decision_reason"]=d.candidate_gate.where(
        ~d.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE"),
        "PENDING_ELEMENT_METHOD_AND_GEOLOGICAL_SURFACE_MATCH")
    write_table(base/"external_validation_decision_ledger.csv",d)
    summary={
        "source":"GEOROC Andean Arc 2026-09",
        "candidate_rows":len(d),
        "automatic_gate_counts":d.candidate_gate.value_counts().to_dict(),
        "citations_in_review_pool":len(citations),
        "review_pool_samples":len(review),
        "references_with_doi":int(citations.DOI.notna().sum()),
        "samples_approved":0,
        "model2_external_roc_auc":None,
        "model2_external_average_precision":None,
        "model2_external_balanced_accuracy":None,
        "evidence_hash":hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "ambiguous_reference_ids":len(ambiguous),
        "remaining_external_dependencies":[
            "Primary papers or supplementary tables proving Li-specific assay technique for each admitted sample",
            "Verifiable original specimen ID, collection type, and exposure at satellite-observed surface",
            "Sample-level and bibliography-level decontamination against Mamani09 training inputs",
            "Independent held-out geochemically labelled surface samples and adequate geographical diversity",
            "Independent holdout definitions and threshold calibrated only on training observations",
        ],
        "integrity":"Do not treat any provisional screening candidate as verified external ground truth."
    }
    write_json(base/"final_evidence_status.json",summary)
    lines=["# Estado final de la auditoría externa GEOROC para LithiumScope",
        "",f"**Alcance:** revisión reproducible de {len(d)} candidatos de roca total con Li.",
        "",f"- Candidatos totales: **{len(d)}**.",
        f"- Superaron filtros automáticos preliminares: **{len(review)}**.",
        f"- Publicaciones/códigos de referencia distintos en ese subconjunto: **{len(citations)}**.",
        f"- Referencias con DOI registrado: **{int(citations.DOI.notna().sum())}**.",
        "- Muestras admisibles para evaluar exactitud externa: **0**.",
        "","## Comprobaciones efectuadas",
        "Cruce con registros primarios, Li numérico, material, coordenadas, duplicación de IDs, profundidad de perforación, litología, distancia a 706 puntos de entrenamiento y tabla bibliográfica oficial.",
        "Se conservan en el ledger todos los registros, el estado individual y el motivo de exclusión o revisión.",
        "","## Limitación científica",
        "La tabla de referencias contiene datos bibliográficos, **no demuestra el método específico de Li ni la independencia de las publicaciones**. Los datos de roca y el score de una escena de 640 m miden objetos distintos; requieren verificación geológica de correspondencia espacial.",
        "Una ubicación a más de 100 km del entrenamiento es solo un filtro de proximidad, no un criterio suficiente de independencia.",
        "No se calculan AUC, AP ni BA externas hasta cumplir las condiciones de evidencia. Cuatro inferencias piloto no son una prueba de generalización.",
        "","## Archivos",
        f"- `external_validation_decision_ledger.csv`: decisión individual de {len(d)} registros.",
        "- `all_reviewable_citations.csv`: referencias distintas de los candidatos con metadatos disponibles.",
        "- `final_evidence_status.json`: resumen, bloqueos y SHA-256.",
        "- Los CSV originales GEOROC, fuentes PDF descargadas, el catálogo Mamani09 y simulaciones M2 se conservan sin alteraciones.",
        "","## Trabajo externo no certificable automáticamente",
        "Consulta de artículos completos y suplementos analíticos, identificación de la técnica específica para Li, revisión de equivalencia de afloramiento con los píxeles y auditoría exhaustiva de solapamiento bibliográfico. Estos controles requieren evidencia primaria, no sustitutos o valores inferidos.",
    ]
    write_text(base/DOC.name,"\n".join(lines)+"\n")
    print("FINAL",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    with run_log("simulations","georoc_final_evidence_report"):
        main()
