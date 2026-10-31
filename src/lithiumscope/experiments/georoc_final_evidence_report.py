"""Complete conservative source-quality review without inventing external validation.

Generates a per-citation audit, full evidence ledger, and final report from the
downloaded public GEOROC data. Does not alter, fit, or evaluate trained models.
"""
from __future__ import annotations
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd

BASE=Path("results/scientific_simulations/external_geochemistry_audit")
DOC=Path("docs/GEOCHEMICAL_EXTERNAL_VALIDATION_FINAL.md")
def main():
    d=pd.read_csv(BASE/"georoc_extended_evidence_screen.csv",low_memory=False)
    ref=pd.read_csv(BASE/"georoc_citations_2024.tab",sep="\t",encoding="latin1",low_memory=False)
    ref["citation_number"]=pd.to_numeric(ref["CITATIONS"],errors="coerce")
    cols=["citation_number","DOI","TITLE","AUTHORS","YEAR","FORMATTED CITATION"]
    ref=ref[cols].drop_duplicates("citation_number")
    review=d.loc[d.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE")].copy()
    citations=review.groupby("citation_number",dropna=False).agg(
        samples=("sample_name","size"),country_count=("country","nunique"),
        countries=("country",lambda x:";".join(sorted(set(x.dropna().astype(str))))),
        minimum_distance_km=("nearest_training_km","min"),
        maximum_distance_km=("nearest_training_km","max"),
        median_li_ppm=("li_ppm","median"),
        distinct_coordinates=("latitude","nunique"),
        example_sample=("sample_name","first"),
    ).reset_index().merge(ref,on="citation_number",how="left",validate="one_to_one")
    citations["primary_method_confirmed_for_li"]=False
    citations["training_source_independence_proven"]=False
    citations["surface_match_confirmed"]=False
    citations["eligibility"]="MANUAL_ARTICLE_AND_SAMPLE_VALIDATION_REQUIRED"
    citations.to_csv(BASE/"all_reviewable_citations.csv",index=False)
    d["distance_gate_pass"]=d.nearest_training_km.ge(100)
    d["citation_metadata_present"]=d.resolved_citation.fillna(False)
    d["element_method_proven"]=False
    d["source_independence_proven"]=False
    d["surface_label_match_proven"]=False
    d["scientific_validation_approved"]=False
    d["decision_reason"]=d.candidate_gate.where(
        ~d.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE"),
        "PENDING_ELEMENT_METHOD_AND_GEOLOGICAL_SURFACE_MATCH")
    d.to_csv(BASE/"external_validation_decision_ledger.csv",index=False)
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
        "evidence_hash":hashlib.sha256((BASE/"georoc_extended_evidence_screen.csv").read_bytes()).hexdigest(),
        "remaining_external_dependencies":[
            "Primary papers or supplementary tables proving Li-specific assay technique for each admitted sample",
            "Verifiable original specimen ID, collection type, and exposure at satellite-observed surface",
            "Sample-level and bibliography-level decontamination against Mamani09 training inputs",
            "Independent held-out geochemically labelled surface samples and adequate geographical diversity",
            "Independent holdout definitions and threshold calibrated only on training observations",
        ],
        "integrity":"Do not treat any provisional screening candidate as verified external ground truth."
    }
    (BASE/"final_evidence_status.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    lines=["# Estado final de la auditoría externa GEOROC para LithiumScope",
        "","**Alcance:** revisión reproducible de los 3.220 candidatos de roca total con Li en la compilación oficial del arco andino.",
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
        "- `external_validation_decision_ledger.csv`: decisión individual de 3.220 registros.",
        "- `all_reviewable_citations.csv`: referencias distintas de los candidatos con metadatos disponibles.",
        "- `final_evidence_status.json`: resumen, bloqueos y SHA-256.",
        "- Los CSV originales GEOROC, fuentes PDF descargadas, el catálogo Mamani09 y simulaciones M2 se conservan sin alteraciones.",
        "","## Trabajo externo no certificable automáticamente",
        "Consulta de artículos completos y suplementos analíticos, identificación de la técnica específica para Li, revisión de equivalencia de afloramiento con los píxeles y auditoría exhaustiva de solapamiento bibliográfico. Estos controles requieren evidencia primaria, no sustitutos o valores inferidos.",
    ]
    DOC.parent.mkdir(parents=True,exist_ok=True)
    DOC.write_text("\n".join(lines)+"\n",encoding="utf8")
    print("FINAL",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
