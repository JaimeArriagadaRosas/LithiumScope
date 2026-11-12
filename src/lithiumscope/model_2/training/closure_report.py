"""Create the automatic M2 manual-training evidence summary (never a release gate)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

OLD="original_release_protocol_retrained"
NEW="corrected_quality95"


def write_closure_report(out:Path,preflight:dict,evaluation:dict,
                         *,model_path:Path|None=None)->dict:
    if evaluation.get("same_706_test_cases") is not True:
        raise ValueError("Refuse to compare models with different test cases")
    metrics=evaluation["metrics"]
    before=metrics[OLD]
    after=metrics[NEW]
    t=evaluation["three_zone"]["arms"]
    old_3=t[OLD]
    new_3=t[NEW]
    if not old_3["triple_cases"]==new_3["triple_cases"]:
        raise ValueError("Three-site comparisons do not have shared case sets")
    changes={
        "roc_auc":float(after["auc"]-before["auc"]),
        "average_precision":float(after["ap"]-before["ap"]),
        "brier":float(after["brier"]-before["brier"]),
        "top1_three_zone":float(new_3["top1_accuracy"]-old_3["top1_accuracy"]),
    }
    criteria={
        "roc_auc_increased":changes["roc_auc"]>0,
        "average_precision_increased":changes["average_precision"]>0,
        "brier_not_worse":changes["brier"]<=0,
        "three_zone_top1_not_worse":changes["top1_three_zone"]>=0,
    }
    internally_better=all(criteria.values())
    outcome={
        "state":"internal_experimental_comparison",
        "comparator":OLD,
        "comparator_method":"retrained published CatBoost hyperparameters, frozen same 706 historic outer folds",
        "candidate":NEW,
        "same_sample_labels_and_folds":True,
        "all_706_test_samples_retained":True,
        "old_metrics":before,
        "new_metrics":after,
        "old_three_zone":old_3,
        "new_three_zone":new_3,
        "deltas":changes,
        "criteria":criteria,
        "passes_descriptive_internal_criteria":internally_better,
        "independent_external_validation":False,
        "supports_claim_of_external_generalization":False,
        "not_for_individual_release":True,
        "candidate_model_path":str(model_path) if model_path else None,
        "note":"Even a positive internal comparison does not certify geology or extraction feasibility.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"experimental_acceptance.json").write_text(
        json.dumps(outcome,ensure_ascii=False,indent=2),
        encoding="utf-8")
    pct=lambda v:f"{100*float(v):.2f}%"
    number=lambda v:f"{float(v):.4f}"
    lines=[
        "# LithiumScope — Informe del entrenamiento manual del Modelo 2",
        "",
        "**Estado:** experimento local descartable, SIN release, SIN activación.",
        "",
        "## Propósito de uso",
        "",
        "Modelo 1 estima ppm de Li. Modelo 2 ofrece una señal espectral relativa para ordenar tres ubicaciones candidatas. Su puntuación NO es probabilidad de yacimiento, estudio económico ni autorización para extraer litio.",
        "",
        "## Condiciones congeladas",
        "",
        "- 706 casos históricos, con cinco folds geográficos originales.",
        "- Todas las etiquetas de test preservadas, incluidas 16 imágenes incompletas.",
        "- En cada fold solo se filtra el TRAIN con cobertura finita conjunta >=95%; el ajuste final utiliza 690 filas.",
        "- Umbral geoquímico q75 derivado exclusivamente del entrenamiento histórico de cada fold.",
        "- CatBoost corregido con parámetros fijos (160 árboles, profundidad cinco, semilla 42, CPU).",
        "- Comparador: se reconstruye por fold el protocolo del CatBoost del release de septiembre (669 árboles, profundidad ocho, tasa ~0,0108), NO se evalúa el artefacto final sobre sus propios datos de entrenamiento.",
        "- Sin las 77 etiquetas GEOROC provisionales, sin búsqueda Optuna ni calibración ajustada con los casos test.",
        "- La representación principal mantiene las 126 características DN del release: la corrección radiométrica sigue siendo experimental porque no mejoró AP ni Brier.",
        "",
        "## Resultados de validación cruzada históricos",
        "",
        "| Métrica | Procedimiento del release anterior | Modelo 2 corregido | Cambio |",
        "|---|---:|---:|---:|",
        f"| ROC-AUC (mayor mejor) | {number(before['auc'])} | {number(after['auc'])} | {changes['roc_auc']:+.4f} |",
        f"| Average Precision (mayor mejor) | {number(before['ap'])} | {number(after['ap'])} | {changes['average_precision']:+.4f} |",
        f"| Brier (menor mejor) | {number(before['brier'])} | {number(after['brier'])} | {changes['brier']:+.4f} |",
        f"| Recall a corte fijo 0,5 | {pct(before['recall_at_threshold'])} | {pct(after['recall_at_threshold'])} | — |",
        f"| Balanced Accuracy a 0,5 | {number(before['balanced_accuracy_at_threshold'])} | {number(after['balanced_accuracy_at_threshold'])} | — |",
        "",
        "## Uso práctico: elegir la mejor entre tres zonas",
        "",
        "Se formaron las mismas ternas, con tres sitios diferentes dentro de cada fold test, y se midió si el score eligió la ubicación con la mayor mediana de Li observado. Cada terna comparte las mismas opciones entre modelos. No se mezclan puntuaciones de modelos entrenados en folds distintos dentro de la misma comparación.",
        "",
        "| Medida | Protocolo anterior | Corregido |",
        "|---|---:|---:|",
        f"| Ternas evaluadas | {old_3['triple_cases']} | {new_3['triple_cases']} |",
        f"| Coincidencia con máximo Li | {pct(old_3['top1_accuracy'])} | {pct(new_3['top1_accuracy'])} |",
        f"| Ternas con al menos una zona Li alto | {old_3['cases_containing_a_high_Li_site']} | {new_3['cases_containing_a_high_Li_site']} |",
        f"| Acierto en esas ternas | {pct(old_3['top1_accuracy_when_high_Li_present'])} | {pct(new_3['top1_accuracy_when_high_Li_present'])} |",
        "",
        "La referencia aleatoria para elegir una de tres zonas sin empates es 33,33%, pero los sorteos reutilizan sitios históricos y NO representan una evaluación externa independiente.",
        "",
        "## Dictamen de cierre experimental",
        "",
        "Criterios internos registrados antes de interpretar este informe:",
    ]
    for k,passed in criteria.items():
        lines.append("- "+("[OK] " if passed else "[NO] ")+k)
    lines += [
        "",
        ("**Resultado:** cumple descriptivamente los cuatro criterios internos. "
         if internally_better else
         "**Resultado:** no cumple todos los criterios internos. ")+
        "No debe presentarse como mejora universal ni publicarse como Modelo 2 independiente.",
        "",
        "Los intervalos de confianza son exploratorios: los 706 ejemplos se reutilizaron durante el desarrollo, y las ternas se forman a partir de esas mismas predicciones. Para un release combinado posterior será necesaria una decisión separada y evidencia apropiada.",
        "",
        "## Material de auditoría",
        "",
        "- all_reference_paired_oof.csv: puntuaciones y etiquetas de idénticos casos por brazo.",
        "- per_fold_comparison.csv: métricas, n train y n test de cada fold.",
        "- metrics_comparison.csv y uncertainty_vs_prior_release.json: comparación pareada y remuestreo espacial.",
        "- three_zone_cases.csv y three_zone_metrics.json: todas las ternas y sus aciertos.",
        "- manual_closure_preflight.json: hashes, política de calidad, trazabilidad y muestras excluidas del TRAIN.",
        "- experimental_acceptance.json: criterios y deltas en formato legible por máquina.",
        "- Ningún archivo de producción o manifiesto original se sobrescribe.",
        "",
        "## Candidato",
        "",
        "Archivo del candidato: "+(str(model_path) if model_path else "(sin candidato guardado)")+".",
        "Su estado es experimental_awaiting_review y no figura en la lista normal de candidatos publicables. El modelo del release anterior sigue activo.",
    ]
    (out/"INFORME_ENTRENAMIENTO_MODELO2_EXPERIMENTAL.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    return outcome
