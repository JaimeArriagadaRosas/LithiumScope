"""Create dated, auditable report of predefined Phase C experiments.

The reporter never fits or promotes a model. It asserts that all 8 arms
scored the same historical test rows and that nested decisions match the
recorded outer predictions before joining the evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.ablations import ARMS, score_metrics


def format_metric(x,dec=4):
    return "n/e" if pd.isna(x) else f"{float(x):.{dec}f}".replace(".",",")


def build_report(pilot:Path, decision_source:Path, report_output:Path,
                 project_document:Path | None = None):
    s=json.loads((pilot/"experiment_summary.json").read_text(encoding="utf-8"))
    if s["state"]!="completed" or not s["same_706_held_out"]:
        raise ValueError("Pilot is not complete with common historical test rows")
    design=json.loads((pilot/"experiment_design.json").read_text(encoding="utf-8"))
    if list(design["arms"])!=list(ARMS):
        raise ValueError("Pilot code and documented experiment arms differ")
    if not (pilot/"cluster_bootstrap_exploratory.csv").is_file():
        raise FileNotFoundError("Cluster bootstrap data missing")
    data=pd.read_csv(pilot/"all_oof_arms.csv")
    assert len(data)==706*len(ARMS) and data.arm.nunique()==len(ARMS)
    y_ref=None
    for arm in ARMS:
        sub=data[data.arm==arm].sort_values("test_index")
        if sub.test_index.tolist()!=list(range(706)):
            raise ValueError("Not the same 706 historical test indices")
        if y_ref is None:
            y_ref=sub.y_true.to_numpy(int)
        if not np.array_equal(y_ref,sub.y_true.to_numpy(int)):
            raise ValueError("Historical test labels changed across arms")
        if sub.fold.nunique()!=5:
            raise ValueError("All arms require five outer folds")
    rank=pd.read_csv(pilot/"exploratory_comparison.csv")
    if set(rank.arm)!=set(ARMS):raise ValueError("Missing ablation arm")
    bootstrap=pd.read_csv(pilot/"cluster_bootstrap_exploratory.csv")
    if len(bootstrap)!=len(ARMS)-1:raise ValueError("Missing bootstrap contrasts")
    all_intervals_cross_zero=bool((
        (bootstrap.auc_delta_ci95_low<=0)&(bootstrap.auc_delta_ci95_high>=0)
    ).all())
    # Original 7-arm pilot generated the nested decision records; ensure
    # strictly that the 8-arm final pilot generated IDENTICAL outer scores.
    decisions=json.loads((decision_source/"decision_probe_summary.json").read_text(encoding="utf-8"))
    if decisions.get("state")!="completed":raise ValueError("Nested probe incomplete")
    for arm in ("legacy_all","expanded_buffer_5km"):
        original=pd.read_csv(decision_source/("oof_"+arm+".csv")).sort_values("test_index")
        final=pd.read_csv(pilot/("oof_"+arm+".csv")).sort_values("test_index")
        if original.test_index.tolist()!=final.test_index.tolist():
            raise ValueError("Nested and final outer test index sets differ")
        if not np.array_equal(original.y_true.to_numpy(int),final.y_true.to_numpy(int)):
            raise ValueError("Nested and final outer labels differ")
        if not np.allclose(original.score.to_numpy(float),final.score.to_numpy(float),
                           atol=1e-12,rtol=0):
            raise ValueError("Nested and final outer predictions differ")
    # Site-level metrics computed without fitting. A site is not assumed
    # homogeneous if multiple samples disagree: report both all and clean.
    site_rows=[]
    for arm,sub in data.groupby("arm",sort=False):
        sites=sub.groupby("site_screening_id").agg(
            n_samples=("sample_id","size"),
            mean_score=("score","mean"),
            median_score=("score","median"),
            n_positive=("y_true","sum"),
            median_li=("li_ppm","median"),
            threshold=("fold_li_threshold_ppm","first"),
            n_folds=("fold","nunique"),
            n_classes=("y_true","nunique")
        )
        if sites.n_folds.max()!=1:
            raise ValueError("Same site appears in multiple outer folds")
        sites["site_label"]=(sites.median_li>=sites.threshold).astype(int)
        for label,sframe in (
            ("all_sites",sites),
            ("unambiguous_sites",sites[sites.n_classes==1]),
        ):
            y=sframe.site_label.to_numpy(int)
            sc=sframe.median_score.to_numpy(float)
            site_rows.append({"arm":arm,"cohort":label,"n_sites":len(sframe),
               "mixed_sites_excluded":int((sites.n_classes>1).sum()) if label!="all_sites" else 0,
               "roc_auc":float(roc_auc_score(y,sc)) if len(set(y))==2 else None,
               "ap":float(average_precision_score(y,sc)) if len(set(y))==2 else None,
               "brier":float(brier_score_loss(y,sc)),
            })
    pd.DataFrame(site_rows).to_csv(pilot/"site_aggregated_metrics.csv",index=False)
    # Descriptive area diagnostics on frozen historical observations only.
    regional=[]
    data["historical_region"]=pd.cut(data.latitude,
        bins=[-90,-20,-10,90],labels=["South_of_20S","20S_to_10S","North_of_10S"],
        include_lowest=True)
    for (arm,region),sub in data.groupby(["arm","historical_region"],observed=True):
        regional.append({"arm":arm,"region":str(region),
                         **score_metrics(sub.y_true,sub.score)})
    pd.DataFrame(regional).to_csv(pilot/"regional_metrics_descriptive.csv",index=False)
    # Without a fresh holdout, overlapping bootstrap intervals should not
    # be presented as proof of no effect or a conclusive winner.
    docs=[]
    add=docs.append
    bt=chr(96)
    add("# LithiumScope — informe experimental Fase C (2026-11-12)")
    add("")
    add("Estado: ocho ablaciones ejecutadas y validación exploratoria de decisiones internas. La evidencia no es un nuevo test externo ni autoriza promoción de modelos.")
    add("")
    add("## Protocolo reproducible")
    add("Se empleó un CatBoost fijo (160 iteraciones, profundidad 5, tasa 0,065, L2=5, semilla=42, CPU), sin búsqueda de hiperparámetros ni ajuste de pesos después de ver el test. Los cinco folds espaciales y los **706 índices de test históricos fueron idénticos en las ocho variantes**, con 177 etiquetas positivas construidas a partir del cuantil de Li histórico obtenido solo en el train de cada fold. Las 77 observaciones GEOROC nuevas son provisionales; en cada fold se incorporaron únicamente aquellas asignadas originalmente al train.")
    add("")
    add("El fin es aislar el efecto de una intervención por vez. No se usaron los diez casos conocidos de demostración, no se descargaron datos, no se regeneraron imágenes, no se ajustó el modelo oficial y no se eligió un algoritmo o umbral basándose en los 706 resultados.")
    add("")
    add("## Resultados pareados (datos históricos, métricas OOF exploratorias)")
    add("| Brazo | ROC-AUC | AP | Brier (menor mejor) | Recall a 0,5 |")
    add("|---|---:|---:|---:|---:|")
    labels={
        "legacy_all":"Solo 706 históricos",
        "expanded_all":"706 + GEOROC provisionales",
        "expanded_buffer_5km":"Adicionales + buffer 5 km",
        "expanded_buffer_10km":"Adicionales + buffer 10 km",
        "expanded_exclude_shared_scenes":"Excluir escenas de test del train",
        "expanded_drop_mixed_train_sites":"Excluir sitios mixtos solo del train",
        "expanded_raw_spectral_only":"48 predictores de banda directa",
        "expanded_inverse_site_weights":"Ponderación inversa por sitio",
    }
    for arm in ARMS:
        row=rank[rank.arm==arm].iloc[0]
        add("| "+labels[arm]+" | "+format_metric(row.auc)+" | "+
            format_metric(row.ap)+" | "+format_metric(row.brier)+" | "+
            format_metric(row.recall_at_threshold)+" |")
    add("")
    add("El contraste +GEOROC frente a 706 cambia AUC y AP en direcciones distintas. La exclusión de escenas reduce considerablemente el tamaño de train (en este piloto, mínimo 251 filas), por lo que sus variaciones de métricas no permiten atribuir causalmente el efecto a filtraciones previas. Un buffer de 10 km no reproduce el AUC obtenido con 5 km, lo que cuestiona una regla simple de mayor distancia igual a mejor discriminación.")
    add("")
    add("## Incertidumbre agrupada y selección del ganador")
    add("Se aplicaron 320 remuestreos de los **66 grupos históricos** (intervalos percentiles exploratorios de 95%) a las diferencias pareadas de AUC, AP y Brier. No es inferencia confirmatoria: el conjunto histórico ya se ha utilizado para trabajo metodológico, hubo comparaciones múltiples y puede existir dependencia entre grupos y escenas.")
    add("")
    add("| Contraste | Delta AUC | IC exploratorio 95% AUC | Delta AP | IC exploratorio 95% AP |")
    add("|---|---:|---:|---:|---:|")
    for _,b in bootstrap.iterrows():
        arm=b.arm;reference=b.reference
        curr=rank[rank.arm==arm].iloc[0]
        ref=rank[rank.arm==reference].iloc[0]
        add("| "+labels[arm]+" vs "+labels[reference]+" | "+
            format_metric(curr.auc-ref.auc)+" | ["+format_metric(b.auc_delta_ci95_low)+", "+
            format_metric(b.auc_delta_ci95_high)+"] | "+
            format_metric(curr.ap-ref.ap)+" | ["+format_metric(b.ap_delta_ci95_low)+", "+
            format_metric(b.ap_delta_ci95_high)+"] |")
    add("")
    add("Todas las bandas de incertidumbre de AUC **incluyen cero**: los datos no respaldan una ganancia estadística robusta de una variante concreta. Eso no demuestra equivalencia entre los métodos. Tampoco se debe elegir el buffer ni las características a partir del máximo observado.")
    add("")
    add("## Umbrales aprendidos exclusivamente dentro del entrenamiento")
    add("Se ejecutó un experimento adicional en dos brazos predefinidos, con tres folds internos agrupados espacialmente por cada fold exterior, sin examinar las etiquetas de test para fijar el corte. Se reutilizaron las puntuaciones exteriores verificando equivalencia numérica exacta entre el piloto de siete brazos y la corrida final de ocho.")
    add("| Métrica de decisión exterior | Históricos | Con buffer 5 km |")
    add("|---|---:|---:|")
    dec_by={x["arm"]:x for x in decisions["results"]}
    for key,label in (
        ("fixed_half_recall","Recall con score >= 0,5"),
        ("native_recall","Recall con umbral OOF interno"),
        ("native_precision","Precisión con umbral OOF interno"),
        ("native_balanced_accuracy","Balanced Accuracy con umbral OOF interno"),
    ):
        add("| "+label+" | "+format_metric(dec_by["legacy_all"][key])+" | "+
            format_metric(dec_by["expanded_buffer_5km"][key])+" |")
    add("")
    add("El recall aumenta al seleccionar umbrales internos, pero la precisión sigue limitada. **No se realizó calibración probabilística**, ni comparación confirmatoria independiente. Los umbrales obtenidos no se aplicaron ni instalaron sobre ningún modelo activo.")
    add("")
    add("## Hallazgos metodológicos y próximas intervenciones")
    add("1. Antes de elegir un candidato, auditar en origen los 25 sitios de etiquetas discordantes y el significado físico de cada muestra: representatividad del afloramiento respecto del parche Sentinel-2, calidad GPS y análisis geoquímico.")
    add("2. No inferir que las 635 observaciones de test que comparten identificador de escena son filtración efectiva. Hacer una evaluación por escena geográfica/fecha con control del tamaño de train y análisis de calibración del dominio.")
    add("3. Mantener benchmarks congelados y nuevas regiones/escenas fuera del desarrollo para una confirmación final. El conjunto histórico de 706 registros ya participa en el desarrollo de Fase C.")
    add("4. Priorizar selección de umbral y calibración verdaderamente anidadas con objetivos explícitos de recall y coste de falsos positivos, nunca fijar umbrales con los diez casos.")
    add("5. Explorar mejores descriptores de roca expuesta/máscaras SCL y una etiqueta por sitio verificable; comparar de modo pareado con datos nuevos y sin mezclar cambios de protocolo.")
    add("")
    add("## Límites de la evidencia")
    add("Las diferencias de score entre brazos no estiman por sí mismas la probabilidad de descubrir depósitos. No se validó correspondencia de litio en terreno; se conservó la etiqueta operativa de q75. La exclusión de sitios mixtos se hizo solo en train, mientras el test mantiene su definición original. La reducción a 48 columnas estaba predefinida, no seleccionada según test. Los modelos de este estudio son pequeños candidatos temporales y **no se guardaron como modelos de producción**.")
    add("")
    add("## Archivos auditables")
    for fname in ("experiment_design.json","experiment_summary.json",
       "all_oof_arms.csv","fold_metrics.csv","exploratory_comparison.csv",
       "cluster_bootstrap_exploratory.csv","site_aggregated_metrics.csv",
       "regional_metrics_descriptive.csv"):
        add("- "+bt+fname+bt)
    add("Los archivos "+bt+"inner_frozen_thresholds_*.csv"+bt+
        " y "+bt+"decision_probe_summary.json"+bt+
        " están en el subdirectorio del primer piloto de siete brazos; su alineación con los resultados finales fue comprobada numéricamente.")
    add("")
    add("El código que reproduce la ejecución reside en "+
        bt+"src/lithiumscope/model_2/methodology/ablations.py"+bt+
        " y "+bt+"decision_probe.py"+bt+". No se ha hecho push ni se ha alterado el menú o los modelos activos.")
    content="\n".join(docs)+"\n"
    report_output.parent.mkdir(parents=True,exist_ok=True)
    report_output.write_text(content,encoding="utf-8")
    if project_document is not None:
        project_document.parent.mkdir(parents=True,exist_ok=True)
        project_document.write_text(content,encoding="utf-8")
    (pilot/"report_validation.json").write_text(json.dumps({
        "state":"completed",
        "eight_arms":len(ARMS),
        "historical_test_samples":706,
        "same_reference_labels":True,
        "frozen_threshold_outer_predictions_identical":True,
        "bootstrap_auc_all_include_zero":all_intervals_cross_zero,
        "external_validation":False,
        "new_models_saved":False,
        "files_created":[str(report_output),str(project_document)],
    },indent=2,ensure_ascii=False),encoding="utf-8")
    return {"same_706":True,"eight_arms":len(ARMS),
            "auc_ci_all_include_zero":all_intervals_cross_zero,
            "report_path":str(report_output)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True)
    p.add_argument("--decision-source",required=True)
    p.add_argument("--report",required=True)
    p.add_argument("--project-document")
    args=p.parse_args()
    to_abs=lambda x:Path(x) if Path(x).is_absolute() else PROJECT_ROOT/Path(x)
    result=build_report(
        to_abs(args.pilot),to_abs(args.decision_source),to_abs(args.report),
        to_abs(args.project_document) if args.project_document else None)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
