"""Consolidate the implemented satellite methodological plan and release gate.

Read-only: consumes generated evidence, never fits or promotes a model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from lithiumscope.core.paths import PROJECT_ROOT

ROOT_REL="results/scientific_simulations/sentinel_methodology_20261112"

def read(path:Path)->dict:
    if not path.is_file():raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def report(root:Path, *, target:Path|None=None)->dict:
    base=root/ROOT_REL
    inv=read(base/"inventory/inventory_summary.json")
    exp=read(base/"experiments/experiment_summary.json")
    source=read(base/"experiments/feature_derivation_audit.json")
    quality=read(root/"results/scientific_simulations/m2_phase_d_20261112/physical_label_audit/scientific_quality_summary.json")
    label_gate=read(root/"results/scientific_simulations/m2_phase_d_20261112/physical_label_audit/scientific_readiness_gate.json")
    scl_path=base/"local_scl/scl_audit_summary.json"
    scl=read(scl_path) if scl_path.exists() else {
       "state":"not_run","patches_scanned":0,
       "patches_remaining":inv["n_unique_images"],
       "patches_proxy_accepted":0,"patches_scl_unavailable":0,
       "patches_proxy_abstained_or_unknown":0,
    }
    scl_samples_path=base/"local_scl/per_sample_scl_quality_783.csv"
    scl_samples=pd.read_csv(scl_samples_path) if scl_samples_path.is_file() else pd.DataFrame()
    sample_proxy_pass=(
        int(scl_samples.scientifically_accepted_proxy.fillna(False).sum())
        if len(scl_samples) else 0
    )
    if len(scl_samples) and len(scl_samples)!=783:
        raise ValueError("Global SCL inventory does not cover all 783 source samples")
    if inv["n_samples"]!=783 or inv["n_unique_scenes"]!=111:
        raise ValueError("Original satellite inventory incomplete")
    if exp["state"]!="completed" or exp["rows"]!=706 or exp["fit_count"]!=15:
        raise ValueError("Paired radiometry comparison incomplete")
    if source["rows"]!=783 or source["scenes_with_stac_radiometry"]!=111:
        raise ValueError("Per-scene STAC radiometry provenance incomplete")
    if quality["historical_conflicting_sites"]!=25:
        raise ValueError("Label diagnosis no longer aligned")
    proposals=pd.read_csv(base/"inventory/same_day_alternative_decisions.csv")
    if len(proposals)!=10 or int(sum(proposals.decision=="SAME_DATE_PROXY_ACCEPTED"))!=4:
        raise ValueError("Same-day scene QA decisions changed")
    scores=pd.read_csv(base/"experiments/paired_comparison.csv")
    bounds=pd.read_csv(base/"experiments/bootstrap_group_comparisons.csv")
    stac=pd.read_csv(base/"inventory/scene_metadata_inventory.csv")
    if len(stac)!=111 or not stac.complete_radiometric_metadata.all():
        raise ValueError("Incomplete STAC scene metadata")
    neg=int(sum(stac.radiometric_offset.astype(str).str.contains("-0.1",regex=False)))
    zero=len(stac)-neg
    if neg<1 or zero<1:
        raise ValueError("Expected mixed baseline radiometric offsets")
    f=lambda x: "no estimable" if pd.isna(x) else f"{float(x):.4f}".replace(".",",")
    names={
        "legacy_archived":"Referencia histórica DN con mediana",
        "raw_valid_only":"DN sin imputación",
        "reflectance_valid_only":"BOA con escala y offset, solo píxeles válidos",
        "reflectance_median_filled":"BOA con escala y offset, mediana histórica",
    }
    lines=[
      "# LithiumScope — Implementación del plan metodológico satelital",
      "",
      "**Fecha:** 12/11/2026. **Estado:** código, auditorías y contrastes internos ejecutados. Ningún cambio en modelos activos, menú, etiquetas, imágenes ni folds originales.",
      "",
      "## Diagnóstico y criterio científico",
      "",
      "Los 16 registros de Sentinel-2 con datos incompletos son 10 imágenes originales distintas; las pruebas de robustez que eliminan píxeles artificialmente se ejecutan por separado. El código anterior seleccionaba el menor porcentaje de nubosidad global, aun cuando el parche de 640 m tuviera cobertura insuficiente o coincidiera con bordes de escena.",
      "",
      "La calidad local se debe juzgar conjuntamente por seis bandas, valores sin datos y la clasificación SCL. Una escena de datos completos puede tener nieve, agua o sombras y carecer de señal útil para la prospección de roca superficial. La clase 5 significa no vegetado y no garantiza litología ni validación de terreno.",
      "",
      "## 1. Inventario del entrenamiento original",
      "",
      "| Indicador | Valor |",
      "|---|---:|",
      f"| Muestras / parches de entrada | {inv['n_samples']} |",
      f"| Imágenes únicas por SHA-256 | {inv['n_unique_images']} |",
      f"| Escenas Sentinel-2 recuperadas | {inv['stac_scenes_retrieved']} / 111 |",
      f"| Escenas con metadatos radiométricos completos | {inv['stac_scene_metadata_complete']} / 111 |",
      f"| Escenas STAC con offset -0,1 | {neg} |",
      f"| Escenas STAC con offset 0 | {zero} |",
      f"| Muestras con cobertura conjunta inferior al 95% | {inv['low_coverage_samples']} |",
      f"| Imágenes únicas incompletas | {inv['low_coverage_distinct_images']} |",
      f"| SCL local de imágenes únicas comprobado | {scl['patches_scanned']} / {inv['n_unique_images']} |",
      f"| SCL local pendiente | {scl['patches_remaining']} |",
      f"| Imágenes únicas con proxy óptico aceptado | {scl['patches_proxy_accepted']} / 593 |",
      f"| Muestras (incluidos sitios repetidos) con proxy óptico aceptado | {sample_proxy_pass} / 783 |",
      f"| Muestras con abstención de proxy óptico | {783-sample_proxy_pass} / 783 |",
      f"| SCL no disponible | {scl['patches_scl_unavailable']} |",
      "",
      "Las máscaras espectrales locales se calcularon sobre los NPY originales, sin reescribirlos. Los metadatos STAC se almacenaron en caché independiente, con identificador de escena, nubosidad, no-datos globales, fecha, baseline y parámetros radiométricos por banda. La lectura SCL está separada del manifiesto geoquímico y tiene checkpoints recuperables.",
      "",
      "## 2. Selector de escenas: calidad local, fecha predefinida y abstención",
      "",
      "La API nueva search_validated_scene permite elegir una escena de manera explícita con ventana de fechas obligatoria y limitación de candidatos, dejando search_best_scene histórico sin cambios. Examina seis bandas, SCL y procedencia radiométrica; acepta parches únicamente con cobertura espectral >=95%, bloqueos SCL <=25% y clase 5 no vegetado >=50% como proxy conservador. El selector nunca certifica roca ni valida litio.",
      "",
      "El parámetro selected_scene de download_patch permite adquirir una nueva muestra desde la escena validada sin volver a seleccionar por nubosidad; si el archivo ya existe, se respeta la caché original y no se sobrescribe.",
      "",
      "| Imagen única | Cobertura antigua | Candidata aceptada de la misma fecha | Decisión |",
      "|---|---:|---|---|",
    ]
    for p in proposals.itertuples(index=False):
        chosen=(str(p.chosen_scene_id) if p.decision=="SAME_DATE_PROXY_ACCEPTED"
                else "ninguna")
        lines.append(f"| {p.sample_id} | {100*float(p.original_joint_coverage):.1f}% | {chosen} | {p.decision} |")
    lines.extend([
      "",
      f"Resultado: **{inv['same_day_accepted_proxy']} de 10** imágenes únicas tienen una candidata satelital de la misma fecha que supera los filtros técnicos; **{inv['same_day_abstentions']} de 10** quedan en abstención. Todas las decisiones quedan auditadas con motivos y fracciones SCL. No se reemplazó ninguna imagen.",
      "",
      "## 3. Radiometría y tratamiento de datos incompletos",
      "",
      "Se implementó conversión por escena y banda reflectancia = DN × scale + offset, leyendo obligatoriamente raster:bands de Earth Search. No se aplica offset -0,1 a escenas antiguas con offset 0. Los píxeles no-datos siguen siendo NaN mientras se calculan índices espectrales, para no confundir un denominador inválido con un índice físico cero. Se han generado tres cachés de 783 × 126 características sin modificar la caché histórica.",
      "",
      "## 4. Experimento pareado interno sobre los 706 test históricos",
      "",
      "Se reutilizó la predicción CatBoost de referencia y se ajustaron 15 modelos experimentales (tres variantes × cinco folds) con los parámetros congelados anteriores. Las 77 etiquetas GEOROC provisionales no se usaron en este entrenamiento, y no se utilizaron los diez casos conocidos para elegir configuraciones.",
      "",
      "| Representación | ROC-AUC | AP | Brier (menor mejor) | Recall 0,5 |",
      "|---|---:|---:|---:|---:|",
    ])
    series=scores.set_index("arm")
    for arm,label in names.items():
        q=series.loc[arm]
        lines.append(f"| {label} | {f(q.auc)} | {f(q.ap)} | {f(q.brier)} | {f(q.recall_at_threshold)} |")
    lines.extend([
      "",
      "Los intervalos exploratorios se calcularon mediante 320 remuestreos por los 66 grupos históricos. Las diferencias ROC-AUC respecto de referencia incluyen cero en todos los casos.",
      "",
      "| Variante respecto a referencia | ΔAUC | IC95% exploratorio | ΔAP | IC95% exploratorio AP | ΔBrier | IC95% exploratorio Brier |",
      "|---|---:|---|---:|---|---:|---|",
    ])
    base_score=series.loc["legacy_archived"]
    for name in names:
        if name=="legacy_archived":continue
        q=series.loc[name]
        b=bounds.set_index("arm").loc[name]
        lines.append(
          f"| {names[name]} | {f(q.auc-base_score.auc)} | [{f(b.auc_delta_ci95_low)}, {f(b.auc_delta_ci95_high)}] | "
          f"{f(q.ap-base_score.ap)} | [{f(b.ap_delta_ci95_low)}, {f(b.ap_delta_ci95_high)}] | "
          f"{f(q.brier-base_score.brier)} | [{f(b.brier_delta_ci95_low)}, {f(b.brier_delta_ci95_high)}] |"
        )
    lines.extend([
      "",
      "**Interpretación:** la variante de reflectancia físicamente corregida sin imputación tuvo menor AP y peor Brier. La corrección combinada con mediana elevó AUC, pero no mejoró simultáneamente AP o calibración. **No está demostrado que ninguna configuración generalice mejor**; la corrección radiométrica es científicamente pertinente pero exige reentrenamiento y validación específica antes de desplegarse.",
      "",
      "## 5. Elegibilidad de muestras y validación independiente",
      "",
      f"Persisten {quality['historical_conflicting_sites']} sitios históricos con etiquetas Li alto/bajo incompatibles sobre idénticos parches y 77 muestras GEOROC provisionales. La documentación original Mamani09 permitió recuperar las fuentes, pero ni el DOI ni la clase SCL de suelo no vegetado certifican la correspondencia muestra–afloramiento–imagen.",
      "",
      f"**Gate de campo: {'APROBADO' if label_gate['scientific_certification_pass'] else 'NO APROBADO'}.** Las revisiones manuales documentadas anteriormente no fueron sustituidas por reglas automáticas. No hubo cambio de etiquetas, exclusiones retroactivas del test ni selección de políticas según los diez casos.",
      "",
      "Se mantiene como pendiente científico un conjunto externo nuevo, con muestras geoquímicas verificadas y bloqueo espacial/temporal desde el inicio. Los 706 históricos ya se han utilizado para desarrollo y sus intervalos no son confirmación independiente.",
      "",
      "## Ejecución y entregables técnicos",
      "",
      "El paquete científico opt-in contiene satellite_quality.py, satellite_inventory.py, scl_inventory.py, radiometric_features.py y radiometric_comparison.py. Su integración compatible está en data/sentinel2.py. Las reglas de selección de escenas no se activaron en el entrenamiento actual.",
      "",
      "Comandos reproducibles (ejecutar desde la raíz del repositorio):",
      "",
      "    python -m lithiumscope.model_2.methodology.satellite_inventory --output results/scientific_simulations/sentinel_methodology_20261112/inventory --online-metadata --alternatives",
      "    python -m lithiumscope.model_2.methodology.scl_inventory --output results/scientific_simulations/sentinel_methodology_20261112/local_scl",
      "    python -m lithiumscope.model_2.methodology.radiometric_comparison --output results/scientific_simulations/sentinel_methodology_20261112/experiments",
      "",
      "Carpeta de evidencias: results/scientific_simulations/sentinel_methodology_20261112. Incluye manifest original auditado, 111 respuestas STAC cacheadas, decisiones mismas fechas, tres tablas espectrales, estimaciones OOF pareadas, incertidumbre y logs.",
      "",
      "## Bibliografía técnica",
      "",
      "- ESA, Sentinel-2 processing, códigos SCL 0–11: https://sentiwiki.copernicus.eu/web/s2-processing",
      "- Documento de especificación Sentinel-2: https://sentinels.copernicus.eu/documents/d/sentinel/s2-pdgs-cs-di-psd-v15-0",
      "- Copernicus Sentinel-2 L2A, harmonización DN y PB 04.00: https://documentation.v1.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html",
      "- Sentinel-2 offset y escala física: https://ycgs-lab.github.io/data-guide/data-products/Sentinel-2.html",
      "",
      "No se efectuó reemplazo de escenas, ni promoción de modelos, ni modificaciones del menú, ni push.",
    ])
    if scl["state"]!="complete":
        lines.insert(0,"**Advertencia:** auditoría SCL global todavía parcial; no declarar completada la clasificación de superficie.")
    contents="\n".join(lines)+"\n"
    target=target or root/"docs/models/M2_SATELLITE_METHOD_DIAGNOSTIC_PLAN_2026.md"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(contents,encoding="utf-8")
    (base/"INFORME_SATELITAL_METODOLOGICO_20261112.md").write_text(
        contents,encoding="utf-8")
    gate={
      "status":"diagnosed_and_implemented",
      "full_original_783_inventory":True,
      "full_111_scene_stac_metadata":True,
      "local_scl_state":scl["state"],
      "local_scl_coverage":scl["patches_scanned"],
      "sample_scl_proxy_accepted":sample_proxy_pass,
      "sample_scl_abstentions":783-sample_proxy_pass,
      "accepted_same_day_proxy_alternatives":inv["same_day_accepted_proxy"],
      "original_scenes_replaced":0,
      "original_labels_changed":0,
      "saved_models_changed":0,
      "source_model_2_activated":False,
      "historical_test_706_unchanged":True,
      "scientific_label_gate":False,
      "independent_external_validation":False,
      "model_eligible_for_promotion":False,
      "all_paired_AUC_intervals_include_zero":all(
        item.auc_delta_ci95_low<=0<=item.auc_delta_ci95_high
        for item in bounds.itertuples(index=False)),
      "report_path":str(target),
    }
    (base/"scientific_release_gate.json").write_text(
        json.dumps(gate,ensure_ascii=False,indent=2),encoding="utf-8")
    print("REPORT_COMPLETED",json.dumps(gate,ensure_ascii=False),flush=True)
    return gate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    args=p.parse_args()
    target=Path(args.output) if args.output else None
    if target is not None and not target.is_absolute():target=PROJECT_ROOT/target
    report(PROJECT_ROOT,target=target)

if __name__=="__main__":
    main()
