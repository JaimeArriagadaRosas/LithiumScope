"""Render Phase E experimental report from inspectable CSV and frozen inputs.

Pure analysis of existing exploratory OOF predictions. Never trains, fits
thresholds, selects best model or updates a model registry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.site_patch_lab import ARMS


DESCRIPTIONS={
    "historical_706_reference":"Referencia histórica (706 muestras)",
    "site_median_li":"Una observación por sitio; Li mediano de TRAIN",
    "site_majority_consensus":"Una observación por sitio; consenso estricto TRAIN",
    "quality_95_train":"Solo entrenamiento con >=95% píxeles finitos",
    "site_median_quality_95":"Sitio mediano y QA del 95% (interacción)",
    "central_16x16_patch":"126 características de región central 16x16",
}


def render(out:Path, documentation:Path|None=None)->dict:
    summary=json.loads((out/"experiment_summary.json").read_text(encoding="utf-8"))
    design=json.loads((out/"experiment_design.json").read_text(encoding="utf-8"))
    if summary.get("state")!="complete" or summary.get("arms")!=list(ARMS):
        raise ValueError("Phase E must have 6 frozen and complete arms")
    if design.get("known_demo_cases_used") or design.get("all_test_cases_same") is not True:
        raise ValueError("Invalid Phase E test isolation design")
    m=pd.read_csv(out/"paired_model_comparison.csv")
    boots=pd.read_csv(out/"group_bootstrap_deltas.csv")
    fold=pd.read_csv(out/"fold_metrics.csv")
    bysite=pd.read_csv(out/"site_level_metrics.csv")
    bysample=pd.read_csv(out/"sample_quality_site_subgroups.csv")
    preds=pd.read_csv(out/"all_arms_paired_oof.csv")
    bounds=pd.read_csv(out/"identical_patch_label_error_bounds.csv")
    inventory=pd.read_csv(out/"train_site_aggregation_decisions.csv")
    missingness_path=out.parent/"missing_geometry_summary.json"
    if not missingness_path.is_file():
        raise FileNotFoundError("Phase E patch missingness evidence was not generated")
    missingness=json.loads(missingness_path.read_text(encoding="utf-8"))
    if (missingness["sample_count"]!=16
        or missingness["all_six_band_masks_equal_count"]!=16):
        raise ValueError("Unexpected original patch missingness status")
    if set(m.arm)!=set(ARMS) or len(boots)!=5 or len(fold)!=25:
        raise ValueError("Incomplete results")
    y=None
    for arm in ARMS:
        arm_prediction=preds[preds.arm==arm].sort_values("test_index")
        if len(arm_prediction)!=706 or arm_prediction.test_index.tolist()!=list(range(706)):
            raise ValueError("The frozen outer test set has changed")
        if arm_prediction.fold.nunique()!=5:
            raise ValueError("Missing outer fold")
        if y is None:
            y=arm_prediction.y_true.to_numpy(int)
        if not np.array_equal(y,arm_prediction.y_true.to_numpy(int)):
            raise ValueError("Label vectors differ among model arms")
    if len(y)!=706 or y.sum()!=177:
        raise ValueError("Unexpected historical cohort label count")
    if int(bounds.loc[bounds.partition=="outer_test","n_samples"].sum())!=706:
        raise ValueError("Irreducible-error report does not cover all tests")
    if bounds.loc[bounds.partition=="outer_test","fold"].nunique()!=5:
        raise ValueError("Missing fold-level label bound")
    base=m.set_index("arm").loc[ARMS[0]]
    quality=m.set_index("arm").loc["quality_95_train"]
    b=boots.set_index("arm").loc["quality_95_train"]
    all_auc_intervals_include_zero=bool(
        ((boots.auc_delta_ci95_low<=0)&(boots.auc_delta_ci95_high>=0)).all()
    )
    all_brier_intervals_include_zero=bool(
        ((boots.brier_delta_ci95_low<=0)&(boots.brier_delta_ci95_high>=0)).all()
    )
    fmt=lambda value: "n/e" if pd.isna(value) else f"{float(value):.4f}".replace(".",",")
    pct=lambda value: "n/e" if pd.isna(value) else f"{100*float(value):.1f}%".replace(".",",")
    lines=[
        "# LithiumScope — Fase E: unidad de sitio y calidad de los parches",
        "",
        "**Fecha:** 12 de noviembre de 2026. **Estado:** piloto ejecutado; no se promovieron modelos y no hay evidencia de validación externa independiente.",
        "",
        "## Pregunta científica",
        "",
        "¿Es suficiente representar múltiples muestras geoquímicas mediante una etiqueta por sitio, excluir imágenes incompletas o restringir la representación espectral al centro de la imagen para mejorar la estabilidad del Modelo 2? Se evaluaron por separado los cambios y una combinación exploratoria.",
        "",
        "La auditoría de Fase D detectó 25 sitios históricos con 77 observaciones que proporcionaban **exactamente el mismo parche** a pesar de tener etiquetas Li alta/baja incompatibles y 16 parches históricos con menos del 95% de píxeles finitos. Ninguna de estas alertas demuestra por sí sola que la medición geoquímica sea errónea ni establece un afloramiento físico.",
        "",
        "## Protocolo experimental",
        "",
        "Se mantuvieron **706 ejemplos históricos de evaluación, cinco folds geográficos congelados y 177 etiquetas positivas**, con umbral Li q75 calculado solo usando las muestras históricas del entrenamiento de cada fold. Se excluyeron de todas estas pruebas las 77 muestras GEOROC provisionales, porque la pregunta era aislar los efectos de representación y QA de la base histórica.",
        "",
        "Para reducir coste, la referencia se reutilizó exactamente desde el piloto histórico de Fase C; se verificaron SHA-256 del manifiesto, caché, folds y archivo OOF. Se entrenaron únicamente 25 estimadores transitorios: cinco variantes nuevas × cinco folds, todos con CatBoost de 160 iteraciones, profundidad cinco, aprendizaje 0,065, regularización L2=5 y semilla 42. No se cambiaron hiperparámetros en función de resultados.",
        "",
        "Las variantes por sitio deciden una clase usando solo **Li de TRAIN**: mediana numérica por sitio o mayoría estricta (los empates se excluyen solo del entrenamiento). Un sitio aporta una sola fila espectral si todos sus parches coinciden en SHA-256; una inconsistencia genera error bloqueante, no promediación silenciosa. Esta política es una hipótesis de etiquetado, no un certificado de identidad litológica.",
        "",
        "En la variante de calidad, únicamente filas de entrenamiento con menos de 95% de valores finitos se retiran; las 706 observaciones de test se mantienen, incluso las 16 incompletas. Para el centro, se extraen 126 características sobre la ventana fija [8:24, 8:24] de cada imagen original de seis bandas; se aplica exactamente la misma transformación a train y test. No se utilizan los diez casos conocidos para elegir variantes.",
        "",
        "## Resultados pareados sobre los mismos 706 test (OOF exploratorio)",
        "",
        "| Variante | ROC-AUC ↑ | Average Precision ↑ | Brier ↓ | Recall con score >= 0,5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        x=m.set_index("arm").loc[arm]
        lines.append(
            f"| {DESCRIPTIONS[arm]} | {fmt(x.auc)} | {fmt(x.ap)} | {fmt(x.brier)} | "
            f"{pct(x.recall_at_threshold)} |"
        )
    lines.extend([
        "",
        f"**La única variante con una mejora descriptiva simultánea en ROC-AUC, AP y Brier fue el filtro de calidad del TRAIN al 95%:** cambios AUC {fmt(quality.auc-base.auc)}, AP {fmt(quality.ap-base.ap)} y Brier {fmt(quality.brier-base.brier)}. El recall a umbral fijo de 0,5 continúa bajo y no se optimizó aquí. **Estas diferencias no equivalen a una mejora confirmada de generalización.**",
        "",
        "Las políticas de mediana y mayoría redujeron AUC/AP al pasar de múltiples muestras a una fila por sitio. No por ello se descartan como principios científicos; no se han evaluado etiquetas respaldadas por geología de campo ni una función objetivo que represente heterogeneidad real.",
        "",
        "La ventana central de 16×16 también disminuyó las métricas agregadas en este ensayo: reducir el tamaño espacial sin una evidencia previa sobre exposición de roca no garantiza mayor correspondencia entre píxel y muestra.",
        "",
        "## Incertidumbre pareada por grupo histórico",
        "",
        "Se hicieron 320 remuestreos por grupos espaciales históricos (**66 grupos**), con intervalos exploratorios de 95% de diferencias entre la referencia y cada alternativa. El trabajo previo de fases A–D ya analizó esos datos; existen comparaciones múltiples y dependencia de escenas. No son pruebas confirmatorias ni un test externo.",
        "",
        "| Variante vs referencia | ΔROC-AUC | IC95% AUC | ΔAP | IC95% AP | ΔBrier | IC95% Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS[1:]:
        x=m.set_index("arm").loc[arm]
        bound=boots.set_index("arm").loc[arm]
        lines.append(
            f"| {DESCRIPTIONS[arm]} | {fmt(x.auc-base.auc)} | "
            f"[{fmt(bound.auc_delta_ci95_low)}, {fmt(bound.auc_delta_ci95_high)}] | "
            f"{fmt(x.ap-base.ap)} | [{fmt(bound.ap_delta_ci95_low)}, {fmt(bound.ap_delta_ci95_high)}] | "
            f"{fmt(x.brier-base.brier)} | "
            f"[{fmt(bound.brier_delta_ci95_low)}, {fmt(bound.brier_delta_ci95_high)}] |"
        )
    lines.extend([
        "",
        "**Todos los intervalos exploratorios de ΔAUC incluyen cero**. El filtro del 95% también tiene intervalos de ΔAP y ΔBrier que incluyen cero. No procede promocionar ningún candidato por estas cifras.",
        "",
        "## ¿Qué cambia cuando evaluamos por sitio?",
        "",
        "La evaluación por sitio se calcula a posteriori como un análisis descriptivo adicional, con una puntuación mediana de cada sitio en el fold test y etiqueta de la mediana de Li usando el umbral del fold. No se utiliza para fijar la política de entrenamiento.",
        "",
        "| Variante | Sitios test | ROC-AUC por sitio | AP por sitio | Brier por sitio |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        row=bysite.loc[(bysite.arm==arm)&(bysite.subset=="all_sites")].iloc[0]
        lines.append(
            f"| {DESCRIPTIONS[arm]} | {int(row.n_sites)} | {fmt(row.roc_auc)} | "
            f"{fmt(row.ap)} | {fmt(row.brier)} |"
        )
    train_bounds=bounds.loc[bounds.partition=="train"]
    test_bounds=bounds.loc[bounds.partition=="outer_test"]
    bound_min=int(test_bounds.minimum_unavoidable_binary_errors.sum())
    sites_mixed=int(test_bounds.mixed_sites.sum())
    lines.extend([
        "",
        f"Se detectaron **{sites_mixed} sitios de test con etiquetas alto/bajo mezcladas** al utilizar el umbral Li específico de su fold, y al menos **{bound_min} errores de clasificación inevitables** en esos sitios si la función solo recibe una misma entrada espectral por ubicación. La cota se calcula como suma de la clase minoritaria por sitio entre las etiquetas test; **no** es el error mínimo para una región externa ni implica que todas las etiquetas sean falsas. Se diferencia de los 25 sitios / 28 etiquetas minoritarias de la Fase D porque allí se usó un corte fijo de 21,6 ppm y aquí cortes propios de folds.",
        "",
        "La reducción de etiquetas contradictorias no crea automáticamente una señal espectral capaz de distinguir litologías si el parche sigue mezclando materiales. Para decidir una política de sitio hacen falta mapas de afloramiento, posiciones GPS y evidencia geológica.",
        "",
        "## Riesgo de los 16 parches incompletos",
        "",
        "Los 16 registros históricos de cobertura espectral insuficiente permanecieron en los cinco test. La exclusión de QA afectó únicamente las muestras train y el número resultante se documenta en fold_metrics.csv.",
        "",
        "Los valores de evaluación del subgrupo de 16 están basados en **solo 3 positivos**; no son suficientes para concluir que el filtro detecta o predice bien muestras de baja calidad.",
        "",
        f"La auditoría adicional de máscaras reveló que las **16 observaciones contienen solo {missingness['distinct_image_bytes']} imágenes distintas**. En las 16, las **seis bandas comparten exactamente la misma máscara de no-datos por píxel**, lo que indica ausencia espacial correlacionada en lugar de una banda aislada. En {missingness['center_at_least_95pct_count']} observaciones el centro 16×16 supera el 95% de cobertura finita; en otras {missingness['center_below_95pct_count']} no. Por tanto, usar siempre el centro **no es una reparación general**.",
        "",
        "La investigación siguiente debe revisar las huellas de píxeles no válidos y la geometría/fecha de adquisición para saber por qué esas ventanas quedan parcialmente sin datos, y si la imputación por mediana introdujo patrones espurios. La readquisición, si se autoriza, debe preservar originales y repetir un protocolo temporal predefinido. Ningún píxel fue interpolado en esta auditoría.",
        "",
        "## Decisión metodológica",
        "",
        "1. **Retener como hipótesis** el filtro de calidad del 95% para un futuro experimento pre-registrado; no activarlo ni escogerlo basándose en este mismo conjunto de desarrollo.",
        "2. No imponer agregación automática por mediana o mayoría: estas dos políticas no resolvieron la relación espectro–Li en los 706 registros.",
        "3. Mantener la unidad de sitio en el diagnóstico, calcular incertidumbre por grupo/escena y añadir evidencia de afloramiento antes de evaluar una nueva etiqueta.",
        "4. Examinar el origen de los 16 parches incompletos; preparar reparación no destructiva con nuevas adquisiciones y comparar las 126 características, sin reemplazar las imágenes históricas.",
        "5. Para confirmar una mejora, reunir etiquetas independientes y verificadas por superficie, separadas geográficamente desde el origen, y después repetir la comparación con los procedimientos previamente congelados.",
        "",
        "## Evidencia reproducible",
        "",
        "El runner opt-in es src/lithiumscope/model_2/methodology/site_patch_lab.py y produce experiment_design.json, central_16x16_features.csv, train_site_aggregation_decisions.csv, all_arms_paired_oof.csv, fold_metrics.csv, sample_quality_site_subgroups.csv, site_level_metrics.csv, identical_patch_label_error_bounds.csv, group_bootstrap_deltas.csv, paired_model_comparison.csv y experiment_summary.json.",
        "",
        "Ejecutar desde la raíz:",
        "",
        "    python -m lithiumscope.model_2.methodology.site_patch_lab --output results/scientific_simulations/m2_phase_e_20261112/pilot",
        "",
        "El runner coteja manifiesto, 126 características, folds, línea base y hashes de imágenes originales con Fases C/D antes de entrenar los candidatos temporales. No utiliza datos GEOROC provisionales, no cambia la función de inferencia del release, no guarda modelos de producción, no hace push.",
    ])
    target=out.parent/"INFORME_FASE_E_SITIOS_Y_PARCHES.md"
    report="\n".join(lines)+"\n"
    target.write_text(report,encoding="utf-8")
    if documentation is not None:
        documentation.parent.mkdir(parents=True,exist_ok=True)
        documentation.write_text(report,encoding="utf-8")
    valid={
        "state":"completed",
        "arm_count":len(ARMS),
        "unchanged_outer_test_samples":706,
        "positive_labels":int(y.sum()),
        "sites_in_test":int(bysite.loc[(bysite.arm==ARMS[0])&(bysite.subset=="all_sites"),"n_sites"].iloc[0]),
        "test_mixed_sites_fold_specific":sites_mixed,
        "irreducible_minimum_test_labels_fold_specific":bound_min,
        "all_auc_intervals_include_zero":all_auc_intervals_include_zero,
        "all_brier_intervals_include_zero":all_brier_intervals_include_zero,
        "quality_95_delta_auc":float(quality.auc-base.auc),
        "quality_95_delta_ap":float(quality.ap-base.ap),
        "quality_95_delta_brier":float(quality.brier-base.brier),
        "no_external_holdout":True,"promotion":False,
        "source_evidence_only":True,
        "missingness_sample_count":missingness["sample_count"],
        "missingness_shared_six_band_masks":missingness["all_six_band_masks_equal_count"],
        "missingness_center_below95":missingness["center_below_95pct_count"],
        "report":str(target),
    }
    (out/"report_validation.json").write_text(json.dumps(
        valid,ensure_ascii=False,indent=2),encoding="utf-8")
    return valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",required=True)
    p.add_argument("--docs",required=True)
    args=p.parse_args()
    ab=lambda path:Path(path) if Path(path).is_absolute() else PROJECT_ROOT/path
    result=render(ab(args.pilot),documentation=ab(args.docs))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
