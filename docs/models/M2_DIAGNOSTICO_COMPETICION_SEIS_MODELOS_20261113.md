# Modelo 2 — Diagnóstico de competencia y correcciones versionadas

**Fecha:** 13 de noviembre de 2026. **Estado:** auditor ejecutado, nuevas políticas de entrenamiento implementadas y probadas en modo unitario/smoke. **La competición completa con dichas políticas todavía no se ha ejecutado.** No existe aprobación para un release nuevo.

## Alcance: seis competidores, ninguna selección prefijada

La competencia normal mantiene las seis familias: Random Forest, Extra Trees, CatBoost, XGBoost, HistGradientBoosting y SVM-RBF. Gana el algoritmo con mejor ROC-AUC medio según la regla existente, con desempates en AP y Balanced Accuracy. El auditor **no escoge un ganador**, solo informa: matrices VP/VN/FP/FN, AP, ROC-AUC, Brier, error positivo/negativo, calibración, procedencia, regiones, OOD, QA, folds y comparaciones pareadas.

- El modelo M1 y su información geoquímica **no entran como características de M2**. M2 conserva 126 descriptores de seis bandas Sentinel-2.
- El release CatBoost `training_20260920_195933_m0300` se cargó con `joblib` y verificó SHA-256 `e41b275d5c0dfff6354e1f21b56d3cf1f4c9751abdb9415e50a571bdca14a45c`, 126 características y predictor CatBoost válido.
- **No** se lo puntúa sobre sus propias muestras de entrenamiento para declarar CV. Su comparador válido es la reconstrucción fuera de fold de los parámetros del protocolo de septiembre; esta comparación **NO** es una nueva validación externa del binario publicado.

## Matriz de confusión: última competencia, umbral común 0,5

Folds espaciales existentes; 783 casos de test OOF, 195 positivos y 588 negativos.

| Algoritmo | VP | VN | FP | FN | AUC OOF | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 90 | 432 | 156 | 105 | 0,6538 | 0,2089 |
| Extra Trees | 82 | 455 | 133 | 113 | 0,6408 | 0,2067 |
| CatBoost | 31 | 553 | 35 | 164 | 0,6336 | 0,1905 |
| XGBoost | 36 | 541 | 47 | 159 | 0,6086 | 0,2046 |
| HistGradientBoosting | 34 | 538 | 50 | 161 | 0,5777 | 0,2114 |
| SVM-RBF | 1 | 588 | 0 | 194 | 0,5981 | 0,1825 |

El **umbral 0,5 es fijo, común y descriptivo**: no se eligió para optimizar test. Los umbrales operativos futuros requieren selección exclusivamente en entrenamiento interno y evaluación separada. Una matriz de confusión no equivale a la calidad del ranking ni implica que un score sea probabilidad de encontrar un depósito.

## Qué mejoró y qué empeoró: diagnóstico de raíz

Comparación de **las mismas 783 predicciones de test OOF, mismas etiquetas/folds** entre 11/11 y 12/11. Cambió QA espectral aplicado solo a TRAIN (95 % de cobertura); el test original se conservó. No puede asignarse causalidad por una sola comparación, porque Optuna y efectos del entrenamiento también pueden intervenir.

| Algoritmo | Δ AUC | Δ AP | Δ Brier (negativo = mejor) |
|---|---:|---:|---:|
| Random Forest | +0,0231 | +0,0130 | −0,0046 |
| Extra Trees | +0,0337 | +0,0185 | −0,0035 |
| CatBoost | +0,0089 | +0,0052 | −0,0048 |
| SVM-RBF | +0,0421 | +0,0419 | −0,0039 |
| HistGradientBoosting | +0,0258 | +0,0296 | −0,0123 |
| XGBoost | −0,0097 | −0,0243 | −0,0050 |

La regresión principal no es que RF empeorara su Brier frente a su propia versión anterior: **RF sí mejoró Brier respecto a su versión anterior**, pero, en la competencia nueva, CatBoost tiene menor Brier y RF mejor ranking/recall. Frente a CatBoost del 12/11, RF muestra una diferencia media Brier +0,01843: en positivos mejora por −0,27588 y en negativos empeora +0,11604. RF produce FP frecuentes (156/588 negativos), a diferencia de CatBoost (35/588). La ponderación de clases `balanced` en RF y Extra Trees es una posible causa a probar; **no se asume demostrada**.

Para scores RF ≥0,5: 246/783 muestras, score medio 0,603 frente a frecuencia positiva real 0,366. Este desajuste motiva calibración espacial predefinida y revisión de las clases, sin reinterpretar el score como certidumbre geológica. En latitudes 20°S a 30°S (96 casos, 50 positivos), RF AUC ≈0,459: **limitación regional** que demanda evaluar representatividad y calidad de etiquetas; no inventar datos para corregirla.

## Comparación pareada frente al protocolo del release de septiembre

Archivo `historical_release_comparison_six_algorithms.csv`: los mismos 706 casos, folds, IDs, grupos y **etiqueta binaria fija del protocolo publicado**. Para el ganador RF del 12/11: Δ AUC +0,02276, Δ AP +0,01720, pero Δ Brier +0,02675. El comparador se **reentrenó por fold** usando las especificaciones del release; no se puntuó el artefacto final publicado sobre sus ejemplos de ajuste. Esta evidencia es interna y reutilizada durante desarrollo. Las 77 muestras GEOROC siguen siendo provisionales.

## Correcciones implementadas en el flujo completo

1. **Diagnóstico durable y recuperable:** `training/diagnostic_audit.py`, `diagnostic_inputs.py`, `diagnostic_metrics.py` y `diagnostic_release_comparison.py`; etapa `competition_diagnostics` incorporada a `training/postprocess.py`. Se regenera de OOF sin reentrenar y sin PDF. Crea:
   - `confusion_matrices_all_algorithms.csv`: VP, VN, FP, FN y tasas, AUC/AP/Brier por algoritmo.
   - `confusion_by_region_quality_ood_fold.csv`: matrices y tasas por fold, procedencia, calidad espectral, SCL, OOD, latitudes y sitios con etiquetas contradictorias.
   - `sample_error_diagnostics.csv`: casos FP/FN, scores, delta de error frente a versión anterior, grupo y fold.
   - `calibration_reliability_bins.csv`: diez bins fijos score vs frecuencia positiva real.
   - `paired_competition_deltas.csv`: diferencias por algoritmo con bootstrap pareado por grupos geográficos.
   - `historical_release_comparison_six_algorithms.csv` y `diagnostic_manifest.json`: comparador histórico y procedencia.
2. **Calibración espacial de todos los algoritmos:** `training/calibration_policy.py` agrega `spatial_sigmoid`, que usa grupos `GroupKFold` y SOLO TRAIN para ajustar una sigmoide en cada candidato; también existe el modo histórico `legacy`. Cuando se habilite experimentalmente, se aplicará a los seis antes de predecir TEST y al modelo ganador final. No fusiona modelos ni usa química como feature. **Por ahora permanece desactivada** (modo `legacy`): la prueba de integración reducida sobre 598 TRAIN y 172 TEST reales del fold 1 produjo 0 VP y 45 FN en los seis modelos al umbral 0,5. Esto no certifica la nueva competición, pero **impide habilitar calibración indiscriminadamente sin validar un umbral seleccionado SOLO en TRAIN y sin controlar el recall**.
3. **Corrección de falsos positivos bajo hipótesis, NO ganancia declarada:** perfiles nuevos y versionados de Optuna para **RF `regularized_v3` y Extra Trees `regularized_v2`** permiten elegir `class_weight=None` o `balanced`, con límites conservadores de profundidad y muestras por hoja. Los otros cuatro algoritmos conservan su búsqueda previa. Esto responde al sesgo observado sin favorecer al RF.
4. **Competición y ganador:** `config/model_2.yaml` conserva la calibración predeterminada `legacy` por seguridad, ofrece `spatial_sigmoid` como experimento optativo, activa perfiles versionados RF/Extra Trees y preserva el mismo criterio de competencia; sus cambios invalidan la firma de las ejecuciones anteriores y evitan reutilizar estudios Optuna incompatibles. El menú no se altera.
5. **Release gate:** los reportes guardan las matrices, trazabilidad y diferencias; el mecanismo científico existente mantiene `release_gate.pass=False` sin evidencia de un conjunto independiente. **No activar ni publicar automáticamente**. La métrica experimental de ternas de M2 aisladas NO sustituye la evaluación M1→3→M2→1.

Los experimentos previos de ganancia sintética sobre DN no mejoraron discriminación; **no** se habilita esa aumentación por defecto. Introducir réplicas sintéticas del mismo sitio tampoco crea nuevas ubicaciones independientes.

## Criterio de cierre

El **código diagnóstico** y la integración de políticas pueden aprobarse con pruebas reproducibles. La **selección del release definitivo sigue pendiente** hasta completar la competición real bajo la nueva política, verificar si cualquier algoritmo supera de manera respaldada al anterior y contrastarlo con evidencia independiente cuando exista. Si la ganancia no se demuestra, conservar el release publicado es un resultado válido; no maquillar la métrica ni iterar indefinidamente sobre los mismos folds.

**Restricciones:** sin modificar el Modelo 1, menú interactivo, PDF, registro de modelos activos, datos originales o checkpoints previos; commits locales, sin push.

## Comando de regeneración de diagnóstico sin entrenamiento

Desde la raíz del clon, en CMD:

    set "PYTHONPATH=%CD%\src"
    "D:\Librerias\Python312\python.exe" -m lithiumscope.model_2.training.diagnostic_audit --manifest data/processed/model_2/vnext_expanded/training_manifest.csv --tables results/model_2/runs/training_20261112_123451_m0300/tables --output results/scientific_simulations/m2_six_algorithm_diagnostics_20261113

Resultados: `results/scientific_simulations/m2_six_algorithm_diagnostics_20261113/`.
