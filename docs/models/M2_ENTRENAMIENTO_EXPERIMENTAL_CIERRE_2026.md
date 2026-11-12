# LithiumScope — cierre operativo del Modelo 2 experimental

**Fecha:** 12 de noviembre de 2026. **Estado:** candidato experimental descartable; NO se prepara release individual. El release futuro, si se autoriza, evaluará conjuntamente un Modelo 1 nuevo y un Modelo 2 corregido. El modelo del release anterior continúa activo sin cambios.

## Entrenamiento manual en el menú

Entrar a **Entrenamiento → 2. Modelo 2**. La opción visual del menú se mantiene, pero ahora ejecuta el flujo experimental corregido: CatBoost CPU, parámetros fijos, cinco folds espaciales, filtro QA y comparación del procedimiento de release anterior. No utiliza la competencia anterior de seis algoritmos ni descarga GEOROC.

La opción **Entrenar ambos** conserva el entrenamiento habitual del Modelo 1 y utiliza este entrenamiento experimental para el Modelo 2. Para el propósito actual se debe elegir únicamente la opción 2.

Implementación: src/lithiumscope/model_2/training/closure.py

Preflight que **no entrena**:

    python -m lithiumscope.model_2.training.closure --preflight --output results/scientific_simulations/model2_manual_ready_20261112

Simulación reducida que **no crea modelo candidato**:

    python -m lithiumscope.model_2.training.closure --smoke --output results/scientific_simulations/model2_manual_ready_20261112/smoke

No ejecutar la CLI sin esos flags hasta el entrenamiento manual, porque inicia el entrenamiento completo.

## Protocolo científico fijado

- Datos: las mismas **706 observaciones históricas Mamani09/Sentinel-2**. No participan las 77 etiquetas GEOROC provisionales.
- **126 características espectrales DN históricas**, sin sustituirlas automáticamente por reflectancia corregida. Los ensayos radiométricos mejoraron algunas métricas y deterioraron otras; modificar esa representación sin validación separada produciría otra fuente de variación.
- **Filtro de cobertura finita conjunta >=95 %** aplicado solo a cada conjunto TRAIN. El conjunto TEST exterior **mantiene las 706 observaciones, incluidas las 16 imágenes incompletas**. El ajuste final utiliza **690/706**.
- **Cinco folds espaciales originales congelados**; cada umbral alto/bajo de Li se calcula con el cuantil 0,75 del Li histórico del TRAIN de ese fold, nunca con el test.
- **CatBoost CPU** (4 hilos), 160 iteraciones, profundidad 5, tasa 0,065, L2 5, semilla 42, pérdida Logloss. Sin Optuna, hiperparámetros optimizados sobre test ni mezcla de nuevas escenas.
- El candidato solo predice un score **espectral relativo**. Si una nueva imagen tiene menos de 95 % de píxeles válidos conjuntamente en las seis bandas, la inferencia experimental responde con **abstención** en lugar de inventar una prioridad.
- Persisten los 25 sitios históricos con etiquetas conflictivas sobre parches idénticos. No se han eliminado, reetiquetado ni considerado terreno confirmado.

## Comparación contra el release anterior

Referencia original activa: models/model_2/trained/training_20260920_195933_m0300/model.joblib

Se reconstruye el **protocolo de CatBoost** del release anterior dentro de los mismos cinco folds, con todos los ejemplos históricos de TRAIN. No se evalúa directamente el artefacto final en muestras de su propio entrenamiento como si fuesen test independientes.

Parámetros recuperados del release: 669 árboles, profundidad ocho, learning rate 0,010818..., L2 1,54029..., semilla 42, CPU. El protocolo corregido se evalúa sobre exactamente los mismos casos históricos. Se reportan ROC-AUC, AP, Brier, Balanced Accuracy, recall, métricas por fold, deltas e intervalos exploratorios por grupos.

Se mantiene una referencia CatBoost secundaria de los experimentos anteriores para cotejar reproducibilidad, sin confundirla con el modelo del release.

## Priorizar tres zonas

Se eligen 250 ternas por fold de sitios diferentes del mismo conjunto TEST. Cada modelo debe identificar cuál tiene la mayor mediana de Li observada. Se conserva idéntica selección de ternas en todos los modelos.

Prueba exploratoria completada antes del nuevo entrenamiento:
- Referencia histórica pequeña: **43,09 %** de elecciones top-1 correctas, 1244 ternas.
- Mismo modelo con filtro QA95: **42,93 %**, mismas ternas.
- Elección aleatoria: aproximadamente **33,33 %**.

Por lo tanto, existe señal potencialmente útil frente al azar en este conjunto histórico, pero el filtro del 95 % por sí solo **no ha demostrado una mejora al elegir la mejor de tres zonas**. El entrenamiento manual informará la nueva comparación con el procedimiento del release de septiembre, sin afirmar superioridad por anticipado.

El Modelo 1 ofrece una estimación de ppm complementaria. El Modelo 2 ordena según score espectral; no combina artificialmente ppm y score, ni convierte el ranking en una decisión de extracción.

## Criterios descriptivos para cerrar la evaluación

Se considerará una mejora interna frente al procedimiento del release cuando coincidan **las cuatro**: ROC-AUC mayor, Average Precision mayor, Brier no peor y acierto top-1 entre tres zonas no menor. El script escribe los resultados separados, los deltas y la evaluación de cada condición, aunque la conclusión sea negativa.

Incluso si supera esos criterios, los 706 casos ya participaron en el desarrollo. No se autoriza reclamar generalización externa ni publicar un Modelo 2 independiente sin otra evaluación futura.

## Evidencia generada por el entrenamiento manual

Carpeta: results/model_2/runs/training_...

- manual_closure_preflight.json: hashes, identidad de muestras, política de calidad y parámetros.
- all_reference_paired_oof.csv: puntuaciones y etiquetas sobre 706 casos idénticos.
- per_fold_comparison.csv, metrics_comparison.csv, uncertainty_vs_prior_release.json.
- three_zone_cases.csv y three_zone_metrics.json.
- experimental_acceptance.json e INFORME_ENTRENAMIENTO_MODELO2_EXPERIMENTAL.md.
- manual_training_summary.json y los logs de ejecución.

El artefacto corregido queda en models/model_2/trained/training_.../model.joblib, con activate=False y candidate_status.json marcado experimental_awaiting_review. No se incorpora a la lista de candidatos pendientes normales de promoción ni reemplaza el release activo.

El experimento anterior de 783 filas (training_20261111_170832_m0300) quedó etiquetado **discarded_experimental** en su candidate_status.json y se conserva solo para auditoría, sin alterar su archivo binario.

## Estado y límites

El entrenamiento completo **no se ejecuta durante esta preparación**. La tarea termina después de la prueba manual reducida y la suite completa de tests. No se generan nuevos experimentos geológicos ni nuevas fases para prolongar el trabajo. No se modifica el menú general, no hay push, tag, release ni cambio en el Modelo 1.
