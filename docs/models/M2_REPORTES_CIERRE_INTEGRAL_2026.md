# LithiumScope — informes finales integrados en la competencia del Modelo 2

**Fecha:** 12/11/2026. **Estado:** preparados y comprobados offline sobre los OOF existentes de 783 muestras. No se reentrenó el modelo en esta preparación y no se crea release individual.

Este documento complementa M2_ENTRENAMIENTO_EXPERIMENTAL_CIERRE_2026.md. La opción original Entrenamiento → 2. Modelo 2 sigue ejecutando los seis algoritmos (Random Forest, Extra Trees, HGB, XGBoost, CatBoost y SVM-RBF), con Optuna, cinco folds espaciales, checkpoints, ranking tradicional y ajuste final del ganador no activo. Los cambios descritos aquí son exclusivamente de **evaluación posterior al entrenamiento**, no de selección o ajuste de modelos.

## 1. Comparación verificable con el release anterior

Al completar la validación cruzada de los nuevos 783 registros, el posprocesador comprueba y toma de la ejecución manual experimental del 12/11 las **706 predicciones OOF del protocolo de CatBoost usado en el release de septiembre, reentrenado sobre los folds históricos**. Su integridad queda fijada por SHA-256 y se verifican ID, orden, concentración observada y número de fold. La metadata real del release confirma CatBoost con 706 muestras y un umbral fijo de Li de 21,6 ppm.

Para una comparación descriptiva en la **misma población de 706 casos**, las puntuaciones de ese protocolo anterior y las puntuaciones de cada algoritmo nuevo se reevalúan con una etiqueta de Li alta definida **por el mismo corte fijo 21,6 ppm**. Se generan ROC-AUC, Average Precision, Brier, Balanced Accuracy, precision, recall y F1 a corte de score 0,5, intervalos exploratorios agrupados y priorización de tres sitios sobre ternas idénticas.

**Limitación metodológica imprescindible:** el artefacto publicado en septiembre NO se usa para predecir registros que pudieron ser su propio entrenamiento. La referencia es el **procedimiento CatBoost reentrenado**, no exactamente el estimador binario del release. El protocolo viejo se entrenó usando 706 muestras y el nuevo usa 783 disponibles con filtro train-only; también pueden diferir los umbrales de etiqueta internos de entrenamiento. La comparación tiene casos y criterio de evaluación equivalentes, pero NO entrenamientos idénticos ni un test externo independiente. Se declaran ambas diferencias en el informe.

Productos:
- old_release_protocol_comparison_706.csv
- old_release_protocol_paired_706.csv
- old_release_protocol_intervals_706.csv
- old_release_protocol_three_zone.json y old_release_protocol_three_zone_cases.csv
- old_release_protocol_comparison_summary.json y OLD_RELEASE_PROTOCOL_COMPARISON.md

## 2. Informe de calidad y limitaciones por algoritmo

Se reconcilian las 783 muestras originales, sin excluir las 77 GEOROC provisionales del análisis, con auditorías ópticas y geoquímicas ya ejecutadas. La auditoría incluye:
- cobertura espectral conjunta por muestra y filtro train-only del 95 %, distinguiendo claramente del TEST completo;
- clasificación SCL local y abstenciones ópticas (clase 5 no es confirmación de roca);
- fracción de características fuera del rango histórico percentil 1–99 y alerta descriptiva mayor al 25 %;
- 25 sitios históricos con Li alto/bajo contradictorio sobre la misma posición bajo corte de 21,6 ppm;
- procedencia histórica/GEOROC, regiones latitudinales y métricas de cada algoritmo por estrato;
- evidencia de 111 escenas STAC, sin cambiar ningún archivo NPY, etiqueta o manifiesto.

Los estratos pequeños pueden quedar sin AUC o AP estimable. Nada de lo anterior sustituye comprobaciones de terreno ni autoriza promocionar un modelo.

Productos:
- scientific_sample_audit_783.csv
- scientific_stratified_metrics.csv
- scientific_diagnostics_summary.json
- SCIENTIFIC_DIAGNOSTICS.md

## 3. Reporte obligatorio y recuperable de tres zonas

Después de que la competencia guarde los OOF por algoritmo, el posprocesador utiliza esos archivos y los mismos folds. Construye **250 ternas por fold** de **tres ubicaciones distintas del mismo fold de evaluación**, aplica las mismas ternas a todos los algoritmos y mide el porcentaje de veces en que el score más alto coincide con el mayor Li observado por sitio (mediana en posiciones repetidas).

La métrica de ternas se incorpora como columna adicional a competition_ranking.csv y a las tablas Excel, pero **no cambia el criterio de ganador existente** (ROC-AUC, AP y Balanced Accuracy de desempate). Las métricas tradicionales, gráficos, logs y checkpoints siguen intactos.

Productos:
- three_zone_competition_cases.csv
- three_zone_competition_summary.json
- columna three_zone_top1_accuracy y hoja three_zone_cases en Excel si se generó el reporte.

Para no perder horas de entrenamiento, una falla de un informe NO elimina los modelos ni sus OOF. **No se silencia el error:** evidence_report_status.json y EVIDENCE_REPORT_STATUS.md informan explícitamente qué reporte está completo y cuál requiere reparación. El tracker también registra el evento de postprocesamiento. Un reporte incompleto no debe presentarse como exitoso.

## Regenerar informes sin entrenar nuevamente

Desde CMD en la raíz del repositorio, una vez que finalice una competencia:

    "D:\Librerias\Python312\python.exe" -m lithiumscope.model_2.training.postprocess --manifest "data/processed/model_2/vnext_expanded/training_manifest.csv" --tables "results/model_2/runs/training_YYYYMMDD_HHMMSS_m0300/tables" --output "results/model_2/runs/training_YYYYMMDD_HHMMSS_m0300/tables"

Cambiar training_YYYYMMDD_HHMMSS_m0300 por el ID real del entrenamiento. Este comando **no entrena**, no sobrescribe candidatos y no activa modelos. Se imprime también automáticamente dentro de evidence_report_status.json con rutas reales cuando hay que reparar un informe.

## Verificación

Se ejecutó el módulo de posprocesamiento sin entrenamiento sobre las predicciones originales de la competencia completa del 11/11/2026, dejando el resultado únicamente en results/scientific_simulations/m2_postprocess_validation_20261112. Los tres informes terminaron con estado complete. Las seis familias de algoritmos aparecen en los resultados.

Estos resultados de comprobación **no son métricas de un nuevo entrenamiento**: se utilizan para verificar el cableado del informe antes de que el usuario ejecute manualmente la competencia completa.

No push, no tag, no GitHub Release. Los anteriores candidatos y el Modelo 2 activo permanecen inalterados.
