# Modelo 1 — Auditoría integral de datos, procedimientos, selección y modelos

**Fecha:** 13-11-2026. **Tipo:** diagnóstico técnico retrospectivo y hoja de ruta; no se cambia el release. **Ámbito:** M1 histórico (1.120) y experimental (3.420), no M2. Complementa `M1_DEFENSA_TESIS_METRICAS_DATOS_Y_RELEASE_20261113.md` y `M1_CUATRO_FASES_DIAGNOSTICO_RELEASE_VS_CANDIDATO_20261113.md`.

## 1. Dictamen

**El candidato ampliado no está suficientemente validado para desplazar al release M1. Tampoco está descartado.** Sus 3.420 registros permiten estudiar rangos más amplios; el análisis emparejado de 1.004 registros históricos mostró reducciones exploratorias de RMSE de 26,091 a 21,469 ppm y de MAE de 8,333 a 7,258 ppm, pero 610 observaciones cambiaron de fold y no constituye un holdout externo común. El SVM experimental presenta **R² TRAIN promedio ≈0,286** y **R² TEST ≈0,247** entre cinco folds: hay indicios de capacidad insuficiente para ciertos dominios, pero también desplazamiento de distribución; NO es legítimo concluir «underfitting universal» con esos dos valores.

**Hay una decisión metodológica pendiente:** el ganador se determina por **RMSE medio de los folds**, no por R²/recall de altas concentraciones ni RMSE OOF agrupado. En las 3.420 predicciones OOF el SVM obtiene **RMSE 159,846 / R² 0,043**, mientras XGBoost obtiene **RMSE 140,128 / R² 0,265**. La regla de selección es reproducible y fue aplicada correctamente; puede ser inadecuada para el objetivo final. El rendimiento de un conjunto de datos más heterogéneo no puede compararse por RMSE bruto con una fuente de litio recortada.

## 2. Evidencia sobre fuentes y filtros

- Histórico de 1.120: **676 Mamani09 + 444 GEOROC**, 100 grupos espaciales.
- Candidato de 3.420: **676 Mamani09 + 2.744 GEOROC**, 244 grupos, 33 predictores. Se incorporaron **2.300 filas GEOROC netas** respecto al histórico.
- En el candidato, **2.329 registros** contienen `Fe_total_provenance=FEOT_CONVERTED_TO_FE2O3T`; **415** conservan `ORIGINAL_FE2O3`. Esto es una intervención **de armonización FeOT total (expresado como FeO) a Fe₂O₃ equivalente mediante factor 1,111348**, no solo una adquisición de nuevas muestras. No se ha aislado todavía cuántas filas adicionales son atribuibles causalmente a cada filtro.
- En el conjunto ampliado el procesamiento pasó de **4.765 registros combinados** a **3.692 con etiqueta Li válida** y a **3.420 luego del QC** de suma de óxidos 94–102. El antiguo, con reglas previas de armonización, también partió de 4.765 y 3.692 válidos, pero el QC retuvo 1.120. **Una política de admisión cambió de forma sustantiva el dominio:** no asumir que las 2.300 filas adicionales son independientes o de idéntica calidad.
- Entre las 3.420 observaciones, mediana de Li **15 ppm**, Q99 **240 ppm**, máximo **4.160 ppm**, **89** por encima de 100 ppm y **8** por encima de 1.000 ppm. En la fuente histórica: mediana 16,1, Q99 98,97, máximo 597,1, 12 >100 y 0 >1.000 ppm.
- Los ocho casos >1.000 ppm están en una zona GEOROC de aproximadamente **latitud −14,0°, longitud −70,67°**, principalmente identificadores **2021-MAC**; algunos comparten coordenadas y corresponden al mismo entorno geológico. **No tenemos pruebas de que sus etiquetas estén equivocadas.** Trazar fichero fuente, muestra, publicación, análisis de laboratorio, unidades, tipo de material, lote y posibles repeticiones antes de editar cualquier valor.

## 3. Problemas de representatividad y de independencia

**Concentración espacial extrema:** varias observaciones diferentes comparten coordenadas; hay celdas con 107, 60 o 47 filas y coordenadas repetidas con valores de Li distintos. No son necesariamente duplicados (podrían ser muestras/mediciones distintas), pero **no equivalen automáticamente a observaciones geográficas independientes**. La regla actual de deduplicación une identificador de muestra y coordenadas redondeadas: por diseño **conserva distintas muestras en la misma ubicación**.

**Validación espacial incompleta:** GroupKFold mantiene disjuntos los grupos cuadrados de 0,5°; sin embargo, una auditoría de distancia haversine en los TEST del candidato encontró **24 observaciones a menos de 1 km de al menos una observación TRAIN**, repartidas entre los cinco folds (0, 5, 1, 5 y 13 respectivamente); la distancia mínima observada fue ≈0,154 km. Eso NO demuestra fuga de etiquetas, pero significa que el esquema de celdas no impone **buffer geográfico**. Debe ensayarse exclusión por 5, 10 y 20 km, con tamaños de TEST y representatividad documentados.

**Variables:** de las 33 variables seleccionadas, 28 son numéricas y 5 categóricas. Incluyen coordenadas de latitud/longitud y variables geoquímicas. Las imputaciones/normalizaciones se realizan mediante `sklearn.pipeline` dentro del ajuste de cada fold; no se constató en esta revisión una fuga por imputación global. Para validar verdadera transferencia a regiones nuevas deben contrastarse modelos con y sin coordenadas, evitando que memoricen ubicación. Los registros de procedencia, categorías y FeOT deben auditarse por cohortes.

## 4. Diagnóstico de capacidad y errores reales

| Indicador SVM | Histórico 1.120 | Candidato 3.420 | Interpretación |
|---|---:|---:|---|
| MAE TRAIN medio | 3,691 ppm | 12,163 ppm | TRAIN nuevo más heterogéneo |
| MAE TEST medio | 8,128 ppm | 16,320 ppm | TEST distinto; no comparación causal |
| R² TRAIN medio | 0,569 | 0,286 | Candidato explica poco incluso en TRAIN |
| R² TEST medio | 0,070 | 0,247 | Mejor media por fold, pero distribución distinta |
| R² OOF agrupado | 0,089 | 0,043 | Agregación distinta de media de folds |
| Recall auxiliar Li alto, q75 TRAIN | 0,704 | 0,630 | Menor sensibilidad en el nuevo dominio |
| RMSE OOF agrupado | 25,685 ppm | 159,846 ppm | Dominado por colas extremas |
| XGBoost RMSE OOF agrupado | 26,625 ppm | **140,128 ppm** | En el candidato supera al SVM en este criterio |

El **fold 5 del candidato** concentra muestras de Li muy alto: SVM RMSE TEST **353,62**, R² TEST **0,0178**, frente a RMSE TRAIN **17,30**. Otro fold presenta RMSE TRAIN **159,79** y TEST **13,86**, por simple diferencia en dificultad del subconjunto. No interpretar todas las brechas RMSE como sobreajuste sin estratificar la distribución de Li.

**Revisión focal de extremos:** de los ocho valores reales >1.000 ppm (media real ≈3.180), el SVM predijo ≈70,9 ppm de media; XGBoost ≈524,8 ppm. Ninguno detectó los ocho por encima de 1.000 ppm con su predicción OOF. Para los 89 casos reales >100 ppm, el SVM obtuvo MAE ≈396,94 ppm. Esto respalda **subestimación de la cola extrema**, no prueba que los valores sean falsos ni que un modelo más complejo sea necesariamente la solución.

El 97,35 % de la suma de errores cuadrados SVM OOF proviene de los **diez mayores residuos**; en el histórico, 84,41 %. Considerar RMSE, MAE, mediana, P90 y estratos de Li siempre juntos.

## 5. Auditoría de procedimiento: qué está corregido y qué NO

| Control | Hecho observado | Riesgo pendiente |
|---|---|---|
| Validación | Optuna anidado y cinco folds GroupKFold espaciales | Buffers y holdout de procedencia/geología |
| Preprocesamiento | Imputación/escalado en pipeline del fold | Imputación sistemática por fuente, categorías desconocidas |
| FeOT | Conversión explícita + procedencia | Auditoría de unidades, QC condicionado y ablation aislada |
| Filtro Li | No se recortó Li por percentiles | Dominio extremo, unidades y ruido de análisis |
| Duplicados | Eliminación por identificador+coordenadas | Pseudorreplicación de sitios y análisis múltiples |
| Selección ganador | Menor RMSE medio por fold | Objetivo de ranking y pérdidas en altas concentraciones |
| Matriz auxiliar | q75 de TRAIN por fold, seis modelos | No constituye calibración ni evaluación externa |
| Generalización | Curvas TRAIN/TEST para SVM y OOF de seis familias | Holdout nuevo y repeticiones controladas |

**Hipótesis separadas:** (H1) las colas altas causan la mayor parte de la pérdida cuadrática (**respaldo descriptivo**, no causal); (H2) la armonización FeOT cambió la selección de muestras y la distribución (**hecho** para la selección, su efecto predictivo no aislado); (H3) la cercanía entre celdas hace optimista parte de la CV (**riesgo**, no fuga demostrada); (H4) el SVM subajusta litio extremo (**compatible**); (H5) la función de selección por RMSE medio penaliza una opción con mejor RMSE OOF global (**hecho de agregación**, no selección errónea); (H6) hay etiquetas/unidades GEOROC defectuosas (**NO demostrado**).

## 6. Plan de corrección antes de otro entrenamiento global

1. **Auditoría inmutable de datos crudos:** listar con procedencia exacta los casos >100 y >1.000 ppm, fuentes de Li, unidades, identificadores repetidos, coordenadas compartidas, lotes y presencia de FeOT. Generar reporte de consistencia y no cambiar etiquetas sin respaldo.
2. **Ablación controlada de armonización:** con un **mismo TEST congelado** de históricos, comparar fuente antigua, FeOT convertido y datos adicionales, manteniendo algoritmo, Optuna, tiempo, clases y folds fijados. Evitar que filtros de calidad eliminen selectivamente los casos difíciles del TEST principal.
3. **Validación espacial con buffers:** repetir exclusión de vecinos TRAIN dentro de 5/10/20 km y análisis de generalización por bloque/región; cuantificar cuántas muestras quedan.
4. **Objetivo operacional:** definir antes de TEST si M1 busca minimizar RMSE en ppm, clasificar «Li alto», o seleccionar Top-3 de zonas; optimizar y elegir con TRAIN anidado. Comparar SVM, XGBoost y CatBoost según la función predefinida; sin escoger retrospectivamente al que gane una métrica.
5. **Promoción solo con evidencia nueva:** comparar candidato y release sobre un holdout verdaderamente independiente. Si no es viable, declarar release provisional/de investigación con limitaciones explícitas; no declarar victoria científica externa.

**No autoriza modificar el release, cambiar la política de calidad ni crear nuevas etiquetas automáticamente.** Mantener commits locales, sin menús ni modelos activados.

## 7. Evidencia localizada

- `results/scientific_simulations/m1_dual_diagnostics_20261113/`: matrices, OOF de seis, brechas y curvas SVM.
- `results/scientific_simulations/vnext_20261106/stage5_training_comparison/m1_paired_comparison.json`: comparación histórica emparejada.
- `data/processed/model_1/training_combined.csv`, `data/processed/model_1/source_merge_audit.json`; `src/lithiumscope/datasets/georoc_ingestion_transform.py`; `config/model_1.yaml`.
- Modelo M1 activo: `models/model_1/trained/training_20260925_111245_m0300/model.joblib`. El experimental no está aprobado ni activado.
