# Bitácora de ejecución — Modelo 2 espectral v4

**Inicio del proceso:** 13 de noviembre de 2026. **Objetivo:** completar adquisición de cuatro bandas red-edge para los 706 históricos, ejecutar E1 (comparación con hiperparámetros congelados) y E2 (competencia Optuna de seis algoritmos), analizar nuevos sitios GEOROC solo si superan QA, comparar frente al protocolo CatBoost del release y documentar incluso resultados negativos.

## Invariantes obligatorias

- Seis algoritmos siempre: Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost y SVM-RBF.
- Las 126 variables espectrales originales se mantienen exactas; v4 añade 44 características físicas de B05/B06/B07/B8A, total 170.
- La salida M2 sigue orientada a elegir entre candidatos producidos por M1. No se integran características geoquímicas predictoras de M1.
- Los releases y el registro `models/active_models.json` no se modifican. No push; commits exclusivamente locales. No se toca menú.
- Mismas 706 muestras, etiquetas históricas, folds y QA train-only. Se evalúa sobre todo TEST sin filtrar observaciones por calidad.
- GEOROC adicional es provisional; no se crea certeza de terreno ni etiquetas sintéticas; sitios nuevos únicamente en TRAIN con distancias a TEST controladas.

## Procedimiento reproducible

1. Se verificó el estado anterior `5a3fb81` (M2 v4 implementado), registro de releases y caché: 3/706 parches históricos reales y 1 sitio nuevo GEOROC aceptado.
2. Se modificó exclusivamente `scripts/scientific/m2_rededge_v4_build.py` para permitir adquisiciones paralelas mediante `ThreadPoolExecutor`, con máximo de 16 conexiones, manteniendo orden de filas y caché SHA por muestra. Se comprobó Ruff.
3. Se inició la adquisición histórica con 8 workers:
   `python -u -m scripts.scientific.m2_rededge_v4_build --cohort 706 --workers 8`, usando `D:\Librerias\Python312\python.exe` y `PYTHONPATH=<raíz>\src`.
4. Se inició la adquisición GEOROC separada de hasta 29 grupos pendientes (sin incorporarlos automáticamente):
   `python -u -m scripts.scientific.m2_v4_new_sites --fetch 29`.
5. Se debe comprobar que el `build_status.json` indique `complete` y que el CSV de 170 dimensiones tenga exactamente 706 filas; de lo contrario no lanzar E1/E2.
6. E1: `python -m scripts.scientific.m2_rededge_v4_competition --cohort 706`. Reutiliza los parámetros históricos de cada fold y compara las seis familias con el v3 de 126 y CatBoost release reconstruido sobre los mismos 706 TEST.
7. E2: `python -m scripts.scientific.m2_rededge_v4_nested --cohort 706`. Nueva competencia interna Optuna, cinco folds externos y tres internos; conserva six-arm y QA.
8. E3: únicamente si existen al menos diez sitios geoquímicos e imágenes QA verificadas. El experimento `m2_v4_new_site_effect --min-new 10 --buffer-km 5` evalúa el efecto de nuevos grupos solo en TRAIN; de no reunir los requisitos, documentar la razón sin inventar datos.
9. Tras una fase **completa**, `python -m scripts.scientific.m2_v4_document_result --phase <fixed|nested|new_sites>` genera conclusiones respaldadas y registradas, aun cuando el release previo siga siendo superior.

## Evidencia de entrada y salida

- Entrada inmutable 706: `data/processed/model_2/training_manifest.csv` (SHA ya verificado en el estudio de 706).
- Extraídos: `results/scientific_simulations/m2_rededge_v4_20261113/patches/` y `cohort_706/`.
- Nuevos sitios: `results/scientific_simulations/m2_rededge_v4_20261113/new_sites/`, con motivos de rechazo, DOI y parches.
- E1/E2/E3: `comparison_706/`, `nested_170_706/`, `data_ablation_170_new_sites/`.
- Por fase: métricas de los seis algoritmos, OOF, fold, bootstrap espacial, matriz, tres zonas y conclusión científica.
- Referencia: `docs/models/M2_EXPERIMENTO_FINAL_ESPECTRAL_V4_20261113.md`.

## Estado de ejecución (actualizar con resultados reales)

- Adquisición histórica: **en ejecución**; última observación: 310/706 sin fallas.
- Recuperación de nuevos sitios: **en ejecución**; aceptación inicial 1, restante no confirmado.
- E1/E2/E3: **pendientes de datos completos**.
- Métricas v4 y decisión de release: **no disponibles todavía**, no atribuir ganancias.

## Primeros resultados reales: adquisición y E1

- Matriz 706 × 170 **completada**: 706 de 706 escenas descargadas, cero errores de adquisición. SHA-256 del CSV f4d1f5809fb1242f27d0bbc2348e9dab2e77a017ea77049090652929244a9513. QA95 690 elegibles para TRAIN; todos los 706 permanecen en TEST.
- Matriz 783 × 170 **completada**, incorporando las 77 fuentes GEOROC provisionales sin cambiar etiquetas. SHA-256 44aa5f7cb51ad92cdb8c3c6d7669573c3f930f3d8f3503194aed0d40461c9781.
- E1, parámetros congelados, seis algoritmos por cinco folds: 30 ajustes finalizados sin errores. Comparación v3 (126 variables) vs v4 (170 variables) sobre los mismos 706 casos.

| Algoritmo | AUC v3 | AUC v4 | Δ AUC | Δ AP | Δ Brier |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,65294 | 0,66539 | +0,01244 | +0,00940 | −0,00098 |
| Extra Trees | 0,61318 | 0,62277 | +0,00959 | +0,00624 | −0,00080 |
| HistGradientBoosting | 0,55166 | 0,54636 | −0,00531 | +0,01162 | −0,00643 |
| XGBoost | 0,59253 | 0,61165 | +0,01913 | +0,03070 | −0,00416 |
| CatBoost | 0,60624 | 0,59874 | −0,00750 | −0,00546 | −0,00128 |
| SVM-RBF | 0,56160 | 0,59297 | +0,03138 | +0,01457 | −0,00126 |

- Comparación primaria vs procedimiento CatBoost del release reconstruido: Random Forest v4 AUC 0,65700 frente a 0,62744 (Δ +0,02956). IC95 exploratorio agrupado de la diferencia [−0,00066; +0,06546]; incluye cero, no acredita superioridad.
- Top-1 entre tres zonas históricas OOF: RF v4 532/1246 (42,70 %); XGBoost v4 551/1246 (44,22 %); protocolo release reconstruido 525/1246 (42,13 %). NO son tres zonas realmente elegidas por M1.
- AUC Random Forest en la franja 20–30°S: 0,4669 v3 a 0,5119 v4; continúa cerca del azar. Brecha TRAIN−TEST aparente AUC Random Forest v4 promedio 0,321; sobreajuste no resuelto.
- Diagnósticos por región, fold, matrices y Brier de clase: comparison_706/scientific_audit_matrices_and_regions.csv y scientific_audit_by_fold.csv.
- Prueba de ingeniería combinando M1 SVM ampliado de noviembre + release M2 CatBoost de septiembre: 10 casos reales ya usados en demostración; ambos eligieron A1015-22 en el Top-3 generado por M1. Dos casos posiblemente duplicados frente al M2 ampliado y ocho fuera de dominio M2. NO constituye evaluación externa. Evidencia: integration_m1nov_m2sep/.
- E2: nueva competición Optuna anidada de **las seis familias**, iniciada, checkpoints aislados por algoritmo/fold. No activación automática.
