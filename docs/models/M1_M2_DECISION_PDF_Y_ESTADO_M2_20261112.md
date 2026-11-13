# LithiumScope: decisión integrada en PDF y estado científico del Modelo 2

**Fecha:** 12/11/2026. **Implementado:** selección exclusivamente matemática en el renderizador del PDF, sin nuevos entrenamientos ni activación de releases. **Estado M2:** decisión sobre el modelo futuro todavía pendiente.

## 1. Arquitectura y objetivo

Los dos modelos tienen funciones independientes. El **Modelo 1** estima concentración de litio (ppm) utilizando variables geoquímicas. El **Modelo 2** estima una **prioridad espectral relativa** a partir de seis bandas Sentinel-2; no recibe como features los resultados del M1 ni concentraciones químicas directas. **Es el renderizador** quien cruza resultados ya calculados mediante `case_id` para decidir dónde revisar primero. El objetivo es priorización exploratoria, **no** viabilidad económica de extracción ni probabilidad calibrada de yacimiento.

El documento tiene una página de decisión nueva entre la primera y la antigua segunda página. Ninguno de los apartados existentes se elimina. Toda la selección se **recalcula al generar o reconstruir el PDF** utilizando los CSV de predicciones de esa sesión, y no valores escritos manualmente.

## 2. Fórmulas fijas del informe (no son nuevos modelos)

**Fase 10 → 3 (Modelo 1):**
1. Tomar casos con identificador único, `Li_icpms_predicted` finito y ≥0, y aplicabilidad M1 = `OK`.
2. Si todos esos casos contienen límite inferior empírico `Li_icpms_interval_low_q90` válido, priorizar por **`S1 = max(0, límite_inferior_q90)`**. El límite es estimación conservadora basada en el residual absoluto histórico q90, **no un intervalo de cobertura externa garantizada**. Si falta algún límite, ordenar **a todos** por `Li_icpms_predicted` para evitar tratamientos heterogéneos.
3. Elegir tres; empates de `S1` se resuelven por `Li_icpms_predicted` y después `case_id`. Con menos de tres casos válidos no se declara selección.

**Fase 3 → 1 (Modelo 2):**
1. Usar solamente los scores espectrales ya predichos sobre los tres casos M1. No ejecutar nuevamente el Modelo 2.
2. Dejar fuera de la competencia los scores no finitos, fuera de [0, 1] o con aplicabilidad M2 diferente de `OK`.
3. Priorizar el mayor score M2, sin convertirlo en porcentaje de probabilidad geológica. Con menos de dos scores utilizables o empate del máximo (tolerancia 1e-9), **abstenerse**; cuando algún tercero está fuera de dominio, declarar como máximo **prioridad provisional**. Tres aplicables permiten `COMPARABLE`, **no validación externa**.

**Verdad de terreno:** `Li_icpms` observado jamás influye en la selección, ni en M1 ni en M2. Si está disponible, se usa **después**, exclusivamente para medir retrospectivamente si el primero de M1 o la elección final acertaron el mayor Li observado **entre los tres**. Esto no significa que el sistema apruebe perforar. No se introduce un score combinado artificial, pesos de conveniencia ni thresholds ajustados sobre los diez casos.

## 3. Ejecución real auditada y PDF regenerado

Ejecución: `results/predictions/demonstration_20261112_164710_m0300/`, diez casos. Utiliza los candidatos de demostración **M1 SVM `training_20261106_111653_m0300`** y **M2 Random Forest `training_20261112_123451_m0300`**; **solo la demostración** los carga. Las predicciones ordinarias continúan utilizando el release activo anterior.

| Preselección por M1 | Li predicho M1 (ppm) | Cota conservadora M1 (ppm) | Score M2 | Aplicabilidad M2 |
|---|---:|---:|---:|---|
| A1015-22 | 22,6476 | 5,967 | 0,409610 | OK |
| PAL14-2 | 21,7087 | 5,028 | 0,185209 | OUT_OF_DOMAIN |
| A1015-21 | 18,8552 | 2,175 | 0,209837 | OK |

**Prioridad espectral provisional: A1015-22.** Es también el primer puesto del M1 y tiene el mayor Li observado entre los diez de la demostración (28,191 ppm); **no se ha demostrado un beneficio incremental de M2 en este único ejemplo**. Un competidor del top-3 tiene M2 fuera de dominio. Siete de los diez casos originales estaban fuera de dominio de M2, y cero superaron el umbral operativo binario de M2; eso NO impide mostrar orden relativo, pero limita su fiabilidad.

**Independencia:** auditoría posterior detectó posibles duplicados contra GEOROC: A1015-22 (~1,907 m) y A1015-25 (~3,789 m); A1015-21 está a ~175 m. Esta demostración **no es validación externa independiente** ni fundamento para declarar la superioridad de un nuevo release.

El PDF se regeneró a partir de los archivos de la sesión, preservando una copia exacta del anterior `report_before_decision_page.pdf`. El archivo nuevo es `report.pdf` (**25 páginas frente a 24 anteriores**). Página 2: tabla de diez casos, tres miniaturas RGB de imágenes reales, ganador señalado, reglas, aplicabilidad y contraste retrospectivo. La antigua página 2 se conserva como página 3; el resto del material se desplaza una página. Las miniaturas RGB son representación humana; M2 utiliza las seis bandas originales.

**Archivos fuente:** `src/lithiumscope/prediction/reporting/decision.py` y `reporting/sections/decision.py`, integración en `reporting/sections/spatial.py`. Se mantiene el flujo original de generación de `report.pdf` y `prediction/rebuild_pdf.py`. La reconstrucción del PDF usa únicamente predicciones archivadas, no reentrena.

## 4. Registro verificable de modelos experimentales M2 (SIN decisión de release)

| Modelo / perfil | Situación | Datos / finalidad |
|---|---|---|
| `training_20260920_195933_m0300` | **ACTIVO PUBLICADO**; intacto | Referencia del release `lithiumscope-training_20260925_111245_m0300` |
| `training_20261105_223618_m0300` | **PENDING**, preservado | Entrenamiento anterior, sin promoción |
| `training_20261106_135029_m0300` | **PENDING**, preservado | Entrenamiento anterior, sin promoción |
| `training_20261111_170832_m0300` | **PENDING**, preservado | Competencia anterior; CatBoost elegido |
| `training_20261112_112450_m0300` | Experimento reducido archivado | Reconstrucción parcial de protocolo anterior; no full competition |
| `training_20261112_123451_m0300` | **PENDING**, candidato experimental principal | Competencia de seis algoritmos; ganador Random Forest, 783 casos OOF, 767 elegibles para entrenamiento final |
| `regularized_v2` | Perfil RF experimental, no release | CV espacial de 5 folds y búsqueda interna Optuna |
| `control_regularized_126` | Brazo experimental, **no modelo persistido/promovido** | 250 árboles, profundidad 8, hoja 10; control para técnicas de overfitting |
| `train_only_kbest_48` | Brazo de selección de variables, no release | Selección train-only de 48 descriptores |
| `gain_augment_train_126` | Brazo sintético probado, no release | Ganancia DN correlacionada; **no adoptada por defecto** |

El registro estructurado preexistente permanece en `docs/models/M2_EXPERIMENTAL_CANDIDATE_REGISTRY_2026.json`. Los archivos binarios y el registro `models/active_models.json` no se modifican.

## 5. Datos, correcciones y resultados ya realizados

- **Base M2:** `data/processed/model_2/vnext_expanded/training_manifest.csv` y `spectral_features.csv`, **706 registros históricos + 77 GEOROC provisionales = 783**; 126 features de seis bandas, 143 grupos espaciales. Las 77 muestras son provisionales y no se han certificado como nuevas observaciones geológicas independientes de campo.
- **QA95:** 767 muestras elegibles para entrenamiento final y 16 muestras QA insuficiente retenidas en evaluación OOF; la calidad no se aplicó para ocultar errores en TEST.
- **Spatial CV:** cinco folds congelados, selección de umbral desde TRAIN, Optuna/calibración en subfolds y auditoría de proximidad. El embargo geográfico verdadero de 5 km se probó **sin modificar el test**, y no produjo mejoría AUC concluyente.
- **Comparación publicada vs RF experimental:** 706 casos históricos emparejados. Release archivado AUC **0,620711**, AP **0,382668**, Brier **0,184115**; RF control regularizado AUC **0,658721**, AP **0,393443**, Brier **0,219656**. Mejora descriptiva de ranking, **peor calibración**; los IC95 pareados para AUC/AP incluyen cero. Con calibración entrenada solo en train: AUC 0,647261, AP 0,366600, Brier 0,180070; no gana en todas las métricas.
- **Comparación entre 3 sitios (NO selección previa por M1):** RF más reciente acierta 568/1244 ternas exploratorias (45,659 %) vs CatBoost anterior 547/1244 (43,971 %). **No corresponde interpretarlo como precisión de la cadena integrada 10→3→1**. La nueva cadena aún no cuenta con una cohorte independiente de M1+M2 emparejada para estimarla.
- **Aumentación sintética (evaluada):** el control regularizado 126 logró AUC 0,65747 y AP 0,38781 sobre 783; aplicar pequeña ganancia DN solo a TRAIN obtuvo AUC 0,64546 y AP 0,37411, con mayor brecha de sobreajuste. El Brier bajó, pero **no se justifica adoptarla como solución de generalización**. Copiar datos no crea nuevas regiones ni etiquetas verificadas.
- **Radiometría:** se investigó escala/offset Sentinel-2 por escena, máscaras y SCL. Se extrajeron variantes de 706 casos con reflectancia y se compararon; ninguna mejoró de forma consistente AUC, AP y Brier frente al release.
- **Fuentes externas geoquímicas:** auditoría de La Negra, salares y Zenodo documentada; **0 etiquetas independientes M2 aprobadas**. Datos químicos externos no se incorporaron como features al M2.
- **Limitación transversal:** evaluación repetida sobre folds de desarrollo; no se ha demostrado superioridad estadística o generalización en sitios confirmatorios nuevos.

Evidencia detallada ya existente:
`docs/models/M2_CIERRE_EXPERIMENTAL_20261112_RF.md`,
`M2_PROTOCOLO_CANONICO_20261112.md`,
`M2_REGULARIZACION_ESPECTRAL_20261112.md`,
`M2_CIERRE_COMPARATIVO_RELEASE_20261112.md`,
`M2_CAMPANA_DATOS_EXTERNOS_20261112.md`.
Experimentos de salida en `results/scientific_simulations/m2_canonical_rf_20261112/`,
`m2_spectral_overfit_20261112/`, `m2_release_closeout_20261112/` y
`sentinel_methodology_20261112/`.

## 6. Decisión actual y condición de cierre de M2

**El PDF de decisión está implementado y no depende de haber elegido el nuevo M2.**
El usuario aún **no ha aprobado un reemplazo**. Mantener el M2 del release activo para predicciones ordinarias; la demostración puede seguir usando el RF pendiente explícitamente configurado, identificado en el manifiesto.

La decisión científica de M2 queda **PENDIENTE**, no omitida. Para adoptarlo responsablemente se necesita una cohorte independiente en la que existan predicciones M1 y M2 para los **mismos sitios** y resultados de referencia compatibles; entonces calcular `Top-1@3` condicionado por M1 y compararlo con M1 solo, M2 solo y release M2 anterior. Predefinir el criterio y el holdout **antes** de elegir, sin optimizar indefinidamente los cinco folds históricos.

Hasta ese momento no puede afirmarse que la nueva técnica resuelva el overfitting ni que M2 mejore la inversión económica frente a M1. Sí puede afirmarse que el sistema genera una **priorización reproducible**, distingue evidencia fuera de dominio y se abstiene cuando no hay datos suficientes.

**Control de cambios:** solo código de cálculo/presentación del reporte, pruebas y documentación. Sin push, sin activar modelos ni alterar menú, datos originales, etiquetas o binarios.


## 7. Verificación final del renderizador

- Suite global ejecutada: **381 tests aprobados**, 15 advertencias no bloqueantes, 0 fallos.
- Pruebas dirigidas tras conservar el orden original de figuras y mapas: **21 aprobadas**, 0 fallos.
- Ruff: sin errores. El test opcional de inspección PDF usa `pypdf` cuando está disponible.
- El PDF reconstruido de la demostración tiene **25 páginas**, y la versión original respaldada tiene **24**. Se comprobó que los textos de **las 23 páginas anteriores después de la portada coinciden 23/23** con las nuevas páginas 3 a 25, ignorando exclusivamente el número de página en el pie. El texto de la portada también coincide. La segunda página nueva incorpora **tres imágenes originales**.
- La reconstrucción respeta el orden de `maps` y `figures` archivado en `prediction_manifest.json`, aun cuando las rutas absolutas originales correspondan a otro clon. La nueva decisión proviene de las predicciones guardadas y no altera ninguna predicción.
