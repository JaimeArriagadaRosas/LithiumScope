# Modelo 2 — Diagnóstico de la competencia con 706 muestras y contraste con el release

**Fecha:** 13 de noviembre de 2026. **Estado:** entrenamiento completo de los seis algoritmos; revisión de resultados y protocolo; **release no reemplazado**.
**Entrenamiento auditado:** `training_20261113_133905_m0300`, log `logs/training/session_20261113T163854_687078Z`.
**Referencia primaria:** protocolo del **release CatBoost `training_20260920_195933_m0300`**, reconstruido sobre los mismos 706 ejemplos históricos y folds.
**Comparación secundaria para diagnosticar la intervención:** RF y otras cinco familias de la competencia ampliada `training_20261113_114607_m0300` (783 fuentes) **restringidas a las primeras 706 observaciones históricas en TEST**.
**Nota:** no se evaluó el CatBoost publicado de septiembre *in sample* sobre datos usados en su ajuste. Se usa la reconstrucción del **procedimiento anterior**, no una supuesta evaluación externa del binario desplegado.

## 1. Resumen ejecutivo

**Conclusión científica provisional: NO se ha demostrado que el candidato de 706 supere al release anterior, aunque sí tiene mejoras parciales en la comparación.** La intervención de retirar 77 GEOROC provisionales mejoró **Brier global** y redujo falsos positivos del RF sobre los 706 históricos, **pero aumentó los falsos negativos** y no produjo una ganancia estadísticamente clara de AUC. La investigación de etiquetas/representatividad de esas 77 **sigue abierta**: mejorar al retirarlas no prueba que estuvieran mal etiquetadas.

La competición seleccionó **Random Forest** con ROC-AUC promedio de folds **0,616210**, Average Precision promedio **0,370656**, con 66 grupos, cinco folds espaciales y tres inner folds. `training.expansion.enabled=false`; 706 registros originales (sin 77 provisionales); 690 elegibles **solo para TRAIN** por QA95, pero TODOS los 706 se mantienen en TEST. Se mantuvieron las búsquedas regularizadas de RF/Extra Trees y `calibration_policy: legacy`.

Los cuatro reportes automáticos quedaron `complete`; `candidate_status=pending`, `experimental_only=true`, `release_gate.pass=false`. El modelo activo continúa siendo `models/model_2/trained/training_20260920_195933_m0300/model.joblib`. El SHA-256 del `model.joblib` candidato coincide con su metadata, y el artefacto carga correctamente.

## 2. Comparación decisiva: protocolo publicado reconstruido frente a RF 706

**Definición única de las etiquetas de esta tabla:** criterio fijo histórico `Li >= 21.6 ppm` para **los mismos 706 sitios en sus mismos folds**, **178 positivos y 528 negativos**. Esto vuelve pareadas las métricas. Son datos de desarrollo reutilizados y no una validación externa nueva.

| Métrica | Protocolo CatBoost anterior reconstruido | Candidato RF 706 | Diferencia |
|---|---:|---:|---:|
| ROC-AUC (mayor mejor) | 0,627442 | **0,643572** | **+0,016130** |
| Average Precision (mayor mejor) | 0,370013 | **0,373831** | **+0,003818** |
| Brier (menor mejor) | **0,185858** | 0,186435 | **+0,000577** (empeora) |
| Top-1 entre tres zonas distintas por caso (mayor mejor) | 42,13 % (525/1.246) | **42,86 % (534/1.246)** | **+9 casos; +0,72 p. p.** |

Intervalos de confianza exploratorios por bootstrap de grupos espaciales sobre las diferencias pareadas:
- ΔAUC: **[−0,0142; +0,0483]** (incluye cero).
- ΔAP: **[−0,0409; +0,0521]** (incluye cero).
- ΔBrier: **[−0,0112; +0,0109]** (incluye cero).

**Interpretación:** no hay superioridad estadísticamente respaldada del nuevo RF frente al procedimiento CatBoost original. La diferencia +0,0161 de AUC es además menor que el umbral interno de desarrollo **+0,020**, aunque dicho umbral actual se implementa respecto al baseline de prevalencia, no al release. La ganancia AP +0,0038 es muy inferior a +0,020. Tampoco puede certificarse mejor calibración frente al release (Brier ligeramente peor).

**Diferencia metodologica historica:** el protocolo original fijaba el percentil 75 de Li con el conjunto completo antes de separar los folds, con riesgo de fuga de informacion en la construccion de la etiqueta. El metodo nuevo determina q75 exclusivamente desde TRAIN de cada fold. El contraste pareado aplica el criterio fijo de 21,6 ppm a ambos scores, pero no vuelve identicos los entrenamientos ni permite atribuir el cambio solamente al algoritmo.

**El gate de consola NO compara contra el release.** Los mensajes `ΔROC-AUC +0.1162` y `ΔAP +0.1206` son diferencias frente al **baseline de prevalencia por fold** (`ROC-AUC=0.5`; `AP≈0.2500`). El mismo registro establece `CANDIDATO / NO VERIFICADO` y `reason=no_independent_external_holdout`. No utilizar esas cifras para afirmar victoria sobre CatBoost publicado.

## 3. Auditoría de las seis matrices de confusión (umbral común de 0,5)

**Definición:** resultados OOF de la nueva competición; **177 positivos y 529 negativos** según **q75 calculado en TRAIN de cada fold**. No mezclar con los 178 positivos de la tabla pareada con umbral Li fijo. Un cambio de cinco decisiones de etiqueta en la frontera explica la discrepancia (hay ejemplos de 21,5 a 21,9 ppm que caen a uno u otro lado del q75 de su fold).

| Algoritmo | VP | VN | FP | FN | AUC OOF | AP OOF | Brier OOF |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **36** | 479 | 50 | 141 | **0,652943** | **0,377440** | **0,184552** |
| CatBoost | 32 | 491 | 38 | 145 | 0,606239 | 0,350536 | 0,208097 |
| Extra Trees | 32 | 486 | 43 | 145 | 0,613181 | 0,342129 | 0,188474 |
| XGBoost | 31 | 476 | 53 | 146 | 0,592526 | 0,331872 | 0,209843 |
| HGB | 38 | 468 | 61 | 139 | 0,551664 | 0,333643 | 0,239296 |
| SVM-RBF | **0** | 528 | 1 | **177** | 0,561597 | 0,317102 | 0,185636 |

**RF a 0,5:** precisión 41,86 %, sensibilidad **20,34 %**, especificidad 90,55 %, F1 0,274. El modelo deja sin detectar 141 de 177 positivos. SVM no detecta ningún positivo; no atribuirlo a ausencia absoluta de ranking (AUC SVM >0,5), puesto que el umbral y la distribución de scores también influyen.

### Umbral operativo elegido con el mismo pool OOF

El entrenamiento guardó **umbral 0,265051857192025** mediante optimización de Balanced Accuracy **sobre el mismo conjunto OOF**. Reproducción numérica de su matriz: **VP=138, VN=247, FP=282, FN=39**, precisión **32,86 %**, recall **77,97 %**, especificidad **46,69 %**, Balanced Accuracy **62,33 %**.

Esto cambia dramáticamente el comportamiento de selección:
- Umbral 0,5: **50 falsos positivos y 141 falsos negativos**.
- Umbral 0,2651: **282 falsos positivos y 39 falsos negativos**.

**Punto metodológico pendiente:** el umbral fue seleccionado con etiquetas OOF reutilizadas durante desarrollo, no en inner folds estrictamente separados del TEST final para evaluar todo el sistema. Es una estimación **optimista** y susceptible a sobreajuste de umbral, no una sensibilidad validada externamente. Hay **dos scores serializados extremadamente próximos al umbral**, por lo que documentar precisión/empates al guardar CSV.

La salida de M2 es un **score relativo de prospectividad**, no la probabilidad calibrada de encontrar un depósito rentable.

## 4. Hipótesis sobre los 77 GEOROC: ablación sobre exactamente los mismos 706 TEST

Las seis nuevas OOF están alineadas por orden, Li, etiqueta de fold y grupo con las **primeras 706 filas** de las OOF del run de **783**. Esto permite investigar el efecto de entrenar con y sin 77 muestras nuevas, sin cambiar el universo evaluado. La búsqueda vuelve a ajustarse en cada tratamiento, por lo que cambian modelos/hiperparámetros como consecuencia de los datos.

| Indicador RF sobre los mismos 706 OOF (q75 fold) | Entrenado con 783 fuentes | Entrenado con 706 | Cambio |
|---|---:|---:|---:|
| AUC OOF | 0,640960 | **0,652943** | +0,011983 |
| AP OOF | 0,371977 | **0,377440** | +0,005462 |
| Brier global (↓) | 0,200356 | **0,184552** | **−0,015805** |
| Brier en positivos (↓) | **0,336416** | 0,410324 | **+0,073908** (peora) |
| Brier en negativos (↓) | 0,154831 | **0,109009** | **−0,045822** |
| VP a 0,5 | **69** | 36 | −33 |
| VN a 0,5 | 424 | **479** | +55 |
| FP a 0,5 | 105 | **50** | −55 |
| FN a 0,5 | **108** | 141 | +33 |

CI95 del ΔAUC **[−0,0291; +0,0430]** y del ΔAP **[−0,0496; +0,0416]**: inconcluyentes. El ΔBrier tiene CI95 **[−0,0317; −0,0041]**, compatible con una mejora de calibración agregada en este remuestreo espacial. **Esa mejora se obtiene sobre todo en la clase negativa y empeora el Brier de positivos.** Por lo tanto, reducir la ampliación NO mejoró uniformemente la generalización ni la recuperación de Li alto.

**Hipótesis H77-A:** los 77 GEOROC podrían contener emparejamientos geoquímico/espectrales incorrectos. **NO demostrado.**
**Hipótesis H77-B:** los 77 son válidos, pero modifican la distribución regional y los mejores hiperparámetros. **Plausible, no aislado.**
**Hipótesis H77-C:** con menos positivos/registros diversos el RF se vuelve más conservador al umbral 0,5. **Compatible con las matrices**, no mecanismo causal demostrado.
**Hipótesis H77-D:** la calibración relativa empeora por incluir los 77, incluso si aportan recall. **Compatible con Brier agregado**, requiere análisis de pesos/regiones y calibración train-only.
**Regla de inferencia:** esta ablación documenta consecuencias de retirar 77 filas, **no identifica automáticamente error de etiqueta**.

La ROC-AUC media de los cinco folds **0,6162** en el run 706 es inferior a la media **0,6336** en el run 783, pero no hay que confundir esa comparación no pareada global con la AUC OOF sobre los mismos 706. Son agregaciones y muestras distintas.

## 5. Diagnóstico de datos, procedencia y región

En los **706** históricos se conservan **66 grupos**, 64 escenas STAC, 690 casos QA95 aptos solo para TRAIN (16 excluidos de TRAIN, conservados en TEST), **566 proxies SCL aceptados**, **140 abstenciones SCL**, **nueve OOD >25 %** y **25 sitios con etiquetas históricas mixtas** (77 filas). No hay certificación geológica de campo para nuevas etiquetas; la auditoría está basada en proxies.

**RF candidato (OOF):**
- Latitudes entre **10°S y 20°S**: 621 observaciones, 132 positivas, **AUC 0,6400**, Brier 0,1699.
- Latitudes entre **20°S y 30°S**: 85 observaciones, 45 positivas, **AUC 0,4669**, Brier 0,2917. **Problema importante de ordenación espacial**, aunque en una región con pocos ejemplos y alta proporción de positivos.
- Sitios de etiquetas mixtas: 77 observaciones, AUC 0,5996; requiere validar si mediciones distintas del mismo sitio reflejan geología real o fallos de procedencia; **no recodificar automáticamente**.
- OOD >25 %: nueve observaciones y AUC 0,3056; tamaño demasiado pequeño para conclusión estadística robusta.
- QA95 descartadas del ajuste pero preservadas en TEST: 16; no excluir de las métricas globales ni usar su performance de subgrupo como garantía de seguridad.
- SCL aceptado: 566, AUC 0,6528; abstención: 140, AUC 0,6788. No afirmar que todas las observaciones SCL-abstención son malas ni que el proxy sea verdad de terreno.

Estos hallazgos no son equivalentes a diagnosticar sobreajuste del estimador: para eso hacen falta **errores TRAIN vs TEST por fold**, **curvas de aprendizaje** y contrastes con distancias entre entrenamiento y validación. La variación regional sí demuestra una generalización **inestable**.

## 6. Ranking de tres sitios: relevancia y limitaciones

Con los mismos **1.246 tríos históricos** compartidos entre brazos del run 706:
- **RF: 534/1.246 = 42,86 %** (entre los 3, elige el de mayor Li mediana).
- **Procedimiento anterior reconstruido: 525/1.246 = 42,13 %**.
- CatBoost nuevo: 527/1.246 = 42,30 %.
- Azar uniforme: aproximadamente 33,33 %.

El beneficio de RF frente al protocolo anterior equivale a **9 decisiones adicionales**, **0,72 puntos porcentuales**, sin CI específico que pruebe una mejora real. Estos tríos fueron generados dentro de la misma validación histórica y **NO representan una prueba completa M1→Top-3→M2→Top-1** ni evaluación independiente.

## 7. Riesgos de interpretación / anomalías del software

1. El mensaje `release_gate ΔAUC +0.1162 / ΔAP +0.1206` es vs **baseline de prevalencia**, NO contra el release.
2. Diferenciar métricas **media de folds** (`AUC=0.6162`, criterio de ganador) de métricas OOF agrupadas (`AUC=0.65294`, 177 positivos) y del protocolo pareado de release (`AUC=0.64357`, 178 positivos). No compararlas sin unificar etiquetas y agregaciones.
3. Algunos metadatos genéricos del auditor aluden a `old_and_new_training_sets_differ=true` incluso en escenario solo 706; el resumen específico `old_release_protocol_comparison_summary.json` confirma **ambos proceden de 706 históricos**, pero persisten **políticas QA95, regularización y Optuna diferentes**. Reconciliar esa redacción en futuras correcciones.
4. El umbral operativo se eligió tras ver OOF; replicación `>=` sobre scores serializados cercanos al umbral merece una prueba de empates; no utilizar las cifras operativas como evaluación confirmatoria.
5. El protocolo viejo fue reconstruido y reentrenado sobre folds. La precisión del artefacto CatBoost publicado de septiembre **no fue evaluada directamente fuera de muestra** en este run. Toda inferencia de superioridad debe quedar calificada.
6. La versión de **706** aplica QA95 train-only: solo 690 son elegibles para ajuste, de modo que **NO** repite exactamente el procedimiento original. Esa es una decisión deliberada y correcta para este contraste de metodología, pero separa la hipótesis «solo cambiar datos» de la hipótesis «restaurar protocolo viejo».

## 8. Decisión de release

**Recomendación: NO activar el nuevo Random Forest como release del M2 todavía.** Razones:
- ΔAUC **+0,0161** y ΔAP **+0,0038** frente al protocolo viejo; CI de ambos incluyen cero.
- Brier apenas peor que el CatBoost de referencia a etiquetas unificadas, y en el grupo positivo hay signos de calibración deficiente.
- Gran compromiso entre falsos negativos y positivos al fijar un umbral; el umbral operativo está optimizado sobre OOF reutilizadas.
- AUC regional **0,4669** en la franja 20°S–30°S y 25 sitios con etiquetas mixtas sin validación definitiva.
- El gate técnico explícitamente **no aprueba** el candidato por falta de holdout externo independiente.

**Siguiente iteración de mayor valor científico (sin otra competencia completa inmediata):**
1. **Auditar muestras y protocolos**: etiquetas conflictivas, resolución geográfica, QA/SCL/OOD, fuentes de Li y correspondencia roca/satélite; conservar originales y emitir diagnóstico de causa, no modificar etiquetas por sospecha.
2. **Analizar sensibilidad umbral vs costes FP/FN**: elegir función objetivo y umbral usando SOLO inner TRAIN, con nested threshold y calibración independiente; declarar escenarios de uso (descubrimiento exploratorio vs clasificación estricta).
3. **Medir brechas y robustez espaciales**: TRAIN/TEST por fold, curvas de aprendizaje, distancia mínima entre grupos/celdas, validaciones con bloque/buffer, e incertidumbre por región.
4. **Ablaciones dirigidas**: calzar 706/783 con mismos folds, QA, hiperparámetros y regiones; probar regularización, ponderación de clase, calibración y fuente GEOROC por separado. No atribuir cambios a etiquetas sin un experimento controlado.
5. **Evaluación aplicada**: M1→Top-3→M2→Top-1 sobre conjuntos nunca usados para seleccionar ambos modelos, y comparar frente a la estrategia vigente. Ante falta de validación independiente, solo release **provisional** con límites y aprobación explícita del responsable.

**Estado del repositorio en esta auditoría:** no se ejecutó entrenamiento, no se hizo push, no se reemplazó el release y no se cambió menú. Únicamente se auditaron artefactos y se guardó documentación.

## 9. Archivos fuente verificables

- `results/model_2/runs/training_20261113_133905_m0300/tables/`: `competition_ranking.csv`, `confusion_matrices_all_algorithms.csv`, `historical_release_comparison_six_algorithms.csv`, `old_release_protocol_intervals_706.csv`, `paired_competition_deltas.csv`, `three_zone_competition_summary.json`, `scientific_stratified_metrics.csv`, `diagnostic_manifest.json`, `winner_operating_threshold.csv`, `evidence_report_status.json`.
- `results/model_2/runs/training_20261113_114607_m0300/tables/`: predicciones anteriores 783 para la ablación comparada sobre los mismos 706 TEST.
- `models/model_2/trained/training_20261113_133905_m0300/metadata.json`, `candidate_status.json`.
- Código de auditoría: `src/lithiumscope/model_2/training/diagnostic_inputs.py`, `diagnostic_audit.py`, `diagnostic_release_comparison.py`.
