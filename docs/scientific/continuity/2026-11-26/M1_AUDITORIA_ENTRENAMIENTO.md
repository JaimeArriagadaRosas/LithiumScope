# M1 — auditoría de entrenamiento y primera optimización de metodología

Fecha: 26/11/2026. Plan aprobado: [M1_PLAN_OPTIMIZACION_INTEGRAL.md](M1_PLAN_OPTIMIZACION_INTEGRAL.md).
Rama feat/m1-targeted-data-refinement. Alcance exclusivo M1. Fase 2A:
auditoría del código y documentación registrados; verificación de artefactos
históricos locales adicionales todavía pendiente cuando resulte necesaria.
Fase 2B: primera campaña de pérdidas nativas implementada, no ejecutada.
No se ha iniciado postentrenamiento, integración, candidato de referencia
ni fine-tuning especializado.

## 1. Evidencia revisada y alcance

Se consultó el árbol recursivo de la rama (no truncado), configuración,
factoría, seis módulos de algoritmos, Optuna, competencia, ejecución de
candidatos, preprocesamiento, tratamiento canónico, variantes experimentales,
ajuste focalizado, expertos y estudio de incertidumbre. Se contrastaron
RESULTADOS.md, M1_EXPANSION.md, M1_AUGMENTATION.md,
M1_CIERRE_DATOS_RESULTADOS.md y docs/models/M1_REFINAMIENTO_FOCALIZADO.md,
M1_REFINAMIENTO_20261125.md, INTEGRACION_TRATAMIENTOS_ENTRENAMIENTO_20261125.md,
GENERALIZACION_DATOS_20261125.md y
M1_CUATRO_FASES_DIAGNOSTICO_RELEASE_VS_CANDIDATO_20261113.md.
No se leyeron nuevamente todos los checkpoints/estudios SQLite ni todas las
tablas locales: la documentación histórica distingue campañas ejecutadas
de posibilidades del código. No afirmar ausencia absoluta de una técnica
por no encontrarla en los archivos examinados. No repetir fits para auditar.

Estados empleados: disponible = biblioteca/factoría lo admite;
implementada = existe ruta de código; evaluada = evidencia de campaña;
rechazada = no alcanzó guardas del protocolo estudiado, no invalidez universal;
pendiente = no se identificó prueba aislada suficiente; no aplicable =
incompatibilidad técnica o estadística justificada.

## 2. Registro de seis algoritmos y espacio de búsqueda canónico

Fuente: config/model_1.yaml y src/lithiumscope/model_1/training/factory.py,
training/algorithms/<familia>.py. Todos cuentan con competencia espacial
histórica; no todos con el mismo refinamiento académico enriquecido.

| Familia | Parámetros por defecto relevantes | Espacio Optuna canónico | Pérdida/criterio | Trials configurados por fold |
|---|---|---|---|---:|
| random_forest | 350 árboles | n_estimators 150–700; max_depth 4–24; min_samples_split 2–12; min_samples_leaf 1–6; max_features sqrt/log2/1.0 | squared_error por defecto | 20 |
| xgboost | 400 árboles, lr .05, profundidad 6, subsample/colsample .9 | n_estimators 150–800; learning_rate .01–.2 log; max_depth 3–12; subsample .6–1; colsample_bytree .6–1 | reg:squarederror explícita | 20 |
| svm | kernel rbf, C 10, epsilon .1, gamma scale | C .1–100 log; epsilon .01–1 log; gamma scale/auto | epsilon-insensitive de SVR | 20 |
| tabnet | max_epochs 150, patience 20, validación .15 | max_epochs 80–250; patience 10–35 | loss por defecto de biblioteca; RMSE de validación | 8 |
| hist_gradient_boosting | 350 iteraciones, lr .06, hojas 31, L2 .1, early_stopping=False | lr .01–.2 log; max_iter 150–650; max_leaf_nodes 15–63; min_samples_leaf 10–40; l2_regularization 1e-4–5 log | squared_error por defecto | 20 |
| catboost | 500 iteraciones, depth 7, lr .05 | iterations 250–900; depth 4–10; learning_rate .01–.15 log; l2_leaf_reg 1e-3–20 log | RMSE explícita | 12 |

Los cinco parámetros de RF/XGB/HGB, tres SVM, dos TabNet y cuatro CatBoost
son los específicos de familia. Se añade tratamiento categórico
treatment_policy en refinement_v1. No presentar número de dimensiones como
porcentaje de exhaustividad. Las dependencias opcionales xgboost, catboost,
optuna, torch, pytorch-tabnet se declaran en pyproject.toml; versiones
instaladas reales deben registrarse al ejecutar, no deducirse de mínimos.

Políticas de búsqueda (training/preprocessing/treatment_policy.py):
- RF: nonnegative, rare_group_sqrt.
- XGB: nonnegative, rare_group_sqrt, bounded_log, bounded_log_no_coordinates.
- SVM: nonnegative, bounded_log, bounded_log_no_coordinates.
- TabNet: nonnegative, robust_identity.
- HGB: nonnegative, rare_group_sqrt, bounded_log, rare_group_bounded_log.
- CatBoost: nonnegative, rare_group_nonnegative, rare_group_no_coordinates.
No son funciones objetivo nuevas: principalmente preparación, muestreo y
transformación del target. El límite de predicción es soporte TRAIN, no
máximo físico. No se modifica esta configuración productiva en fase 2.

## 3. Validación y ejecución de entrenamiento existente

training/optimization/optimizer.py usa TPE con semilla, MedianPruner
(4 trials iniciales y 2 folds de calentamiento en config), objetivo
minimizar promedio RMSE en folds internos, SQLite y contrato de estudio
acotado. CV 5x5 espacial, semilla42, grupos .5 grados, sin fallback
aleatorio en competencia canónica. Contratos verifican particiones, estados
y firmas; candidatos comparten folds. Ranking por RMSE medio, MAE y R².
La selección por RMSE medio no equivale a selección por RMSE global.
El estudio académico piloto RF utiliza 20 trials/fold y espacio distinto
(n_estimators 300–1500, profundidad 4–20, split 2–20, leaf 1–10,
max_features sqrt/log2/None, bootstrap true/false). No confundir con
los 100 trials por fold de OASIS ni los parámetros de competencia espacial.

training/preprocessing/spatial_tabnet.py realiza early stopping con grupos
espaciales separados dentro de TRAIN, preprocesador ajustado sin validación,
elige época y refit completo. HGB prohíbe early stopping aleatorio en
factoría; la comparación de max_iter fijo ya ofrece control de complejidad.
No activar parada aleatoria para aparentar generalización. TabNet cuenta
con regularización/épocas y escalado robusto evaluados; arquitectura,
pérdida y pretraining neuronal no aparecen como ejes de optuna_space.
SVM usa escalado; RF/CatBoost mediana en flujo común; XGB NaN nativo;
HGB/ CatBoost nativo NaN también probado en laboratorio académico.
Selección estable/imputación por litología, KNN, máscaras, ratios y
tratamiento de edad pertenecen al inventario anterior; no reabrir datos.

## 4. Campañas ejecutadas, con alcance y decisión

| Campaña | Evidencia y tamaño | Decisión |
|---|---|---|
| Competencias espaciales iniciales seis familias | 1120 y 3420 observaciones, documentación 13/11 | SVM ganó RMSE medio, no necesariamente global; no comparar poblaciones directamente |
| Evaluación 25/11 | run 20261125T111144_fe968d37, 1800 interiores y 60 exteriores exitosos | CatBoost 79.4087 RMSE medio; HGB 81.9934; RF 141.4558 global seleccionado; log1p TabNet inestable en un fold |
| Refinamiento dirigido seis familias | 26 políticas, 130 ajustes diagnósticos previstos por documento; otra bitácora consigna 260 en integración; no sumar sin verificar alcance | integración refinement_v1, tratamiento seleccionado internamente |
| Competencia referencia conservada | training_20261125_115932_m0300_826123c1, 3420 y 244 grupos | RF 152.302214 global/.131184 |
| Focalizada RF/HGB/CatBoost | 45 configuraciones, 1350 evaluaciones + 540 ajustes calibración, 164160 predicciones registradas | no supera simultáneamente referencias, no repetir |
| Balanceo y expertos | 21 configuraciones, 630 evaluaciones + 591 ajustes expertos | mezcla por régimen no recomendada; RF control conservado |
| Transferencia espacial A–K, HGB/CatBoost | M1_EXPANSION.md, sensibilidad por peso, selección interna y reserva ya explorada | CatBoost H 111.579554 global/.533679; no elegible |
| Piloto académico RF | 20261126T123613_24ef3f6c, 3263, 20 trials/fold | 7.723716/.561374 |
| Puente académico RF/CatBoost/HGB | 30 fits con perfiles fijos, histórico vs real adicional | HGB 7.291410 histórico; extensión uniforme empeora las tres familias |
| Refinamiento académico HGB | 110 ajustes, 20261126T161005_a8008a65 | 7.225062/.612628 selección interna; no repetir pesos |
| Ablación de predictores | 80 fits, 20261126T163711_3f9dd35a | canónico 7.239729/.611629; combinado 7.074911/.631231 |
| Cierre académico de familias/analogías | 35 árboles +20 estimadores locales, 20261126T165331_935bc05f | HGB combinado mejor RMSE; mezclas no mejoran |
| Síntesis | 115 nuevos fits, 20261126T180155_005d9a59 | todos peores al control |
| Cierre de datos | 15 nuevos fits, 20261126T213014_fc01460c | C2 7.053262/.633404, cola peor; conservar A |

Los recuentos se registran por campaña y no se suman a una cifra universal de
modelos: una calibración, un fit interno y un candidato guardado son objetos
distintos. Hashes y rutas completas de campañas de expansión en documentos
vinculados. Los resultados de las campañas del 25/11 se citan desde bitácoras,
no se certifican nuevamente sus artefactos locales en esta auditoría.

## 5. Cobertura por técnica y oportunidad

| Técnica | Estado y evidencia | Acción |
|---|---|---|
| Optuna y regularización estructural | evaluada en seis familias; espacios limitados descritos arriba | no repetir búsqueda global; reservar refinamiento local para fase 6 |
| RF criterion / ccp_alpha / max_samples | no incluidos en espacio consultado | investigar solo si RF competitivo; no forzar criterio absoluto para RMSE |
| XGB objetivos y reg_alpha/reg_lambda/min_child_weight/gamma | disponibles en biblioteca, no incluidos en búsqueda consultada | prioridad posterior: contrastar si resulta competitivo en académico enriquecido |
| SVM C/epsilon/gamma | evaluados; gamma solo scale/auto | no repetir búsqueda convencional; reevaluar familia solo si justificada |
| TabNet épocas/paciencia/refit espacial | implementado y evaluado | no duplicar; arquitectura/pretraining requieren hipótesis y más recursos, no por defecto |
| HGB pérdidas nativas | gap: no aparecen en optuna_space ni en perfiles académicos inspeccionados | primera campaña aislada |
| HGB early stopping aleatorio | explícitamente prohibido | no usar; investigar parada agrupada solo con evidencia de necesidad |
| CatBoost Huber/Lq/Expectile | disponibles; no figuran en optuna_space ni perfiles fijos inspeccionados | evaluar tras resultados de HGB; asimetría no garantiza mejora RMSE |
| Transformación identidad/log1p/sqrt/potencia/Yeo-Johnson | evaluada, inversa en ppm; casos inestables | no repetir |
| Ponderaciones, remuestreo, fuentes, síntesis | cerrados en fase 1 | no reabrir |
| Expertos de régimen | balanced/regimes.py: gate RF clasifica desde X, expertos regresión con mínimo 40 filas/5 grupos, mezcla por probabilidad | evaluado; no repetir idéntico |
| Mezcla convexa | expansion/blend_policy y academic_final | evaluada; no confundir con stacking aprendido |
| Corrección de sesgo aditiva | targeted/fit.py: GroupKFold(3) en TRAIN, media residual original ppm | evaluada junto con sqrt; falta aislar calibración sobre mejor control en fase 3 |
| Stacking entrenado / calibración afín / isotónica / residuos multivariados | no se identificó evaluación específica suficiente en módulos revisados | fase 3, no en esta campaña |
| Incertidumbre agrupada | uncertainty_study.py llama group_intervals con flag --execute; existencia de código no prueba ejecución | fase 3 verificar resultados locales; cobertura no equivale a RMSE |
| Continuación / warm_start | no identificado como tratamiento controlado en rutas revisadas | fase 6 condicional; no igualar más árboles con nueva información |

No es exhaustivo de toda la literatura. Antes de ampliar otra familia o
pérdida, comprobar evidencia local y escoger un presupuesto explícito.
Si el resultado de una técnica es negativo, conservar control. No exigir
probar toda combinación teórica: combinatoria y reutilización de TEST
impiden una estimación confirmatoria. Pendientes de fase 2 que no sean
justificados podrán cerrarse con motivación.

## 6. Documentación externa de referencia

- scikit-learn HistGradientBoostingRegressor 1.5:
  https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
  Admite squared_error, absolute_error, gamma, poisson y quantile. Gamma
  exige y>0 y log-link; Poisson y>=0 y log-link; cuantiles estiman un
  funcional condicional distinto de la media. No afirmar distribuciones
  físicas Gamma/Poisson para Li por usar su pérdida. La pérdida cuadrática
  apunta a media condicional, más alineada con RMSE; alternativas pueden
  mejorar aproximación finita o empeorar la cola. Ensayo empírico, no garantía.
- XGBoost 2.1:
  https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html
  Regresión cuadrática, Pseudo-Huber, cuantiles y regularización; versiones
  y disponibilidad deben verificarse antes de usarlas.
- CatBoost:
  https://catboost.ai/docs/en/concepts/loss-functions-regression
  RMSE, Huber, Lq y Expectile, entre otras. Revisar compatibilidad CPU/GPU
  y versión local antes de definir próxima prueba.
Las fuentes describen APIs y propiedades, no evidencia de mejora en M1.
No se implementa una pérdida personalizada no validada ni se altera la
métrica de evaluación para favorecer una pérdida determinada.

## 7. Primera implementación de fase 2B: HGB pérdidas nativas

Código: src/lithiumscope/experiments/model_1/integral/training_losses.py.
Nuevas pérdidas: absolute_error, gamma, poisson, quantile=.60; control
squared_error. Cuantil .60 es hipótesis predeclarada de asimetría moderada,
no tuned, no media condicional ni valor predictivo mejor garantizado.
Mantener HGB enriquecido A histórico, mismos perfiles nativos:
max_iter=500, max_leaf_nodes=31, min_samples_leaf=5, learning_rate=.05,
l2_regularization=.1, early_stopping=False, seed42, NaN nativo,
esquema seleccionado exclusivamente en TRAIN. Pesos unitarios.
No añadir filas, datos externos ni variables. Reutiliza recuperación
histórica retrospectiva asistida por Li (flag explícito, no certificada);
no apta para producción sin resolver identidad/disponibilidad inferencia.
Validar hashes/identidad del piloto, referencia de características y
recuperación; mismas 3263 filas, folds académicos aleatorios reutilizados.
No modificar las 3420 espaciales ni confundir evaluación descriptiva con
validación independiente. No reusar valores de TEST para selección.

Presupuesto: 5 pérdidas x 3 folds internos x 5 exteriores = 75 ajustes
internos; 4 alternativas x 5 exteriores = 20 ajustes externos. Total
95 nuevos, cinco controles externos archivados, seis políticas reportadas
(cinco pérdidas + seleccionada internamente). No Optuna, no cambios de
hiperparámetros más allá de la función de pérdida y su cuantil.
Selección dentro de TRAIN: RMSE medio menor al control interno, R² medio
>.499, MAE <= control + .1 ppm, RMSE global de cola >40 no superior al
control interno. Elegir menor RMSE, luego MAE y nombre; fallback control.
El cálculo de RMSE cola usa suma de SSE y n sobre los tres folds, no
promedio de RMSE de bandas pequeñas. Las alternativas fijas externas son
diagnóstico; la política seleccionada internamente evalúa el procedimiento.
No declarar vencedor general por consultar retrospectivamente todos los
TEST. Para proseguir, exigir beneficio consistente frente a A y evaluar
espacial/independiente en fase de integración, no promover.

Salida: predictions.csv (19578 OOF esperadas), fold_metrics.csv (30 filas),
inner_scores.csv (75), fit_trace.json (95), selection.json (5),
partitions.json (5), summary.csv (6), band_metrics.csv y manifest.json.
Hashes, identidades, parámetros, MAE, R², RMSE medio/global, error de cola,
sesgo, desviación por fold. No guardar estimadores ni aprobar muestras.
Nueve pruebas nuevas (94 esperadas en m1_expansion frente a 85 previas):
contrato de pérdidas, selección guardada/fallback, cinco smoke fits reales,
rechazo solapamiento y etiquetas no positivas. Pendientes en Windows.

CMD desde raíz:

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.integral.training_losses --allow-unverified-metadata
```

Salida results/generalization/model_1_integral_training_losses/<run>.
Enviar consola y ZIP completo de esa ejecución. Verificar outputs y
comparación por identidad antes de decidir otra técnica. No repetir datos,
preflight ni los 15 fits anteriores. Todavía no es el entrenamiento de
referencia de fase 5, ni el fine-tuning especializado de fase 6.
No se ha ejecutado la suite local ni el experimento en esta intervención.
M2, main, release y producción intactos.


## 8. Resultado validado de pérdidas HGB — cierre de esta brecha

Run 20261126T214831_885e8ea1, commit 5582fc28fc8989b742aaa1b86cfb12e06beeb79c.
94/94 pruebas focalizadas aprobadas en 7.43 s; 95 fits nuevos, 75 internos y
20 exteriores. ZIP evidencia.zip SHA256 e2604b2f3fbf3aa761b5495b83924f5a7256ffd3832679f6c45d01047eae25ec;
CRC correcto; manifiesto SHA256 93af2f8fa5dbe2bfb07e3c4caa1b30599a5c197f6941487594390326ce901e29;
8/8 hashes de salidas recalculados. 19578 OOF, seis políticas con 3263 IDs
únicos cada una, idénticos fold/grupo/verdad. Recalculadas métricas de
30 filas por fold y seis resúmenes (RMSE, MAE, R², RMSE global/cola y sesgo).
Cinco particiones exteriores completas/disjuntas; 15 interiores disjuntas,
cada triple de validaciones cubre TRAIN exterior. 75 puntuaciones internas
y 95 trazas de fit. Recalculada selección interna: cinco controles,
ninguna alternativa elegible en ninguna de las cinco particiones; OOF de
inner_selected exactamente iguales a squared_error. No se regeneraron
predicciones internas desde estimadores; se verificó selección a partir
de puntuaciones internas registradas, no prueba independiente.

| Política | RMSE medio | R² medio | MAE medio | RMSE cola >40 | Sesgo cola | Cambio RMSE vs A |
|---|---:|---:|---:|---:|---:|---:|
| squared_error / inner_selected | 7.074911 | .631231 | 3.892598 | 24.888575 | -16.247821 | 0% |
| poisson | 7.342075 | .604482 | 3.845246 | 28.506887 | -20.552137 | +3.78% |
| gamma | 7.475108 | .590341 | 3.920118 | 29.150437 | -20.781379 | +5.66% |
| quantile_060 | 7.595791 | .575489 | 4.116071 | 28.276044 | -20.687031 | +7.36% |
| absolute_error | 7.985510 | .531709 | 4.088510 | 31.843241 | -24.393462 | +12.87% |

La pérdida cuadrática domina en RMSE de los cinco folds exteriores;
Poisson mejora MAE medio pero empeora RMSE y cola. Ninguna alternativa
mejora la cola y todas subestiman más el Li alto. Conservar HGB squared_error
como método de referencia; no repetir estas pérdidas. La decisión está
condicionada a perfiles y datos congelados, no demuestra invalidez universal.
No se supera RMSE OASIS; no se guardaron estimadores ni se promovieron filas.

## 9. Segunda implementación de fase 2B: otros métodos de boosting

Razón: XGB no se evaluó con el conjunto enriquecido fijo de A en la campaña
académica final; CatBoost tiene control RMSE enriquecido previo, pero no
pérdidas Huber/Expectile evaluadas aisladamente en esa campaña. Cerrar
estas brechas antes de pasar a postentrenamiento. Referencias oficiales:
https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html y
https://catboost.ai/docs/en/concepts/loss-functions-regression .
Pseudo-Huber reduce gradiente de errores grandes frente a cuadrática;
Expectile .65 pondera más el error por subestimación (no es cuantil .65).
Delta/slope 10 ppm y alpha .65 son hipótesis fijas, no óptimos demostrados.

Código: integral/boosting_methods.py. Cinco perfiles nuevos:

| Perfil | Método |
|---|---|
| xgb_squared | XGB 400 árboles, .05, depth6, subsample/colsample .9, L1=0, L2=1, min_child_weight=1, cuadrática |
| xgb_regularized | mismo XGB, L1=.1, L2=5, min_child_weight=5; paquete de regularización, no ablación de cada parámetro |
| xgb_pseudohuber | mismo XGB cuadrático base, cambiar objective a reg:pseudohubererror, huber_slope=10 |
| catboost_huber | perfil académico previo 700 iteraciones, depth6, lr .05, L2=3; Huber:delta=10 |
| catboost_expectile | mismo perfil CatBoost; Expectile:alpha=0.65 |

Nuevos ajustes CPU explícito (evitar incompatibilidad de objetivos GPU),
NaN numérico nativo, mismas variables, etiquetas y filas, semilla42.
No alterar preentrenamiento, usar datos externos, ni retocar hiperparámetros
por resultados del TEST. El control CatBoost archivado puede haber usado
un backend diferente (su traza no certifica backend); comparación de
CatBoost nuevo vs archivado es descriptiva, no ablación causal pura de
pérdida. No se hace búsqueda de arquitectura, no Optuna. XGB regularización
agrupada no permite atribuir el efecto a un único parámetro.

Reutilizar 5 controles exteriores HGB y 15 puntuaciones internas cuadráticas
del run HGB validado arriba, y 5 controles exteriores CatBoost del run
20261126T165331_935bc05f, manifiesto
ec1831183f53a4b25cd905ac742ad06a019d1f39ce7132f13c3a0f09fa8baa66.
Verificar hashes de ambos manifests y salidas, fuente, esquema enriquecido,
identidades/verdad/fold/grupo, particiones interiores y exteriores del HGB,
n de cola interna y trazas de columnas. CatBoost archivado no entra en
selección interna: faltan SSE de cola internas. No reconstruirlas a partir
de RMSE global ni consultar TEST para elegir. HGB es fallback. 75 ajustes
internos (5 perfiles x 3 x 5), 25 externos: 100 nuevos; 10 controles
exteriores y 15 puntuaciones internas reutilizadas. Ocho políticas OOF
incluida selección interna, 26104 predicciones esperadas. Seleccionar
solo en TRAIN con guardas idénticas al estudio de pérdidas anterior.
Comparación exterior fija diagnóstica; no seleccionar candidato productivo
retrospectivamente. Reportar si mejoras son compatibles con cola/MAE/R².
Mantener objetivo OASIS como descriptivo. Si no hay mejora suficiente,
no prolongar búsqueda por pérdidas o repetir regularización general:
cerrar brechas restantes justificadamente y auditar postentrenamiento.
No adelantar fine-tuning de fase 6.

Nueve casos nuevos de prueba (103 esperadas en m1_expansion), incluyendo
cinco fits smoke CPU con dependencias opcionales, contratos, guardas,
identidad de particiones y rechazo de solapamiento. Si faltan dependencias,
los smoke se omiten y el experimento requiere instalarlas; no declarar
validación completa con skips no revisados. No ejecutado localmente aún.

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.integral.boosting_methods --allow-unverified-metadata
```

Salida results/generalization/model_1_integral_boosting_methods/<run>.
Adjuntar manifest y salidas completas. Ocho políticas, 40 métricas por fold,
90 puntuaciones internas (15 archivadas), 100 trazas nuevas, cinco
particiones y selecciones, bandas y manifiesto con hashes/runtime.
Sin nuevos datos, guardado, release, main ni M2.


## 10. Corrección de preflight XGBoost
Intento local commit 7625d15: 100 aprobadas, 3 fallidas, 8.32 s. Las tres
pruebas XGB fallaron por colisión entre DeviceInfo de la factoría y parámetro
`device` del estimador. El comando con && impidió iniciar la campaña: 0 fits
experimentales. CatBoost pasó sus smoke tests. Corrección a8f8416: copiar
perfil, separar backend antes de llamar a factoría y fijar model__device=cpu
antes del fit; sin alterar parámetros científicos ni producción. Test
ac97be1 verifica CPU incluso con indicador CUDA y que el perfil no se muta.
104 pruebas esperadas, pendiente validación local. Mismo comando y 100 fits
presupuestados; no repetir campaña HGB.
