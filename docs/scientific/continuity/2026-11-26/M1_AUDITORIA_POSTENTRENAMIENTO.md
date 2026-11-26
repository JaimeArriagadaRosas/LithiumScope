# M1 — auditoría y optimización de postentrenamiento

Fecha: 26/11/2026. Plan aprobado M1_PLAN_OPTIMIZACION_INTEGRAL.md.
Fase 3, primera implementación pendiente de ejecución. Fase 2 cerrada
operativamente con HGB cuadrático como control enriquecido A; no afirmar
exhaustividad de todas las posibles configuraciones. Solo M1, investigación.
No adquirir datos, alterar variables originales, entrenar candidato definitivo
ni modificar M2, producción, main o release.

## 1. Auditoría de técnicas anteriores

Se revisaron árbol de la rama, targeted/fit.py, balanced/regimes.py,
expansion/blend_policy.py, academic_final.py, uncertainty_study.py y
model_1/diagnostics/group_intervals.py (evaluación es import de compatibilidad).

| Técnica | Evidencia | Decisión |
|---|---|---|
| Proyección no negativa y límites de soporte | treatment_policy.py; campañas previas | conservar no negatividad; límite de TRAIN no es máximo geológico |
| Corrección aditiva | targeted/fit.py: residual medio OOF GroupKFold(3) en TRAIN, junto a sqrt; 540 ajustes adicionales de modelos en campaña focalizada | no repetir combinado; aislar sobre HGB cuadrático enriquecido |
| Mezcla de expertos | balanced/regimes.py; gate RF X-only y regresores por régimen; 591 ajustes expertos | no mejora demostrada; no repetir |
| Mezcla convexa y cap de discrepancia | expansion/blend_policy.py; selección interna; academic_final.py HGB/CatBoost/analogía | evaluada; combinación no supera HGB |
| Calibración afín e isotónica | no se encontró evaluación aislada suficiente sobre A en módulos revisados | nueva campaña |
| Regresión residual multivariada | no se encontró evaluación aislada suficiente sobre A | nueva campaña con Ridge y predictores canónicos |
| Stacking aprendido | mezcla fija previa no es stacking entrenado | posible campaña posterior con crossfit de HGB/XGB; condicionado a evidencia y costo |
| Intervalos conformales por grupo | diagnostics/group_intervals.py: GroupShuffleSplit, máximo de errores por grupo, radio y cobertura simultánea; uncertainty_study.py requiere --execute para fit | código disponible, ejecución histórica no revalidada; incertidumbre no mejora por sí sola RMSE puntual |

Revisar existencia de artefactos locales de incertidumbre antes de reejecutar.
Calibración M2 de clasificación es otra tarea: fuera de alcance. No afirmar
que la auditoría exhausta toda biblioteca o certifica todos los runs locales.
No confundir corrección de predicción puntual con cobertura probabilística.
La validez conformal depende de intercambiabilidad de grupos, no demostrada.

## 2. Primera campaña: calibración de predicción puntual y residuos

Código integral/calibration.py y integral/posttraining.py. Base congelada
HGB cuadrática, sin nuevos datos ni hiperparámetros; mismo enriquecimiento,
selección TRAIN de columnas y cohorte académica de 3263. Control exterior
archivado, validado con manifest de pérdidas HGB y referencia de
características. Corregir predicción en ppm, no modificar la variable objetivo.
Los correctores usan solamente predicción del modelo y, para Ridge
multivariado, las 33 características canónicas disponibles. No Li real,
fuente/ID de fila ni predictor adicional en inferencia. Ajuste de imputación,
escalado y codificación dentro de cada corrector TRAIN. No reusar objetivos
de TEST. Cinco políticas fijas, más política seleccionada internamente.

| Política | Definición |
|---|---|
| identity | predicción HGB original |
| additive | añadir media de y - OOF |
| affine_shrunk | Ridge alpha10 sobre predicción estandarizada, aprende residual; aplicar .5 de corrección |
| isotonic_shrunk | ajuste isotónico creciente de y frente a OOF, out_of_bounds=clip; aplicar .5 de diferencia |
| residual_ridge | Ridge alpha100 sobre predicción y 33 predictores originales (mediana/escala/one-hot); aplicar .5 del residual predicho |

Todas las correcciones acotadas a ±.5*max(predicción, max(1, mediana OOF
TRAIN)); proyección final a >=0 ppm. El cap y shrinkage son hipótesis
predeclaradas, no óptimos ni límites físicos. No se tunean sobre TEST.
Isotónica estima relación monótona entre predicción y verdad, no garantiza
generalizar fuera del soporte. Ridge residual puede aprender señal de error
restante, no garantiza extrapolar mejor. Aditiva aislada sobre HGB cuadrático
es distinta del ensayo anterior combinado con raíz cuadrada.
La calibración usa respuestas reales TRAIN para aprender errores; se aplica
con predicción/variables disponibles sin conocer respuesta de evaluación.

## 3. Protocolo de cross-fitting anidado

Por cada fold exterior:
- Verificar hashes archivados, identidades y esquema, misma partición externa
  y tres internas del estudio de pérdidas HGB (semilla42+fold).
- Cada TRAIN interno produce OOF de tres subfolds (seed42+100*fold+j),
  ajustando HGB y selección de variables solo en cada sub-TRAIN. Ajustar
  correctores con esas predicciones y verdades del TRAIN interno.
- Ajustar HGB en TRAIN interno y predecir validación interna sin verla.
  Reproducir scores de control archivado (RMSE, R², MAE, SSE cola) con
  tolerancia numérica 1e-6; fallar explícitamente si no coinciden.
- Aplicar cada corrector a validación interna; seleccionar por menor RMSE
  medio que identidad, R² medio >.499, MAE <= identidad+.1 y RMSE de
  cola global no peor. Fallback identidad. No seleccionar con TEST.
- Reusar las tres predicciones de validación interna del modelo base
  (sin corrección) como OOF que cubre TRAIN exterior. Ajustar correctores
  sobre ese TRAIN, y aplicar a predicciones HGB exteriores archivadas.

Coste: 5 x 3 x (3 subfits + 1 base interno) = 60 nuevos fits HGB.
20 conjuntos de calibración (15 internos, cinco externos): 60 fits de
correctores (Ridge afín, isotónica, Ridge multivariado), 20 estadísticas
de sesgo. Cinco bases externas archivadas, no volver a ajustarlas.
75 scores interiores, 30 scores exteriores, 19578 OOF de evaluación,
60 trazas base y 20 trazas de calibración, IDs explícitos, predicciones
OOF de calibración exportadas para auditar exclusiones y cobertura.
Los folds aleatorios académicos comparten grupos espaciales: esto es
crossfit por identidad en desarrollo, no independencia espacial.
La recuperación retrospectiva asistida por objetivo y metadatos no
certificados se mantienen como limitaciones explícitas; no se aprueba
su uso productivo. No se confunde resultado con OASIS, no modificar TEST.

## 4. Verificación y decisión

Seis pruebas nuevas (110 esperadas frente a 104 verificadas): identidad,
correcciones finitas y acotadas, sesgo TRAIN, esquema/solapamiento,
crossfit cobertura, datos inválidos y selección guardada.
Ejecutar desde raíz:

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.integral.posttraining --allow-unverified-metadata
```

Salida results/generalization/model_1_integral_posttraining/<run>.
Adjuntar consola y ZIP/RAR de la ejecución completa. Recalcular hashes,
métricas, exclusiones TRAIN/evaluación en todos los niveles y selección.
Si ninguna corrección aporta, conservar identidad. No afirmar mejora
por comparar error in-sample de calibración. Después decidir si stacking
aprendido, residual no lineal o incertidumbre requieren campaña adicional
justificada; no asumir que todo debe combinarse. No iniciar fase 4
hasta cerrar fase 3. La validación espacial/independiente sigue pendiente.
No se ha ejecutado en Windows ni guardado nuevo estimador; las pruebas
son nuevas y pendientes de validación. Fase 2 no se reejecuta.


## 5. Calibración puntual: resultados verificados

Run 20261126T223705_0c698520, commit 80205aa432d05519a7e5679c6aa8c978f18ce2fc.
110/110 pruebas focalizadas, 28.45 s. RAR Evidencias(4).rar SHA256
9f2b1e33ec110292e2d584cb3574af280d477d4a57f692761d80b2c51cc85122.
Manifest SHA256 e3c94c67492c219925b8e9ccc355d9e75d2dcd99c4f677f94ce38c95da31c852;
9/9 hashes correctos. 60 fits base, 20 conjuntos de calibración, 60
estimadores correctores y 20 estadísticas de sesgo. 19578 OOF de evaluación,
39156 OOF de calibración, 60 trazas base, 20 trazas correctores y 75 scores
internos. Se verificaron identidades/verdad/fold/grupo, disjunción de TRAIN
y evaluación en los cinco folds exteriores, 15 internos y 45 subfolds,
cobertura de TRAIN por subvalidaciones, correspondencia de los IDs de
calibración, valores finitos, selección interna reproducida y predicciones
seleccionadas idénticas a su política. No se regeneraron todos los estimadores
ni las predicciones individuales de calibración en chat.

| Política | RMSE medio | R² medio | MAE medio | RMSE cola >40 | Sesgo cola |
|---|---:|---:|---:|---:|---:|
| identity | 7.074911 | .631231 | 3.892598 | 24.888575 | -16.247821 |
| additive | 7.073829 | .631361 | 3.888219 | 24.953450 | -16.354732 |
| affine_shrunk | 7.079459 | .630774 | 3.895006 | 25.040810 | -16.625328 |
| isotonic_shrunk | 7.079013 | .630329 | 3.909470 | 25.064379 | -16.458991 |
| residual_ridge | 7.119657 | .626340 | 3.923507 | 25.145015 | -16.590737 |
| inner_selected | 7.071948 | .631321 | 3.904815 | 24.830341 | -16.003423 |

Selección: isotónica, identidad, Ridge residual, identidad, identidad.
Única política que pasa guardas descriptivas vs A: inner_selected. RMSE
medio -0.002963 ppm (~0.042%), RMSE global 7.096043 vs 7.100761; cola
-0.058234 (~0.234%), MAE +0.012217 dentro de tolerancia .1. Bootstrap
pareado por grupo espacial 5000 réplicas, semilla42, IC percentil 95%
de delta RMSE global [-.063239,.031412] ppm, incluye cero. Descriptivo,
no confirmatorio: grupos y TEST reutilizados. Mantener identidad como
control y selección de calibración como candidata experimental para
integración, sin adoptarla automáticamente. No confundir con C2 (7.053262,
no aprobado por cola). No supera RMSE de OASIS ni se guardó estimador.

## 6. Fase 3B: última comparación acotada de postentrenamiento puntual

Código integral/stacking.py e integral/stacking_study.py. Se cubren dos
brechas: combinación aprendida de HGB con XGB regularizado, y corrección
residual no lineal de HGB. El XGB fue elegido como componente competitivo
a partir de desarrollo anterior (no selección independiente); no volver
a comparar toda la parrilla de familias. Mantener cohorte, etiquetas,
esquema y perfil de cada base. Sin nuevos predictores externos ni muestras.
La predicción de cada base es característica interna del metaestimador.

Políticas internas:
- identity: HGB cuadrático.
- equal_blend: mezcla fija HGB/XGB al 50%, control de combinación.
- convex_stack: peso XGB aprendido mediante mínimo cuadrático con
  penalización n*w², acotado 0–.75; HGB recibe peso 1-w.
- ridge_stack: Ridge alpha100 sobre dos predicciones estandarizadas,
  ajusta residual de HGB, aplica .5 de corrección.
- nonlinear_residual: HGB superficial (100 iteraciones, 7 hojas,
  mínimo40 por hoja, lr .05, L2=10), residual a partir de 33
  características canónicas + predicción HGB; aplica .5 de corrección.
Ambas correcciones residuales acotadas a ±.5*max(predicción HGB,
max(1,mediana HGB OOF TRAIN)) y proyectadas a >=0 ppm. Parámetros
predeclarados, no son óptimos demostrados ni límites físicos. Los pesos
convexos y la mezcla fija no son calibración probabilística.

Comparación externa adicional, descriptiva: XGB regularizado solo,
selección de calibración anterior archivada, y selección interna de
las cinco políticas de esta campaña. Ocho políticas exteriores; las
dos referencias archivadas extra no participan en selección interna.
Guardas RMSE, R², MAE y cola idénticas a fase 3A. HGB fallback.

Reutilizar los 39156 registros HGB OOF de calibración de fase 3A
(60 fits archivados) y las cinco predicciones exteriores HGB/XGB
respectivamente; también las cinco predicciones seleccionadas de
calibración anterior. Verificar hashes del run anterior y del boosting
(ba2090c1e92a62a669fe175382db61d8e124baf2cd7f38eee0977c2d988e7f9f),
cohortes, perfiles, índices, verdad, esquemas y trazas. Reconstruir XGB
solo en las mismas 45 subparticiones y 15 TRAIN internos que HGB
archivado. Validar RMSE/R²/MAE/SSE cola de predicciones internas de ambos
contra scores archivados; abortar ante discrepancia numérica. Cada
corrector se ajusta con OOF dentro del TRAIN específico, nunca sobre
validación interna ni TEST exterior. En exterior se usan las bases OOF
de los tres folds internos para ajustar el metaestimador.

Presupuesto 60 nuevos fits XGB, 20 Ridge y 20 HGB residuales
(40 metaestimadores), 20 pesos convexos analíticos. No nuevos fits base
HGB ni exteriores. 26104 OOF exteriores, 39156 filas de calibración
pareadas, 75 scores interiores, 40 exteriores. Cinco pruebas nuevas,
115 esperadas (110 verificadas anteriormente). No ejecutado aún.
No se afirma que haya mejora por añadir complejidad. Si no mejora con
estabilidad, conservar control o candidato simple; no extender
indefinidamente postentrenamiento puntual. La incertidumbre agrupada
es diagnóstico de fiabilidad separado, pendiente verificar evidencia
local; no sirve para reducir directamente RMSE. Después resolver
contrato de inferencia y validación espacial de integración, antes de
fit de referencia y ajuste fino. No tocar M2, main ni producción.

CMD:
```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.integral.stacking_study --allow-unverified-metadata
```
Salida results/generalization/model_1_integral_stacking/<run>.
Enviar consola y evidencias completas. No repetir posttraining.
