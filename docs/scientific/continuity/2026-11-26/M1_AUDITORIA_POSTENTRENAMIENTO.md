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
