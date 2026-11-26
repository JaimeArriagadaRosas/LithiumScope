# M1 — plan aprobado de optimización integral y fine-tuning final

Fecha: 26 de noviembre de 2026. Rama: feat/m1-targeted-data-refinement.
Aprobación explícita del usuario: documentar y ejecutar por etapas la nueva línea
de investigación; primero auditar entrenamiento, después optimizar entrenamiento
y postentrenamiento, integrar metodología, entrenar un candidato de referencia y
solo entonces realizar fine-tuning especializado y validación final.

## 1. Propósito y alcance

Construir el mejor procedimiento reproducible posible para predecir concentración
de Li en ppm con M1 y la información disponible. Diferenciar optimización de la
metodología de aprendizaje de ajuste específico del candidato resultante. El
usuario desea un modelo entrenado con el procedimiento integrado antes de la
última fase de ajuste fino, no otra búsqueda de hiperparámetros sin dirección.

El preentrenamiento (preparación de datos en ML tabular, no pretraining neuronal)
queda cerrado operativamente. No repetir adquisición pública, limpieza,
enriquecimiento, balanceo ni síntesis por defecto. Conservar evidencia negativa.
No cambiar la naturaleza de M1, objetivo Li ppm, entradas disponibles para
inferencia, ni población de evaluación para favorecer métricas. Los predictores
experimentales enriquecidos no constituyen un nuevo contrato productivo aprobado.
Cero muestras externas nuevas aprobadas. No tocar M2, main, modelos activos,
datasets productivos ni publicar/promover sin autorización separada.

La aprobación del plan no implica aprobación automática de todas las técnicas,
de nuevos datos, de un presupuesto ilimitado ni de un release. Avanzar por
criterios de salida verificables. Un candidato experimental puede guardarse
aislado y versionado en fase 5, si supera las guardas de esa fase; no sustituir
artefactos productivos. Si no existe candidato elegible, registrar bloqueo.

## 2. Estado inicial congelado

| Referencia | Población/protocolo | Resultado |
|---|---|---|
| SVM release | 3420 filas, 244 grupos, CV espacial | RMSE global 159.846300, R² global .042980 |
| RF candidato conservado | Mismos TEST espaciales | RMSE global 152.302214, R² global .131184 |
| CatBoost H experimental | Mismos TEST espaciales, transferencia atenuada | RMSE global 111.579554, R² global .533679; no elegible |
| HGB A histórico enriquecido | 3263 filas, CV aleatoria académica | RMSE medio 7.074911, R² medio .631231, MAE 3.892598; control conservado |
| HGB C2 transferencia ponderada | Mismos TEST académicos | RMSE medio 7.053262, R² medio .633404, MAE 3.864404; no aprobado |
| RF de memoria OASIS | 681 filas, VCA 5x5 | RMSE medio 5.361 ± .595, R² medio .499 ± .086 |

C2 empeora RMSE de cola >40 (25.093647 vs 24.888575 de A); IC pareado agrupado
incluye cero. No adoptarlo automáticamente ni volver a incorporar Chen.
Cohorte académica: 135 muestras >40, solo 8 >100; límites congelados
4.2439–126.12 ppm. Mayor dispersión por fold que OASIS. RMSE y R² calculados
correctamente en ppm/escala original; RMSE medio de folds no equivale a global.
Meta descriptiva: RMSE medio <5.361 y R² medio >.499, falta ~24% de reducción
del RMSE respecto de C2. No prometer logro ni superioridad científica directa.
Comparación fuerte requiere conjunto y particiones comunes o reserva
independiente representativa. Los TEST históricos ya se han examinado muchas
veces: son desarrollo, no un holdout nuevo. Las 1483 filas reservadas exploradas
anteriormente tampoco certifican independencia.

Último experimento: results/generalization/model_1_data_closure/20261126T213014_fc01460c,
commit ff4a4a5a7adff25b11d7895903eed1d91b26297f, 85/85 pruebas focalizadas,
15 fits nuevos, 9 hashes verificados. Última suite completa anterior a expansión:
1002 aprobadas, 3 omitidas, 0 fallos, cobertura 57.05%. Falta suite completa
después de las nuevas implementaciones. Estado histórico y hashes en state.json
y M1_CIERRE_DATOS_RESULTADOS.md. No se ha entrenado/guardado un nuevo M1 definitivo.

## 3. Fases de la nueva línea

### Fase 1. Preentrenamiento — cerrada

Conservar procedimiento histórico, alternativas enriquecidas ya documentadas,
los resultados de ponderaciones, transferencias, SMOTER y generación sintética,
sin asumir que todas las técnicas sean compatibles o deban acumularse.
Fuentes, decisiones y hashes en M1_FUENTES_Y_CIERRE.md, M1_AUGMENTATION.md,
M1_EXPANSION.md y M1_CIERRE_DATOS_RESULTADOS.md.
Antes del fit de referencia seleccionar únicamente preparación admisible para el
contrato de inferencia; resolver trazabilidad de recuperación retrospectiva
asistida por Li sin emplear el objetivo de evaluación. Si no es viable,
usar preparación canónica y documentar diferencia. No inventar métodos/CRS.

### Fase 2. Auditoría y optimización del entrenamiento — siguiente

Subfase 2A (primera tarea): inventario exhaustivo técnico/documental en un
Markdown independiente M1_AUDITORIA_ENTRENAMIENTO.md. Revisar:
- registro/factoría de modelos, configuraciones, espacios de búsqueda reales,
  pipelines de entrenamiento, selección, experimentos, métricas, tests y
  artefactos archivados; enumerar familias disponibles frente a ejecutadas.
- cada técnica: estado (disponible, implementada, ejecutada, validada,
  rechazada, pendiente o no aplicable), alcance y población, código y parámetros,
  ID de run, manifest y hashes, métricas, costo, compatibilidad y limitaciones.
- comprobar Optuna, objetivos, rangos, folds, semillas, regularización,
  early stopping, transformaciones del objetivo, ponderaciones,
  pérdidas nativas y personalizadas, modelos especializados, ensambles,
  posibilidad real de continuation/warm_start según biblioteca.
- contrastar alternativas con publicaciones/documentación oficial actual,
  distinguiendo evidencia externa de resultados propios.
- separar evidencia espacial de académica y no contar fits como modelos
  guardados. Si falta evidencia, marcar desconocido; no inventar cobertura.

Subfase 2B: priorizar solo brechas reales, con controles archivados y presupuesto
de fits predeclarado antes de ejecutar. Candidatas a investigar (no aprobadas
como mejoras): pérdidas robustas/asimétricas compatibles con regresión,
regresión cuantílica o heteroscedástica como diagnóstico/alternativa,
regularización y early stopping si no cubiertos; aprendizaje especializado
por perfil químico inferible, continuación de boosting solo si procede.
No repetir campañas convencionales. No cambiar de familia por moda; técnicas
no aplicables a árboles se justifican y se cierran. Evaluar aporte individual
sobre idéntico TEST y selección interna. Criterio: mejora reproducible,
sin deterioro material de métricas ni cola, y disponibilidad en inferencia.

Salida: inventario completo de familias/técnicas, plan experimental limitado,
evidencias y elección de metodología de entrenamiento o justificación de
mantener control. Fase 2 no se marca agotada sin auditar lo realmente probado.

### Fase 3. Auditoría y optimización del postentrenamiento

Auditar primero la corrección aditiva ya ensayada y los expertos/mezclas
existentes. Documentar en M1_AUDITORIA_POSTENTRENAMIENTO.md. Evaluar, solo si
hay soporte bibliográfico y diagnóstico de residuos:
- calibración afín regularizada, multiplicativa o monotónica (isotónica),
  con identidad como control y límites/extrapolación explícitos;
- corrector de residuos r = y - predicción fuera de muestra del modelo base,
  utilizando predicción y características permitidas en inferencia;
- stacking con metaestimador sencillo regularizado, mezclas adaptativas
  basadas en entradas/predicciones disponibles, no en Li real desconocido;
- incertidumbre y detección fuera de dominio como fiabilidad, no confundir
  intervalos de predicción con disminución del RMSE puntual.

Generar predicciones de calibración exclusivamente por cross-fitting dentro
de TRAIN externo. El corrector y la selección no pueden ver la verdad de TEST.
En cada validación interna, producir OOF de su TRAIN interno para entrenar
metaestimador; no reutilizar un OOF global que haya visto el TEST exterior.
No calibrar sobre OOF históricos completos y reportar esos mismos OOF como
evaluación independiente. No aplicar reglas de corrección según banda de Li
real en inferencia. Evaluar si corrección de cola mejora otras regiones;
evitar sobreajuste de isotónica con pocas observaciones extremas.
Si no hay señal residual aprendible fuera de muestra, conservar identidad.

Salida: técnica compatible elegida o cierre negativo, métricas por fold/banda,
trazas de ajuste, costos y pruebas de fuga de datos.

### Fase 4. Integración metodológica

Combinar únicamente tratamientos que aportaron mejora individual verificable.
Comparación/ablación: preprocesamiento congelado + control, entrenamiento
optimizado, postprocesamiento solo, ambos. Comprobar interacciones y
disponibilidad de entradas, alineación de escalas y unidades, ausencia de
fuga de Li/identidad y correcta inversión de transformaciones. Selección
interna, presupuesto limitado, fallback al control. Congelar configuración,
versiones de bibliotecas, semilla, código, hashes y contrato antes de fase 5.
Revisar protocolo de reserva independiente: buscar disponibilidad del dataset
OASIS original y folds si el usuario puede obtenerlos, no reabrir adquisición
masiva de fuentes. Si no están disponibles, distinguir desarrollo de evaluación
independiente; no afirmar superioridad por comparación de cifras heterogéneas.

### Fase 5. Entrenamiento del mejor M1 de referencia

Una vez cerradas 2–4, ejecutar la metodología seleccionada en el entorno local,
medir desempeño mediante evaluación externa por folds y, si cumple guardas,
ajustar un candidato de referencia sobre TRAIN permitido sin tocar la reserva.
Guardar en ubicación experimental separada: pipeline completo (preprocesador,
regresor, calibrador/metaestimador si aplica), esquema de inferencia,
hiperparámetros, semillas, ambiente, identidad de fuente y manifiesto.
Probar guardado/carga, paridad de predicción y recuperación; conservar release
y candidato anterior. No convertir una media de cinco modelos CV en un
modelo único sin entrenarlo; no confundir evaluación CV con fit final.
Si no hay mejora elegible, detener fit del candidato y documentar razones.

Comparar frente a release en protocolo espacial completo y A/C2 en académico
separado. La expectativa de superar release no es garantía ni autorización
de promoción. Si no se cumplen guardas, el candidato no se declara mejor.

### Fase 6. Fine-tuning especializado del candidato

Solo tras obtener una referencia integrada reproducible, realizar refinamiento
local del procedimiento/configuración del candidato, con objetivo principal
RMSE medio <5.361 ppm y R² medio >.499, y guarda de errores por banda.
No cambiar dataset/variables/TEST para conseguir el umbral. Mantener
referencia de fase 5 como control. Para RF/HGB y otros árboles, ajuste fino
suele ser reentrenar clones con cambios acotados; no equivale a actualizar
pesos de una red preentrenada. Warm start/continuación de boosting solo
si la API lo permite, sobre TRAIN compatible, y si la auditoría lo justifica.
No sustituir reducción de RMSE con una métrica más fácil; analizar errores
extremos, estabilidad, MAE, sesgo, incertidumbre e intervalos agrupados.
Predeclarar rangos, número de trials/fits, criterio de parada, particiones y
regla de selección exclusivamente interna. No ajustar sobre la reserva.
Si no se alcanza la meta, reportar mejor evidencia y brecha sin reiniciar
búsquedas indefinidamente.

### Fase 7. Validación final y cierre

Comparación de procedimiento/candidato/fine-tuned/release sobre las mismas
3420 muestras espaciales cuando corresponda, y 3263 académicas congeladas.
Usar datos OASIS comunes si se consiguen de forma autorizada y puede evitarse
solapamiento, manteniendo esa evaluación separada. Una reserva genuina se
abre una sola vez tras congelar procedimiento y selección. Si no existe,
registrar limitación y no declarar certificación independiente.
Reportar RMSE medio y global, R² medio y global, MAE, sesgo, dispersión por fold,
error por concentración (incluida cola), fuentes/grupos, intervalos pareados
y fiabilidad de inferencia. Full suite, regresión, reproducibilidad y
documentación de limitaciones. Fit productivo, promoción, merge y release
requieren decisión explícita posterior; M2 permanece congelado.

## 4. Inventario preliminar (no equivale a auditoría exhaustiva de fase 2A)

- docs/models/M1_REFINAMIENTO_FOCALIZADO.md: 45 configuraciones RF/HGB/CatBoost,
  1350 evaluaciones base y 540 ajustes de calibración cruzada, corrección
  aditiva, transformaciones potencia .75/Yeo-Johnson, ponderaciones,
  selección estable, imputación por litología y SMOTER. Resultados en RESULTADOS.md.
  No confundir 540 ajustes con 540 técnicas de calibración distintas.
- RESULTADOS.md: balanceo 21 configuraciones, 630 evaluaciones base y
  591 ajustes expertos; control RF conservado, expertos sin evidencia
  suficiente de mejora. Revisar archivos reales y distinguir los recuentos.
- M1_EXPANSION.md: transferencia atenuada RF/HGB/CatBoost, selección anidada,
  puente académico, 110 ajustes del refinamiento HGB, 80 ajustes de
  características, 35 ajustes de árboles + 20 estimadores locales del cierre
  histórico, mezclas HGB/CatBoost/analogías sin mejorar RMSE del HGB combinado.
- M1_AUGMENTATION.md: 115 fits de síntesis/balanceo; todos peores que A.
- M1_CIERRE_DATOS_RESULTADOS.md: 15 fits nuevos; C2 mejora .31% RMSE medio
  pero no guarda de cola. No reejecutar.
- Pendiente verificar con el código la cobertura exacta de fine-tuning,
  pérdidas, early stopping, calibración lineal/no lineal, stacking entrenado,
  residuos multivariados, incertidumbre y warm start. No declararlas ausentes
  ni implementadas por el nombre de un módulo. Distinguir posibilidad técnica
  de justificación estadística. La auditoría precede a nuevos fits.

## 5. Continuidad al cambiar de chat

1. Abrir README.md de esta carpeta y este plan. Leer state.json (m1.integral_optimization
   identifica fase vigente; m1.augmentation conserva cierre histórico).
2. Leer M1_CIERRE_DATOS_RESULTADOS.md, M1_FUENTES_Y_CIERRE.md,
   M1_DATA_CHECKPOINT.md y M1_EXPANSION.md; después RESULTADOS.md,
   EVIDENCIAS.md y docs/models/M1_REFINAMIENTO_FOCALIZADO.md según necesidad.
3. Inspeccionar el repositorio en la rama exacta, nunca inferir paths de memoria.
   Actualizar inventario y auditoría de fase 2A antes de escribir entrenamiento.
   No pedir al usuario repetir información ni recuperar archivos que ya existan.
4. Registrar por fase commits, código, tests, comando CMD local exacto,
   outputs, hashes, resultados y decisión. Diferenciar implementado de validado.
   Actualizar README/state al cerrar cada fase; conservar historial negativo.
5. Usuario ejecuta localmente en D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope.
   No usar conexión remota sin autorización actual; no afirmar ejecución local
   desde GitHub. No publicar datasets/evidencias privadas. Mantener M2 intacto.
6. No hay comando pendiente ahora: el siguiente trabajo es auditoría documental
   y técnica de fase 2A. Al llegar a ejecución, dar solo el comando necesario.

Mensaje sugerido para retomar en otro chat:
"Continúa LithiumScope en feat/m1-targeted-data-refinement. Lee
docs/scientific/continuity/2026-11-26/README.md, state.json y
M1_PLAN_OPTIMIZACION_INTEGRAL.md. El plan está aprobado. Retoma la fase vigente
sin repetir datos ni campañas anteriores; solo M1, M2 congelado."

## 6. Estado al aprobar

Fase 1 cerrada; fases 2–7 pendientes. Fase 2A siguiente. Ningún nuevo ajuste,
modelo, cambio de código o prueba ejecutado por este registro documental.
La fase 7 del plan de aumento previo sigue sin ejecutar; este plan nuevo
no altera sus resultados ni convierte un experimento negativo en éxito.
