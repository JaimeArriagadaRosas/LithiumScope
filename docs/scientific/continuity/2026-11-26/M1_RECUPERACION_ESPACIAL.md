# M1 — recuperación acotada de estabilidad espacial

Aprobación del usuario: continuar la revisión dirigida tras el bloqueo de
fase 4. No se reabre la adquisición ni se modifica M2, main o producción.
Run de integración de origen 20261126T232226_c7612b4d, manifest SHA256
0ec679dfe7c15a318869bbcdb65bbb5e148dfba089ade5d993da5649db9b5d72.
124 pruebas aprobadas; 90 ajustes verificados. Fase 5 sigue bloqueada.

## Hipótesis y alcance

Las falsas concentraciones elevadas de boosting dañan la estabilidad entre
grupos. No está demostrada su causa química. Se investiga si desacuerdo de
estimadores y soporte de concentraciones en vecinos químicos de TRAIN
identifican predicciones poco fiables. Un RF de perfil fijo (500 árboles,
min_samples_leaf=1, max_features=.8, semilla42; preprocesador canónico)
actúa como respaldo; no es el RF conservado con hiperparámetros por fold.
Este perfil proviene del puente académico previo. Los OOF históricos de RF
conservado y SVM sirven exclusivamente como referencias espaciales.

## Protocolo predeclarado

Reutilizar los OOF de integración de HGB y Ridge, verificar manifiesto,
seis hashes, fuente, identidad, grupo, verdad y fold. Repetir comprobación
de paridad del contrato canónico sin Li en entrada. Mismas 3420 filas/244
grupos espaciales y 3263 académicas aleatorias, cinco folds por protocolo.
Entrenar solamente un RF en cada TRAIN exterior, 10 fits nuevos; no
reentrenar HGB, XGB ni Ridge. No incluir valores de evaluación en la
regla de corrección ni recalibrar el stacking con estos TEST.

Soporte geoquímico: reutilizar multivariate_support de expansion/identity.py,
22 posibles mediciones químicas, sin coordenadas ni target en la distancia,
mediana e IQR TRAIN, campos no constantes con cobertura >=50%, al menos
cinco campos válidos, indicadores de ausencia; distancia al vecino más
cercano frente a q95 de distancias leave-one-out de TRAIN. Además, hasta
15 vecinos de TRAIN dentro de dos veces ese radio; contar grupos distintos
con concentración medida >= max(40 ppm, 0.5*predicción Ridge). Considerar
respaldo de concentración alto si existe soporte químico básico y >=2
grupos. Sin cobertura suficiente, el soporte se declara falso. Esto es
un indicador heurístico, no prueba de equivalencia mineralógica o de
independencia de grupos. Vecindad química no equivale a vecindad geográfica.

Desacuerdo de riesgo: predicción Ridge >40 ppm, >2 veces RF y diferencia
Ridge−RF >20 ppm. Se mantiene Ridge salvo que se active una regla:

| Política | Activación del respaldo RF |
|---|---|
| ridge_disagreement | Desacuerdo de riesgo |
| ridge_supported | Desacuerdo y falta de soporte químico básico |
| ridge_high_support | Desacuerdo y falta de respaldo de concentraciones altas |

La última es la política primaria fija, no se elige con resultados TEST.
Las otras dos son diagnósticos. Umbrales heurísticos predeclarados para
esta campaña (no optimizados ni garantizados); los 40 ppm ya son el
umbral de cola de anteriores campañas. La concentración usada para
contar grupos corresponde exclusivamente a etiquetas de TRAIN. No usar
row_id, grupo TEST, región concreta, etiqueta TEST, ni las nueve
observaciones >696 como reglas de inferencia. Riesgo: el respaldo puede
reducir falsas alarmas pero también subestimar verdaderos extremos.
Se conservan todas las filas y se evalúan los dos protocolos por separado.

## Criterios y evidencias

Usar better() de integración sin relajar guardas. La primaria debe
mejorar HGB canónico académico en RMSE medio/global, cola no peor,
MAE global <= control+.1, R² medio >.499. En espacial debe superar
SVM, RF conservado y HGB canónico en RMSE medio/global, cola >40,
extremos >696 no peores y MAE global <= cada control+.1.
No seleccionar retrospectivamente política diagnóstica aunque tenga
mejores métricas. Ningún resultado es independiente: la hipótesis surge
de observar errores en folds de desarrollo ya reutilizados. Si falla,
registrar bloqueo; no prolongar barridos arbitrarios ni rebajar guardas.
Si pasa, revisar persistencia de la metodología completa y viabilidad de
fase 5 antes de guardar candidato. Objetivo OASIS descriptivo separado.

Salida esperada: 46.938 predicciones (8 políticas espaciales, 6 académicas),
6683 filas de diagnóstico de soporte y activaciones, diez trazas TRAIN/TEST,
14 resúmenes, decisión primaria y manifest con hashes. Seis tests nuevos,
130 esperados frente a 124 comprobados. No se ha ejecutado localmente;
no se ha ajustado ni guardado modelo en esta intervención.
No incorporar muestras o modificar código productivo.

CMD desde raíz:
```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.integral.spatial_recovery
```

Salida results/generalization/model_1_integral_spatial_recovery/<run>.
Enviar consola y archivo completo. Verificar activaciones, identidad,
exclusiones, métricas y guardas. Examinar soporte de los grupos problemáticos
con el diagnóstico exportado; no declarar su causa a priori.
