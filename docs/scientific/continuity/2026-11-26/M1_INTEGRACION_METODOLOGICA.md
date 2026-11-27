# M1 — integración metodológica y contrato de inferencia

Fecha: 26/11/2026. Fase 4 del plan aprobado, implementación pendiente
de ejecución local. Preentrenamiento, entrenamiento y postentrenamiento
puntual cerrados operativamente; fine-tuning aún no iniciado.
Solo M1; no adquirir datos, modificar modelos activos, main o M2.

## 1. Decisión sobre enriquecimiento e inferencia

El mejor resultado descriptivo enriquecido es mezcla 50/50 RMSE medio
7.021003, R² .637216, no cumple guarda de cola. Ridge enriquecido
7.034486/.635544, cola 24.844553, sí cumple guardas descriptivas.
Los experimentos anteriores de enriquecimiento incorporan recuperación
retrospectiva de mediciones mediante vínculos asistidos por Li real.
No se ha demostrado que las identidades se puedan resolver sin Li en
inferencia ni que todas esas variables estén disponibles en las entradas
del producto. No utilizar la recuperación ni sus variables en el candidato
de referencia por defecto. No eliminar las evidencias enriquecidas: son
comparaciones de investigación, no certificación de un contrato ampliado.
La preparación canónica conserva 33 variables exactas (10 óxidos,
12 elementos traza, dos coordenadas, cuatro derivadas, cinco categorías).
No se altera el objetivo Li_icpms ppm, fuentes, QC o cohortes congeladas.
Se prueba paridad de prepare_prediction_data sobre filas históricas
originales sin la columna objetivo, contra matriz prepare_training_data,
incluyendo orden, faltantes y categorías. Una discrepancia aborta
antes de entrenar; no corregir silenciosamente ni cambiar el dataset.
La paridad no implica garantía de calidad analítica ni de metadatos.

## 2. Integración predeclarada

Código integral/inference_contract.py e integral/integration.py.
Perfiles fijos de campañas previas: HGB cuadrático, 500 iteraciones,
31 hojas, mínimo 5, lr .05, L2 .1, sin early stopping aleatorio;
XGB cuadrático regularizado 400 árboles, lr .05, depth6, subsample
y colsample .9, L1 .1, L2 5, min_child_weight 5, CPU hist.
Preprocesador canónico con NaN nativo numérico, codificación estable,
semilla42. No tuning de parámetros en esta fase. Las etiquetas
válidas no negativas (incluido cero en dominio completo) se conservan.
Ridge alpha100 estandariza las dos predicciones OOF de TRAIN,
ajusta y−HGB, aplica mitad del residual predicho, acota corrección
a ±.5*max(HGB,max(1,mediana HGB OOF TRAIN)), salida no negativa.
La clase RidgeCorrection es serializable; prueba de roundtrip con
datos sintéticos, no se guarda modelo entrenado definitivo aún.
Comparar HGB, XGB y Ridge con exactamente los mismos predictores.
No combinar la calibración previa con Ridge: su efecto combinado no
está validado y la ganancia previa era marginal. XGB solo se reporta
como control de componente, no como candidato de referencia por defecto.

## 3. Dos protocolos separados

- Espacial completo, 3420 filas, 244 grupos, cinco folds históricos.
  Verificar fuente SHA256 94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94,
  manifiesto exterior histórico SHA b4d2af7501933169e69ecc99073266a6479fa3fdfb4efc7e978113460f5fe977,
  posiciones, índices, coordenadas, grupos; train/test separados.
  Recuperar OOF SVM de training_20261106_111653_m0300 y RF de
  training_20261125_115932_m0300_826123c1, mismos TEST e identidades,
  comprobar run.json, hash de fuente, verdad y RMSE global archivado
  159.846300 y 152.302214, registrar hashes de tablas. Son referencias
  OOF históricas, no una nueva ejecución directa del artefacto desplegado.
  Tres subfolds GroupKFold dentro de TRAIN exterior para aprender Ridge,
  aislamiento de grupos en ambos niveles; no recortar casos extremos.
- Académico, 3263 filas congeladas, KFold aleatorio seed42 igual al piloto.
  Verificar identidad del piloto y manifiesto/hash del stacking.
  Recuperar HGB y Ridge enriquecidos como contraste descriptivo de
  diferencia de entradas; no usar sus predicciones al ajustar Ridge
  canónico. Tres subfolds KFold con seed42+fold en TRAIN exterior.
  Grupos compartidos, desarrollo reutilizado; no independiente.
  Objetivo OASIS solo descriptivo RMSE medio <5.361, R² medio >.499.
  No confundirlo con el RMSE global del protocolo espacial.

Cada fold: dos modelos base x tres subfolds para OOF TRAIN y dos
modelos base sobre TRAIN completo, Ridge ajustado exclusivamente
en predicciones OOF de TRAIN. Sin selección basada en TEST,
sin recuperar variables externas, sin reejecutar los ajustes
enriquecidos. 10 folds entre los dos protocolos x 8 = 80 fits
base (40 HGB y 40 XGB), 10 Ridge. Referencias externas archivadas.
Predicciones: 5 políticas por protocolo, 33415 OOF esperadas.
Calibración OOF pareada 26732 filas; 40 trazas de pares de modelos.
Registrar RMSE medio/global, R² medio/global, MAE medio/global, RMSE
cola >40, sesgo cola, extremos >696 (RMSE si hay observaciones),
guardas y hashes. Los dos protocolos no se mezclan en una métrica.

## 4. Guardas de revisión de candidato

La comparación es de desarrollo y no habilita release automáticamente.
HGB canónico requiere R² medio académico >.499 y superar ambos
controles SVM/RF espaciales en RMSE medio y global; cola >40 y extremos
>696 no peores, MAE global dentro de .1 ppm del control respectivo.
Ridge requiere además superar HGB canónico en ambos protocolos
en RMSE medio y global, conservar guarda de cola, MAE global <=
control+.1, R² medio académico >.499 y extremos espaciales no peores.
Estas guardas son predeclaradas para la revisión experimental; no son
umbrales geológicos ni un cambio de criterios del release. Si ninguna
pasa, no guardar un candidato como mejor; documentar el bloqueo y
revisar evidencia, no ajustar retrospectivamente la guarda. Incluso
si una pasa, no demostrar superioridad independiente con TEST conocidos.
Fase 5 seguirá condicionada a la evidencia, evaluación de reserva
y autorización de alcance experimental. No confundir mejora de la
metodología con lograr objetivo académico absoluto. El fine-tuning
posterior queda separado, no reabrir búsqueda de datos.

## 5. Ejecución y continuidad

Seis pruebas nuevas: nombres exactos del contrato, paridad con objetivo
retirado, crossfit cobertura y exclusión, serialización y límites Ridge,
smoke real HGB/XGB y guardas de decisión. Esperadas 121 pruebas frente
a 115 verificadas, no ejecutadas en Windows todavía.
Desde la raíz:

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.integral.integration
```

Salida results/generalization/model_1_integral_integration/<run>.
Entregar consola y archivo de evidencias completo. Si falla la paridad o
una referencia archivada, resolver esa identidad antes de iniciar fits.
No ejecutar nuevas búsquedas, ni entrenar el candidato todavía.
No se afirma certificación de metadatos o evaluación independiente;
los datos históricos y modelos activos permanecen intactos.


## 6. Corrección de compatibilidad de OOF históricos
Intento local commit 1b0cb60: 121/121 pruebas aprobadas en 7.76 s.
Fuente, cohorte, paridad sin objetivo y manifiesto espacial superaron
preflight; KeyError row_position al leer OOF SVM legado. Ningún fit de
integración iniciado. El lector anterior asumía formato moderno.
El procedimiento histórico load_m1_oof de compare_vnext_trained_runs.py
confirma que row_index vive en OOF y los campos espaciales/row_position
en outer_fold_manifest.csv. Corrección 5539d47: alinear por row_index
único con el manifiesto de cada run, validar cobertura completa,
posiciones, folds, grupos, coordenadas, verdad y predicciones finitas.
Verificar campos redundantes si existen; rechazar inconsistencias.
Mantener RMSE esperado, hash de fuente y registrar hashes de ambas tablas
y ambos manifiestos, sin modificar originales. Tres pruebas nuevas f43deda:
formato legado reordenado, formato moderno con campos redundantes, rechazo
de IDs/verdad/manifiesto incorrectos. 124 pruebas esperadas. Mismo comando
y presupuesto 80 bases +10 Ridge; pendiente ejecución local. Sin cambios
metodológicos, ni nuevos datos, ni modelos productivos/M2.


## 7. Integración ejecutada, verificada y no elegible

Run 20261126T232226_c7612b4d, commit a2832985584d439a4492112ff1e15ea0a25475d4.
124/124 pruebas focalizadas aprobadas en 8.69 s; 80 fits base +10 Ridge.
ZIP Evidencias(10).zip SHA256 4a8226c17ddd0729c93d7f707f6c8a69f65ba21dd37623baf4db4400305e40bf.
Manifest SHA256 0ec679dfe7c15a318869bbcdb65bbb5e148dfba089ade5d993da5649db9b5d72.
CRC correcto, 6/6 hashes de salidas recalculados.
33415 OOF de evaluación (5 políticas por protocolo, 3420 espaciales y 3263
académicas), 26732 OOF de calibración y 40 trazas de pares base.
50 métricas por fold y diez resúmenes recalculados. Identidades, verdad,
fold, grupo, predicciones finitas, cobertura y disjunción por identidad
de todos los TRAIN/TEST y subfolds; disjunción de grupos en protocolo
espacial. Diez correctores Ridge reconstruidos a partir de los OOF de
TRAIN exportados: reproducen las predicciones corregidas con tolerancia
numérica. No se volvieron a ajustar los 80 modelos base. La paridad
sin objetivo de 3420x33 está registrada en el manifiesto y el preflight
local; no se inspeccionó el CSV histórico bruto en chat.

Fuente y ambos manifiestos exteriores coinciden con los hashes congelados.
OOF SVM SHA256 9573fdd444696be16439debb0cd9b7b2f955ee004fe8af74d5cff46f6a6842c3;
RF SHA256 4496741fbaf28d710af22e9b2568d10e139768e515b9171853c42d950656b4c9.
Sin mediciones recuperadas ni enlace asistido por objetivo en los nuevos
modelos, sin filas nuevas, sin modificación de contrato de producción.

| Académico | RMSE medio | R² medio | MAE global | RMSE cola >40 |
|---|---:|---:|---:|---:|
| HGB enriquecido (histórico descriptivo) | 7.074911 | .631231 | 3.892448 | 24.888575 |
| Ridge enriquecido (histórico descriptivo) | 7.034486 | .635544 | 3.865885 | 24.844553 |
| HGB canónico | 7.239729 | .611629 | 4.025531 | 25.398793 |
| Ridge canónico | 7.185906 | .617830 | 3.993663 | 25.237865 |
| XGB canónico | 7.265974 | .610501 | 4.036117 | 25.229835 |

Ridge reduce RMSE medio ~.74% frente a HGB canónico, y pasa guardas
académicas de comparación, pero no objetivo absoluto OASIS. Diferentes
poblaciones y protocolo; no afirmar superioridad directa.

| Espacial completo | RMSE medio | RMSE global | R² medio | R² global | MAE global | RMSE >40 | RMSE >696 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM release OOF | 87.448705 | 159.846300 | .246772 | .042980 | 16.320391 | 635.179289 | 3067.144820 |
| RF conservado OOF | 82.310168 | 152.302214 | .251477 | .131184 | 15.303595 | 604.981485 | 2932.717138 |
| HGB canónico | 109.262477 | 150.269621 | -7.098508 | .154219 | 17.969185 | 549.250259 | 2568.500470 |
| XGB canónico | 107.920694 | 151.691814 | -5.952006 | .138134 | 17.937807 | 563.709175 | 2659.647019 |
| Ridge canónico | 118.718659 | 156.647097 | -8.117336 | .080905 | 19.742290 | 572.688238 | 2544.342308 |

## 8. Diagnóstico del bloqueo espacial

Todos los folds tienen 684 filas. HGB reduce error de los nueve casos
>696, pero crea sobreestimaciones graves donde el valor real es bajo:
row_id 1191, fold 1, g_143_221: real 9.2 ppm; HGB 3413.126085,
XGB 3106.281006, Ridge 3557.789520, RF 25.908126, SVM 74.257802.
Una sola fila representa ~98.5% del SSE HGB del fold 1.
En fold 3 g_151_219 concentra 98.63% del SSE HGB y 99.00% del
SSE Ridge. HGB: 91→1058.813, 77→908.041, 282→1066.164 ppm.
No se elimina ninguna fila ni se asigna causa química sin datos brutos.

Diagnóstico por bandas de Li real (solo para análisis, nunca para
seleccionar rama de inferencia):
| Banda | n | RMSE RF | RMSE HGB | RMSE Ridge | RMSE XGB |
|---|---:|---:|---:|---:|---:|
| <=40 | 3204 | 9.241475 | 61.364268 | 63.891483 | 56.027724 |
| 40–126.12 | 135 | 51.330466 | 115.521252 | 163.998844 | 108.755187 |
| 126.12–696 | 72 | 134.029041 | 235.278627 | 352.528638 | 216.591552 |
| >696 | 9 | 2932.717138 | 2568.500470 | 2544.342308 | 2659.647019 |

RMSE diagnóstico de las 3411 observaciones <=696: RF 23.742053,
SVM 28.225520, HGB 72.344270, XGB 66.384827, Ridge 86.730120.
Los nueve extremos representan 97.58% del SSE RF y 76.88% del SSE HGB.
La mejora global HGB (-1.33% frente a RF) no implica robustez territorial:
RMSE medio y MAE incumplen guardas. Ridge mejora académico pero empeora
espacial. En particular, Ridge no es un detector de extrapolación.
No recortar la evaluación espacial ni usar Li real como regla de predicción.

## 9. Decisión y continuidad

El experimento de fase 4 terminó correctamente, pero HGB y Ridge no son
elegibles para revisión de candidato. No modificar los criterios ni
declarar un nuevo modelo superior. Se conserva RF como candidato previo
y SVM como release; no se cambia el estado de producción. Fase 5
bloqueada por calidad de generalización, no por fallo técnico. Fases
6 y 7 no iniciadas. 505 ajustes ejecutados en fases 2–4 (195 +120 +100 +90);
no incluye cálculos analíticos. No hay CMD pendiente ni nuevo fit autorizado
por este cierre. Última suite focalizada 124/124, no suite completa.

Siguiente ruta recomendada: una revisión dirigida del soporte espacial
de predicciones extremas, no otra búsqueda general ni adquisición de datos.
Identificar si los modelos aprenden relaciones dependientes de pocos
grupos ricos en Li y si existe un criterio inferible de aplicabilidad.
Examinar los predictores canónicos y proximidad multivariada de los
grupos problemáticos, sin utilizar sus etiquetas para entrenar su
evaluación. Evaluar si una corrección o fallback hacia RF basado
exclusivamente en predicciones/soporte TRAIN podría evitar falsas
concentraciones elevadas conservando detección de altas. No utilizar
IDs/grupos de los TEST como reglas de corrección. No confundir diagnóstico
retrospectivo con selección interna o certificación independiente.
Antes de un nuevo experimento, declarar presupuesto, hipótesis, guardas
sin cambios y protocolo de validación. Si no hay solución viable, elevar
decisión sobre reformulación de fase 5 (partir del RF conservado y no
del nuevo HGB/Ridge). No iniciar fine-tuning ni entrenar/guardar un
candidato definitivo sin resolver este bloqueo. M2 sigue congelado.


## 10. Recuperación aprobada e implementada
El usuario aprobó continuar. Ver [M1_RECUPERACION_ESPACIAL.md](M1_RECUPERACION_ESPACIAL.md):
10 fits RF de perfil fijo, OOF HGB/Ridge archivados, soporte químico TRAIN y
respaldo por desacuerdo. Política primaria fija, dos diagnósticas; mismas
guardas y cohortes. Pendiente ejecución local; 130 pruebas esperadas.
Fase 5 sigue bloqueada hasta verificar resultados, sin nuevos datos ni M2.
