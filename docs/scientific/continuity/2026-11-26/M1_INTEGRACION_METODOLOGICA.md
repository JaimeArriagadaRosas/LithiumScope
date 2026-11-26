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
