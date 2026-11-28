# M1 — preparación y tratamiento final de datos (28/11/2026)

Plan de seis etapas aprobado por el usuario tras la validación local del EDA.
Implementación publicada; ejecución local y evaluación predictiva pendientes.
No se modifica el combinado congelado, ni M2, main o modelos productivos.

## Flujo operativo

`5 > 2 > 1` ejecuta exclusivamente la nueva preparación M1, no la antigua
campaña de doce brazos genéricos. M2 conserva su ruta anterior. La ejecución
no adquiere fuentes, no reejecuta EDA y no entrena predictores.

1. Verifica el checkpoint EDA 28/11 11:31 UTC y su inventario 11:32 UTC.
   Identidad canónica SHA256 94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94.
   Inventario SHA256 cbed2c7bccc8239ea22ef643f1f066b0e928922fbb4a4597fd0dcc859eae561d.
   Comprueba hashes, configuración, perfiles originales, evidencia histórica,
   snapshot determinista y protocolo de folds. Se resuelven rutas a partir de
   los IDs de ejecución y directorios configurados, no rutas absolutas ajenas.
   Si falta el checkpoint canónico, bloquear; no sustituir por otra ejecución.
2. Aplica decisiones de fuente aprobadas, condicionadas a evidencia íntegra:
   Chen, Meixner y Fernández-Turiel retiradas del EDA activo con ficha histórica;
   Andean Arc restringida a recuperación, Granite/Rhyolite a reconciliación,
   Macusani a cuarentena y diagnóstico de identidad. El retiro se registra en
   results/model_1/source_catalog/retired; no borra originales ni perfiles.
   Repetir preparación no duplica el retiro y un reapertura explícita no se
   revierte por esta aprobación. Si una fuente falta o su evidencia cambia,
   informar bloqueo y preservar las demás decisiones.
3. Reconciliar el antiguo inventario GEOROC (2998 y 8234) con los perfiles
   actuales (2801 y 7996), utilizando claves de comparación de muestra, archivo,
   coordenadas y Li. Las coincidencias de claves no certifican identidad ni
   la causa de las diferencias; el Li solo se usa en esta comparación
   retrospectiva y no en la recuperación de mediciones ni en inferencia.
   Revisar seis andinas y 69 félsicas altas según el cribado verificado, con
   soporte químico exclusivamente TRAIN, grupos y publicaciones conocidas
   de evaluación bloqueados como diagnóstico. Métodos y CRS no certificados.
   No admitir filas por ausencia de alertas.
4. Recuperación: buscar correspondencias sin Li por nombre explícito,
   coordenadas nominales y al menos cinco campos químicos concordantes.
   Excluir hierro de la comparación por diferencias de base. Rechazar enlaces
   ambiguos o reutilizados; comprobar las tres particiones Andean verificadas
   y, para exclusiones QC identificadas, originales históricos con cadena de
   hashes de ensamblaje e ingestión disponible. Las coincidencias son
   propuestas, no certificación. Extraer únicamente valores originales
   numéricos con wt% explícito, no sustituciones de censura ni conversiones
   de Fe. No imputar QC ni modificar el objetivo. Registrar referencias de
   pie resueltas, conflictos de mediciones, 109 exclusiones QC y estado de
   revisión. Las 76 sin óxidos no se dan por recuperadas. Macusani se revisa
   por identidad, no se incorpora. Falta de evidencia de hierro/base
   anhidra/método requiere revisión científica posterior.
5. Preparar cuatro contratos aislados en los 5 folds exteriores y 25
   interiores, conservando exactamente la identidad de folds EDA. Esperados
   30 particiones y 120 verificaciones, no ajustes de predictores. El
   control tiene 33 variables; sin categorías 28; ILR sustituye los diez
   óxidos por nueve balances (32 variables, manteniendo derivadas para
   aislar efecto); indicadores de ausencia de química: cantidad variable
   calculada en TRAIN. ILR usa base Helmert ortonormal fija, mediana de
   positivos TRAIN para ausencias y mitad del menor positivo TRAIN para
   ceros, hipótesis experimental (no límite de detección certificado).
   No usar esta representación para QC. Los indicadores solo se crean
   para campos químicos con ausencia variable en TRAIN, no duplicar
   máscara TabNet existente. Se transforma la partición reservada con
   contrato fijo, sin etiquetas reservadas, para comprobar esquema y
   paridad por fila. Se guardan contratos, identidad y hash de matrices
   reproducibles a partir del snapshot, no duplicaciones completas.
6. Exportar informe de preparación, manifiesto SHA256, decisiones, variantes,
   particiones, cobertura por banda/grupo/fuente y diagnóstico de candidatas.
   Un bloqueo de fuente o variante deja estado parcial. Un control o
   protocolo canónico incompatible bloquea preparación. Aprobaciones de
   fuentes no equivalen a aprobación de modelos. Cero filas nuevas
   admitidas, cero ajustes, sin autorización de evaluación automática.

## Evidencias y continuación

Directorio `results/model_1/final_data_preparation/preparation_*/`:
- manifest.json, source_decisions.json, treatment_decisions.json.
- georoc_reconciliation.csv, georoc_membership_differences.csv,
  georoc_review_candidates.csv, candidate_fold_preflight.csv,
  candidate_fold_summary.csv (si hay candidatas).
- recovery_identity_links.csv, publication_recovery_review.csv,
  historical_original_review.csv, oxide_recovery_review.csv,
  qc_recovery_disposition.csv, macusani_identity_review.csv.
- treatments.csv, transform_contracts.json, partitions.json,
  train_concentration_coverage.csv.

Informe de menú en results/eda_verification/eda_verification_*.json.
Ejecutar tras git pull --ff-only: 5 > 3 > 1 (validación software pendiente)
y 5 > 2 > 1 (preparación final). Entregar consola, informe y carpeta de
evidencias; conservar datos originales locales. No ejecutar aún 5 > 3 > 3:
sus rutas históricas general/focalizada no consumen todavía estos contratos.
La evaluación predictiva se adaptará al informe aprobado y a la referencia
congelada, sin repetir campañas descartadas. Variantes de recuperación,
expansión y combinación quedan bloqueadas hasta revisar las mediciones,
metadatos, identidad y resultados de los brazos individuales. La
recuperación espacial 4B no se ejecuta en esta etapa. No hay nueva
promoción, ni se afirma mejora de RMSE o generalización.

## Validación de código

Se incorporan pruebas focalizadas de retiros/reapertura, identidad sin
Li, ortonormalidad e invariancia de ILR, indicadores TRAIN, ablación,
despacho de menú y particiones externas/internas. Las pruebas no han
sido ejecutadas contra el clon local; verificar suite y resultados reales.
Todos los commits de esta intervención incluyen [skip ci].


## Validación local posterior y corrección (28/11, 12:01–12:13 UTC)

Commit probado 76069a3a7f2875aab45d43be21dad60fce48a96f.
Ejecución de suite 20261128T120137Z_93730df54b214f699ead8d9323553997:
1313 casos, 1310 aprobados, cero fallos, cero errores, tres omisiones,
300/300 módulos, 8/8 comprobaciones, cobertura de líneas 57.47%.
Pytest 0, validación integral 10 (incompleta por omisiones), Git limpio.
Criterios con pruebas asociadas aprobadas 161/166; cinco con evidencia
incompleta. Omisiones: dos pruebas de artefactos históricos locales
ausentes y una de la extensión provisional M2 de 783 observaciones.
Datos de consola aportados por el usuario; XML/summary no adjuntados
en este punto de control. No interpretar como cierre científico.

Preparación bloqueada en etapa 1, informe
eda_verification_20261128T121341_88fefb14.json, sin ajustes.
La creación en frío del snapshot M1 importaba prepare_training_data
desde model_1.pipeline después de la reubicación del propietario.
Corregido al módulo canónico model_1.preparation.training_data.
Prueba de regresión de creación real, persistencia, reutilización
y correspondencia de folds con datos sintéticos. Aún pendiente ejecutar
esta nueva prueba en el clon local. No se ejecutaron retiros en la
preparación bloqueada ni se modificó la cohorte.

Consola de validación reorganizada en cinco bloques: resultado general,
integridad, cobertura y criterios, incidencias, evidencias. Agrupa las
omisiones por motivo y distingue pytest de la política de validación.
La tabla completa de responsabilidades, nombres de pruebas omitidas,
tiempos y rutas por comprobación permanece en los informes/logs.
Sin cambios a la política de salida ni reducción de la suite.

Actualizar con git pull --ff-only y ejecutar primero 5 > 2 > 1 para
obtener el resultado real de preparación. Después ejecutar 5 > 3 > 1
para revalidar las correcciones. No repetir EDA ni iniciar 5 > 3 > 3.
