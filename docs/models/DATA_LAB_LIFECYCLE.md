# Ciclo de datos del laboratorio

5 > 1: EDA descriptivo. M1 recorre secuencialmente el combinado y fuentes
registradas locales, sin adquisición automática; los faltantes se registran y
el recorrido continúa. 5 > 2: preparación determinista y planes experimentales TRAIN;
5 > 3: tests y simulaciones; 5 > 4: salud. Las fuentes de investigación se
guardan en data/raw/model_1/research, fuera del glob productivo georoc/*.csv.
Se registran DOI, identificadores, hashes y plan de particiones. El grupo de
rocas graníticas/riolíticas se descubre una sola vez desde la API de Dataverse,
con un presupuesto de descarga de 300 MB. No se actualiza automáticamente
la versión de una fuente ya registrada.

El EDA candidato utiliza una vista armonizada de análisis, guardada en resultados,
no un dataset de entrenamiento. Audita Li, calidad, litología, faltantes, bandas
de concentración, identidad/ubicación y solapamiento con el combinado congelado.
Los indicadores de solapamiento son conservadores y requieren revisión. Ninguna
muestra candidata queda aprobada para entrenamiento de forma automática. Las
evidencias originales se conservan y los artefactos compatibles se reutilizan.
La preparación conserva la cohorte congelada y ajusta tratamientos aprendidos
solamente dentro de TRAIN. No se entrena ni se promueve ningún modelo.

5 > 3 > 2 compara artefactos OOF ya guardados sin repetir las simulaciones.
La campaña histórica balanceada M1 ejecutó de 21 configuraciones (7 por
familia) en 630 evaluaciones base. Los expertos requieren ajustes adicionales
por régimen, incluidos en el manifiesto. La campaña anterior de 45 configuraciones
permanece disponible por su comando Python explícito, sin volver a ejecutarla.
Las metas 3000/2700/2400/2043 son tamaños equivalentes de cohorte; se aplican
proporcionalmente a cada TRAIN, nunca se recorta TEST. Se conserva cada estrato
fuente/grupo/litología/banda y toda muestra TRAIN >=100 ppm. Los expertos se
combinan por probabilidades estimadas solo con X; si un régimen carece de datos
suficientes usa el predictor global. Ninguna modificación productiva ni M2.
El anexo PDF actual consume evidencia por identidad de modelo y de dataset,
no se reutilizan métricas históricas de una pareja diferente.
El piloto histórico académico OASIS-like usó con RF, recorte Q2.5–Q97.5
antes del control de calidad, KFold aleatorio 5x5 y 20 trials por fold externo.
Con --trials 100 en el módulo experiments.model_1.balanced.academic se iguala
el presupuesto histórico, no la cohorte original. La cohorte propia debe conservar
al menos 2043 filas reales. Se registra la superposición de grupos espaciales y
la limitación del dominio recortado. No cambia la validación espacial principal.
Pendientes: ejecución local, comparación de evidencia, decisión de selección y
entrenamiento final solo si se demuestra una alternativa favorable.
Las decisiones de incorporación de nuevas fuentes se basarán en las tablas de
elegibilidad, no solo en cantidad de muestras.

## Refinamiento M1 del 28/11

Consultar [plan de continuidad](../scientific/continuity/2026-11-26/M1_REFINAMIENTO_FINAL_EDA_DATOS_20261128.md).
El inventario de fuentes separa estado de ejecución y decisión científica.
La retirada exige perfil íntegro y decisión explícita, conserva históricos y
no elimina originales. El nuevo EDA no modifica la preparación canónica.
Validar primero suite y EDA local; no reejecutar tratamientos generales hasta
revisar los resultados y decidir las brechas experimentales pendientes.

## Preparación final M1 aprobada (28/11)

La opción 5 > 2 > 1 consume el checkpoint EDA aprobado y ejecuta la nueva
[preparación final M1](../scientific/continuity/2026-11-26/M1_PREPARACION_FINAL_20261128.md), no los doce brazos genéricos anteriores. M2 permanece sin cambios. Las rutas de evaluación histórica todavía no consumen los contratos nuevos y no deben ejecutarse para este punto de control.
