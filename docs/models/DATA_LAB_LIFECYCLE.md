# Ciclo de datos del laboratorio

5 > 1: EDA descriptivo, con adquisición explícitamente autorizada de fuentes M1
registradas. 5 > 2: preparación determinista y planes experimentales TRAIN;
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

5 > 3 > 5 compara artefactos OOF ya guardados sin repetir las simulaciones.
5 > 3 > 3 > 2 ejecuta la campaña balanceada M1 de 21 configuraciones (7 por
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
5 > 3 > 6 ejecuta un piloto académico OASIS-like con RF, recorte Q2.5–Q97.5
antes del control de calidad, KFold aleatorio 5x5 y 20 trials por fold externo.
Con --trials 100 en el módulo experiments.model_1.balanced.academic se iguala
el presupuesto histórico, no la cohorte original. La cohorte propia debe conservar
al menos 2043 filas reales. Se registra la superposición de grupos espaciales y
la limitación del dominio recortado. No cambia la validación espacial principal.
Pendientes: ejecución local, comparación de evidencia, decisión de selección y
entrenamiento final solo si se demuestra una alternativa favorable.
Las decisiones de incorporación de nuevas fuentes se basarán en las tablas de
elegibilidad, no solo en cantidad de muestras.
