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

Próximas fases: comparación emparejada release/candidato, anexo PDF dinámico,
balanceo TRAIN con mínimo real por cohorte, expertos por régimen y evaluación
de paridad OASIS. Las decisiones de incorporación de nuevas fuentes se basarán
en las tablas de elegibilidad, no solo en cantidad de muestras.
