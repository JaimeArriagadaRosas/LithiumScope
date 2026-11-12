# Evaluación de SERNAGEOMIN como fuente complementaria

Referencia recomendada por el profesor: https://sernageomin.maps.arcgis.com/apps/dashboards/c7d2481ce62a4a55b233605f06274209

Tablero público ?Plataforma de Salares del Norte de Chile?, identificador ArcGIS `c7d2481ce62a4a55b233605f06274209`. Se examinó su metadato público y un servicio público de química de aguas y costras salinas relacionado con la plataforma de sistemas salinos: `Potencial_Litio_WFL1` (campos `Li_mgL` en agua y `Li_ppm` en costras).

## Decisión

**Considerado, pero NO se integra al entrenamiento actual.** La química de salmuera/agua (mg/L) y costra salina (ppm) no equivale a la geoquímica de roca total del Modelo 1, ni demuestra una etiqueta de litio para la superficie espectral Sentinel-2 del Modelo 2. La mera coincidencia geográfica no certifica representatividad del píxel ni independencia de la muestra.

Para cualquier posible utilización futura se exigiría verificar licencias, procedencia de capa/estudio, método analítico, tipo de muestra, coordenadas, unidades, periodo de muestreo, duplicados y correspondencia con datos históricos de entrenamiento. En una investigación de contraste se podrían visualizar y comparar distribuciones **sin fusionarlas con M1/M2 ni computar métricas externas no admisibles**.

No se propone Modelo 3, ni otro objetivo, ni ingestión automática, ni nuevos pasos en el menú interactivo. Esta evaluación cumple el análisis superficial de utilidad solicitado para el proyecto de tesis.