# Fuentes geoquímicas candidatas — revisión antes de importar

Fecha de revisión: 2026-10-30. Este archivo es un inventario de **candidatas**, no evidencia analítica incorporada ni conjunto de validación.

## GEOROC
- Portal: https://georoc.eu/georoc/new-start.asp
- Contenido: geoquímica de rocas, con referencias bibliográficas; la compilación de arcos convergentes puede contener muestras de los Andes.
- Evaluación: requiere descarga/selección de muestras y comprobación de unidades, método, duplicados y solapamiento con entrenamiento de LithiumScope.
- Estado: NO IMPORTADO, NO VALIDADO COMO INDEPENDIENTE.

## SERNAGEOMIN — compilaciones geoquímicas
- Portal oficial de recursos geoquímicos y servicios de mapas publicados por SERNAGEOMIN; existen series de sedimentos y otros materiales.
- Evaluación: comprobar atributo de tipo de muestra y disponibilidad de Li por análisis; valores de sedimento, salmuera y relave no equivalen automáticamente a mediciones de roca total.
- Estado: NO IMPORTADO COMO MUESTRAS DE ROCA.

## Requisitos para incorporar mediciones como test independiente
- Li medido, unidad compatible (ppm o mg/kg de roca), método analítico, fecha/procedencia, coordenadas y referencia.
- Auditoría de coincidencias por sample_id/source_sample, proximidad geográfica y pertenencia a las fuentes usadas para entrenar.
- Mantener las salmueras fuera del test geoquímico de roca; no presentar geometrías de salares como evidencia química.

El esquema de entrada está en `data/external/independent_geochemistry/independent_geochemistry_template.csv`. Nunca inferir concentraciones de Li de scores M2 ni de los nombres de salares.

## Estado tras ampliación temporal (30-10-2026)
- GEOROC y fuentes geoquímicas de SERNAGEOMIN se mantienen como candidatas: aún no se descargó ni verificó un conjunto externo que contenga litio, material roca, método analítico, coordenadas y procedencia suficientes. Los datos de sedimentos y salmueras no se mezclarán con los de roca.
- Condición de incorporación: verificar identificadores repetidos, bibliografía, unidades ppm/mg/kg, proximidad espacial frente al entrenamiento de 706 muestras, y variables requeridas para cada modelo. Registrar por muestra las exclusiones y el motivo. Hasta completar esta auditoría: **cero muestras externas aceptadas**.

## Auditoría de una fuente UNAB adicional (30-10-2026)
- Dataset público DOI `10.60483/UNAB/PLCJLQ`, tabla `Table_S01-Chemical&SrNd_Analyses.tab` (archivo Dataverse 592), analizada directamente desde la API pública. Contiene 23 columnas de muestras con coordenadas y parámetros geoquímicos transpuestos, pero **NO incluye el analito litio Li**. No se puede usar para evaluar exactitud del Modelo 2; estado `EXCLUDED_NO_LITHIUM_LABEL`.
- Evidencia reproducible en `results/scientific_simulations/external_geochemistry_audit/unab_chemistry_review.json`, con URL, SHA256, esquema y motivo de exclusión.
- La API de Mendeley del candidato `2tskc6rcfm` devolvió HTTP 401; archivo no importado. No atribuirle contenido validado.
- GEOROC 302366 conserva el estado `REQUIRES_METHOD_AND_OVERLAP_AUDIT`. Sin método analítico ni prueba de independencia, permanece como candidata y **ninguna muestra es aceptada** para validación científica externa.

## Descarga efectiva GEOROC — Andean Arc (30/10/2026)
- Fuente de datos: GEOROC Compilation: Convergent Margins, DOI https://doi.org/10.25625/PVFZCE, archivos publicados en septiembre de 2026, parte 1 (Dataverse ID 219128), parte 2 (219105) y parte 3 (219094), licencia y atribución según las condiciones del repositorio original.
- **Descarga local completada**: `results/scientific_simulations/external_geochemistry_audit/georoc_andean_raw/2026-09-PVFZCE_ANDEAN_ARC_part{1,2,3}.csv`; tamaños 6.836.260, 9.070.244 y 7.738.033 bytes. Huellas SHA-256 registradas en `georoc_andean_audit.json`. Reproductores en `src/lithiumscope/experiments/download_georoc_andean.py` y `filter_georoc_andean.py`.
- Filtrado transparente: 27.077 registros originales; 3.253 filas con Li numérico, 3.221 de roca total con Li numérico y coordenadas; tras quitar un duplicado exacto de nombre+coordenadas+Li, **3.220 candidatos**. Al medir distancia mínima al manifiesto de las 706 muestras de entrenamiento: **988 candidatos a menos de 50 km**, **2.232 a 50 km o más**.
- Archivos de análisis: `results/scientific_simulations/external_geochemistry_audit/georoc_andean_candidates.csv` y `georoc_andean_audit.json`.
- **Sin validar independencia**: las coordenadas distantes no garantizan fuentes independientes. Falta comprobar las citas y métodos químicos por muestra, unidades y solapamiento bibliográfico, así como obtener parches Sentinel-2 correspondientes antes de probar M2. Ninguna de estas muestras se ha utilizado para entrenamiento o métricas predictivas, y aún hay **0 etiquetas externas aprobadas**.

## Auditoría ampliada de GEOROC y piloto M2 (30/10/2026)
- El conjunto real de 3.220 candidatos GEOROC se examinó por referencia, país, ubicación y distancia a los 706 puntos del manifiesto de entrenamiento. Auditoría en `results/scientific_simulations/external_geochemistry_audit/georoc_provenance_screen.csv` y `provenance_screen_summary.json`.
- **31 filas con repetición del identificador de muestra y coordenadas**; 988 candidatas a menos de 50 km del entrenamiento, 202 entre 50 y 100 km y 2.030 a 100 km o más. Hay 1.710 candidatas cuya ubicación indica Chile; 1.093 superaron el filtro provisional de país, distancia, no duplicación y cita no vacía. Una ubicación lejana o referencia distinta **no certifica independencia bibliográfica**.
- Se descargaron parches Sentinel-2 reales en 2025 de cuatro ubicaciones preseleccionadas: citas [27313], [29268], [29484] y [23382], a 156,659; 306,726; 374,638 y 860,859 km del entrenamiento más cercano. **4/4 aceptados**; 76 simulaciones M2, de ellas 52 puntuadas y 24 abstenciones por banda completa ausente.
- Scores basales para las cuatro ubicaciones: 0,318088; 0,157057; 0,313086 y 0,327652. Las concentraciones de Li recogidas en GEOROC son 50,38; 41,60; 5,50 y 9,18 ppm respectivamente, **sin establecer aún que la medición y el parche sean comparables**.
- Artefactos: `pilot_priority_candidates.csv`, `pilot_target_metadata.csv`, `pilot_sentinel/manifest.csv` y `pilot_sentinel/batch/batch_m2.csv`. Código reproducible: `georoc_provenance_screen.py` y `georoc_imagery_pilot.py`.
- **Prohibición metodológica:** no calcular ROC-AUC, Average Precision ni Balanced Accuracy externa de estas cuatro observaciones. Faltan auditoría de métodos geoquímicos, referencias originales, independencia real de las publicaciones/fuentes utilizadas en entrenamiento, escala espacial y suficiencia muestral. El conteo de etiquetas externas aprobadas permanece en **0**.
