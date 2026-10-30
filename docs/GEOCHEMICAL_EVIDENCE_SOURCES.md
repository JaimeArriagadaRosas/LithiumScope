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
