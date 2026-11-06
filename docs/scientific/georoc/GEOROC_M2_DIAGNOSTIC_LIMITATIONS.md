# Diagnóstico final de limitaciones de M2 con GEOROC (sin modificaciones del modelo)

## Objetivo y decisión
Diagnosticar las limitaciones actuales, no optimizar ni reentrenar. Este informe separa evidencia de desempeño descriptivo, incertidumbre estadística, limitaciones de etiquetas y propuestas para **otra etapa**.

## Datos
El lote ejecutó 498 ubicaciones, con 407 escenas aceptadas y puntuadas, y 91 sin escena aceptada. Las 407 ubicaciones representan 1.010 registros geoquímicos (249 Li alto con umbral 21,6 ppm y 761 bajo); 297 registros originales no recibieron puntuación (22 altos). La unidad satelital es el parche, no la muestra individual. De los 407 sitios, 98 tienen varias muestras y 21 contienen ambas clases de Li.

## Métricas y pruebas de sensibilidad

| Unidad y definición | n | ROC-AUC | AP | BA con umbral 0,5 | BA con umbral histórico 0,08752 |
|---|---:|---:|---:|---:|---:|
| Muestras, sin corregir dependencia | 1.010 | 0,64794 | 0,35836 | 0,50413 | 0,65413 |
| Sitios, mediana de Li por ubicación | 407 | 0,51015 | 0,18505 | 0,51006 | 0,48708 |
| Sitios excluyendo clases mixtas | 386 | 0,50038 | 0,18694 | 0,51331 | 0,46650 |
| Sitios con una sola muestra | 309 | 0,47074 | 0,17922 | 0,50707 | 0,44392 |

Con bootstrap re-muestreando **ubicaciones completas**, 1.200 replicaciones (semilla 42), el intervalo percentil del 95 % para el AUC de muestra es **[0,51848; 0,75735]**; al evaluar un sitio con etiqueta de Li mediana, el intervalo es **[0,41992; 0,59708]**, compatible con AUC de 0,5. Son intervalos condicionales al conjunto existente y no resuelven el sesgo de selección ni la independencia de las fuentes.

Se excluyó cada publicación por turno y se recalculó el AUC ingenuo por muestra: rango **0,58617–0,66903** para las 46 publicaciones con puntuaciones. Esto no corrige la pseudorreplicación espacial ni garantiza independencia por referencia.

## Diagnóstico por limitación
1. **Unidad de evaluación incorrecta**: 1.010 etiquetas de Li se asocian a solo 407 predicciones satelitales. La puntuación se repite entre muestras del mismo lugar; las métricas por muestra sobrerrepresentan localidades con más análisis.
2. **Discrepancia de escala entre muestra y escena**: 21 ubicaciones contienen Li alto y bajo con un score idéntico. La mediana de Li por ubicación es un análisis de sensibilidad, **no una verdad geológica certificada**.
3. **Separación espacial y generalización**: el AUC por ubicación cae a 0,510, sin evidencia convincente de discriminación en este conjunto. El experimento interno previamente corregido daba AUC medio 0,54694; ambos resultados sugieren precaución, pero no deben tratarse como validaciones equivalentes.
4. **Sesgo de disponibilidad de escenas**: las muestras sin puntuación tienen menor prevalencia de Li alto (22/297 = 7,4 %) que las puntuadas (249/1.010 = 24,7 %). Las métricas se refieren solo a imágenes aceptadas.
5. **Umbral decisional**: BA a 0,5 ronda 0,5; la BA más alta a umbral 0,08752 fue optimizada previamente sobre OOF histórico y no demuestra desempeño independiente certificado.
6. **Calidad de referencia externa**: GEOROC es una compilación y sus muestras no han sido aprobadas individualmente como etiquetas geológicas de parches Sentinel-2. La distancia a Mamani09 reduce riesgos obvios de solapamiento, pero no demuestra independencia bibliográfica.
7. **Temporalidad y dominio**: las escenas proceden de 2025, aunque las muestras geoquímicas pueden ser anteriores; la variabilidad de superficie y otras condiciones de adquisición no está controlada.

## Propuestas para mejora futura (NO IMPLEMENTADAS)
- **P1 — Unidad espacial coherente**: revisar definición de etiqueta por parche/afloramiento y políticas explícitas para muestras de clases opuestas en el mismo sitio. Estudiar análisis por sitio y evaluación leave-region-out.
- **P2 — Trazabilidad y elegibilidad de etiquetas**: crear una muestra externa georreferenciada con método específico de Li, coordenadas comprobadas, tipo de material y correspondencia roca-superficie.
- **P3 — Diseño de evaluación**: bloquear antes del experimento los umbrales y las particiones de validación; usar incertidumbre agrupada por ubicación y publicación. Documentar los tamaños efectivos de muestra y prevalencias.
- **P4 — Control de dominio y cobertura**: cuantificar cambios según sensores, fechas, localidades, bandas y nubosidad; separar efectos de escena y geología. Tratar cobertura no aleatoria en los informes.
- **P5 — Desarrollo posterior sujeto a hipótesis**: evaluar si modelos geológicos/regionales o etiquetas agregadas a escala de píxel son más coherentes con la resolución de Sentinel-2. No elegir arquitectura ni reentrenar antes de fijar el problema científico y sus criterios de éxito.

## Conclusión
**El diagnóstico se considera cerrado.** Existe asociación ingenua por muestra (AUC 0,648), pero desaparece al evaluar unidades satelitales independientes (AUC 0,510; intervalo 95 % 0,420–0,597). No se demuestra generalización robusta del modelo actual. El valor de esta etapa es identificar **qué limita el método y qué mejoras merecerían evaluarse después**, no proclamar precisión ni corregir el modelo inmediatamente.

## Datos reproducibles
- `results/scientific_simulations/external_geochemistry_audit/background_georoc_exploratory/diagnostic_grouped_statistics.json`
- `diagnostic_site_level.csv`, `diagnostic_leave_citation_out.csv`, `diagnostic_coverage_bias.csv`
- `src/lithiumscope/experiments/georoc/diagnose_georoc_grouped.py`

Se mantuvieron sin cambios el modelo entrenado, la rama `main` y todos los datos de origen.
