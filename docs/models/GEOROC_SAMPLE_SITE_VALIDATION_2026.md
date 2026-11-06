# LithiumScope — validación de referencias y ubicaciones GEOROC para M2

Fecha: 2026-11-06. Estado: auditoría preliminar; **sin nuevas etiquetas aprobadas para el entrenamiento definitivo**.

## Propósito y límites

Determinar cuáles de las 2.744 muestras GEOROC del protocolo B podrían supervisar imágenes Sentinel-2 en M2. **GEOROC aporta solamente la medición de Li y la ubicación, no variables químicas predictoras**. La etiqueta corresponde a una muestra/parche, nunca a todos los píxeles. Se conservaron las seis bandas y las 126 características espectrales originales, los 706 ejemplos previos y los modelos activos.

Una publicación identificada, una suma de óxidos válida, 13 decimales geográficos o seis bandas Sentinel-2 completas **no prueban** que el punto esté localizado correctamente ni que la roca refleje la superficie del parche.

## Resultados reproducibles

Se recuperó el catálogo de referencias de https://georoc.mpch-mainz.gwdg.de/georoc/reference_list.csv, con **21.372 referencias** y SHA-256 registrada en los resultados. Contiene **83 de los 110 IDs de publicación** de GEOROC B; quedan **27 sin resolver**. Las referencias de **2.178 de 2.744 muestras** están completas en esta versión. No se deben inventar autores o títulos para los IDs recientes. GEOROC advierte sobre las limitaciones de precisión analítica y geográfica: https://georoc.eu/georoc/disclaimer.asp. También exige atribución de GEOROC y las publicaciones originales: https://georoc.eu/georoc/cite.asp.

| Categoría de cribado | Muestras | Interpretación |
|---|---:|---|
| MANUAL_SOURCE_REVIEW | **1.523** | Referencia incompleta, coordenada gruesa, sitio repetido, identidad ambigua o contexto minero |
| PROVISIONAL_SITE_REVIEW | **1.094** | Cumplen criterios preliminares, pero **no hay verificación de campo** |
| EXCLUDE_SUBSURFACE_LABEL | **127** | Evidencia textual explícita de perforación o muestreo subterráneo |

Los indicadores se superponen: **1.047** registros tienen dos o menos decimales al menos en una coordenada; **1.087** comparten una misma coordenada redondeada con cuatro o más registros; **19** tienen identidad en la fuente original potencialmente ambigua. Ninguno es por sí mismo una garantía de exactitud geográfica.

De los 1.094 casos provisionales, **465 corresponden a Chile**, 323 a Perú, 186 a Argentina, 71 a Ecuador, 36 a Colombia y 13 a Bolivia; contienen **992 parejas de coordenadas diferentes**, no necesariamente 992 sitios geológicos distintos.

## Verificación de los seis parches existentes

| Muestra | Evidencia bibliográfica y espacial | Resultado |
|---|---|---|
| CH DB 13XIX [20960] | Ancellin et al. (2017), Chimborazo; coordenadas con dos decimales compartidas por 60 rocas | Revisión manual |
| ET09MB173 [26689] | Se indica expresamente **DRILL CORE SG246, EL TENIENTE MINE**; referencia ausente de catálogo antiguo | **Excluir como etiqueta de superficie** |
| ILL09148 [23126] | Higgins y Morata (2019), intrusión Cortaderas Gabbro; faltan coordenadas de campo verificadas | Revisión de sitio provisional |
| 2021-MAC-025 [30009] | Contexto **FALCHANI LITHIUM PROJECT**, Li 2.920 ppm; fuente bibliográfica reciente no identificada y posición no certificada | Revisión manual |
| G0020315-2 [23382] | Wieser et al. (2019), Don Casimiro-Maipo; muestra descrita en material científico, sin validación de precisión GPS aún | Revisión de sitio provisional |
| QUIS-96H-9 [4065][24044] | Lindsay et al. (2001) y Meixner et al. (2020), pumita/ignimbrita de La Pacana; falta verificar afloramiento/punto | Revisión de sitio provisional |

Los metadatos bibliográficos de **cuatro muestras** se resolvieron completamente. **Tres** pasan los filtros preliminares, pero la cifra de muestras con geolocalización y correspondencia geológica **científicamente certificadas sigue siendo cero**.

La pertenencia al material WHOLE ROCK de GEOROC no evita que un análisis proceda de un testigo de perforación. En ese caso no puede asumirse que el espectro de superficie corresponda a la composición de profundidad.

## Decisión científica

1. **No ampliar todavía el entrenamiento M2.** Priorizar el estudio de los tres sitios provisionales del piloto y revisar sus publicaciones originales, suplementos y mapas de muestreo. Exigir ubicación verificable a escala del parche y material expuesto que pueda observar Sentinel-2.
2. Buscar el catálogo bibliográfico actualizado para los IDs [26689] y [30009]. No confundir una referencia bibliográfica recuperada con validación completa de un punto de muestreo.
3. Mantener excluidas las etiquetas de testigos de perforación y muestreo subterráneo; la revisión debe ser por publicación y sitio, no solamente por número de filas.
4. Una vez verificadas suficientes ubicaciones, seleccionar grupos espaciales independientes, separar datos de evaluación antes de optimizar y **solo entonces** ampliar imágenes y evaluar un entrenamiento experimental con las mismas 126 variables espectrales.
5. **Sin descargas masivas, sin push, sin release y sin cambios en el entrenamiento productivo** en esta etapa.

## Archivos

Código: `src/lithiumscope/experiments/analysis/audit_georoc_source_references.py`.
Pruebas: `tests/scientific/test_georoc_source_references.py`.
Resultados locales: `results/scientific_simulations/vnext_20261106/georoc_source_validation/`.

En resultados: `GEOROC_source_validation_summary.json`, `GEOROC_B_bibliography.csv`, `georoc_B_screened_sites.csv`, `six_pilot_site_reviews.csv` y copia del catálogo para reproducibilidad.

Documentación anterior: `docs/models/GEOROC_M2_LABELLED_SENTINEL_PILOT_2026.md` y `docs/models/INTEGRAL_TRAINING_VNEXT_2026.md`.


Nota adicional: ambos dominios oficiales GEOROC entregaron el mismo catalogo de 5.681.283 bytes, cuyo ID maximo es 26640. Los IDs [26689] y [30009] no estan presentes. No se infieren sus autores ni DOI.
