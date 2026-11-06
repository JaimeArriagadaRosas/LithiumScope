# LithiumScope ? verificaci?n de suficiencia de datos M1 y M2

Fecha: 2026-11-06. Estado: **auditor?a y decisi?n cient?fica; no es un release ni un cambio al flujo de producci?n**.

## Alcance y metodolog?a

Se revisaron GEOROC, Mamani09, los manifiestos de M2, los artefactos de experimentaci?n, el c?digo de selecci?n de coordenadas y 50 ajustes de curvas de aprendizaje M2 (cinco folds espaciales; tres semillas en las fracciones parciales). Sin nuevas descargas, sin modificaci?n de datasets originales y sin entrenamiento definitivo.

C?digos reproducibles: src/lithiumscope/experiments/analysis/verify_vnext_data_adequacy.py y audit_georoc_m2_candidate_coverage.py.
Reportes: results/scientific_simulations/vnext_20261106/stage4_evidence/.

## M1 ? verificaci?n de muestras FEOT y decisi?n sobre nuevos datos

| Indicador | Resultado |
|---|---:|
| Muestras originales A | 1.120 |
| Muestras reconstruidas B con 10 ?xidos y QC 94?102 | 3.420 |
| Incremento neto y porcentual | **+2.300 / +205,36 %** |
| Registros Mamani09 / GEOROC en B | 676 / 2.744 |
| Claves A perdidas al pasar a B | **0** |
| Claves expl?citas adicionales | 2.300 |
| Grupos espaciales A ? B | 100 ? 244 |
| GEOROC B con FeOT convertido / Fe2O3 original | 2.329 / 415 |
| Registros GEOROC B que fallan cierre 94?102 | **0** |
| An?lisis con Li y referencia bibliogr?fica en GEOROC original | 3.221 de 3.221 |

FEOT es el hierro total reportado en forma FeO. Fe2O3T equivalente = FEOT ? 1,111348. No se agregan los valores de FEOT y Fe2O3 entre s?. La conversi?n y las convenciones est?n documentadas por Natural Resources Canada. No obstante, 32 registros con Li informan simult?neamente FeOT y Fe2O3 con relaciones muy diferentes; requiere revisi?n por publicaci?n y m?todo anal?tico. Siete identificadores gen?ricos de muestras GEOROC se repiten en distintos grupos: no confundir identidad solo por nombre.

Las 2.744 muestras GEOROC B tienen Li, coordenadas num?ricas y cierre de ?xidos v?lido. No se certific? todav?a su exactitud anal?tica/espacial ni independencia geol?gica. GEOROC advierte que su precisi?n y metadatos dependen de las publicaciones originales.

### ?Es suficiente para mejorar M1?

| Comparaci?n con los mismos folds espaciales de desarrollo | A | B |
|---|---:|---:|
| RMSE medio (ppm) | 20,7817 | **19,3999** |
| MAE medio (ppm) | 7,7399 | **7,1796** |
| RMSE GEOROC OOF (ppm) | 39,5693 | **36,3443** |
| RMSE litio alto (ppm) | 50,4721 | **46,3225** |

La mejora relativa observada fue 6,65 % en RMSE y 7,24 % en MAE. Para la ganancia RMSE A menos B, el intervalo bootstrap agrupado al 95 % fue [-0,204; 3,202] ppm, que **incluye cero**. La mejora del MAE tiene intervalo exploratorio [0,047; 0,908] ppm. La selecci?n de B y el diagn?stico utilizaron los mismos folds de desarrollo, por lo que NO constituye validaci?n externa independiente.

**Decisi?n M1:** no buscar todav?a m?s fuentes. La ganancia interna B justifica validar anal?tica/procedencia y despu?s evaluar sobre un conjunto independiente. No afirmar a?n superioridad estad?stica general ni activar B en entrenamiento principal; descartar C provisional hasta disponer de validaci?n adecuada.

## M2 ? suficiencia de las 706 muestras

| Indicador | Resultado |
|---|---:|
| Muestras actuales M2 | 706 |
| Grupos espaciales | 66 |
| Grupo m?s grande | 156 (22,1 %) |
| Cinco grupos mayores | 45,2 % |
| Grupos con menos de cinco muestras | 33 |
| Grupos con una muestra | 9 |
| ?ndice de concentraci?n de grupos, 1 / suma(p?) | 13,79 |
| Escenas Sentinel-2 distintas | 64 |

El ?ndice 13,79 es **solo una medida descriptiva de concentraci?n**, no un c?lculo formal del tama?o muestral efectivo. Las observaciones no son geogr?ficamente equivalentes ni necesariamente independientes.

### Curva de aprendizaje espacial: 50 ajustes RF

Mismos conjuntos externos de cada fold; tama?o reducido tomando **grupos completos** solo de entrenamiento; umbral de Li alto calculado con entrenamiento externo completo de cada fold, constante entre las fracciones de ese fold. RF con 200 ?rboles e hiperpar?metros restantes fijados; no son cifras directamente equivalentes a todos los algoritmos del release.

| Fracci?n de grupos | Entrenamiento medio | Grupos medios | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|
| 35 % | 193,5 | 18,8 | 0,5752 | 0,3450 | 0,2219 |
| 60 % | 330,4 | 31,6 | 0,5610 | 0,3443 | 0,2168 |
| 80 % | 446,3 | 42,2 | 0,5651 | 0,3506 | 0,2124 |
| 100 % | 564,8 | 52,8 | 0,5689 | 0,3509 | 0,2093 |

Aumentar muestras de grupos existentes **no produjo una mejora monot?nica en ROC-AUC/AP**; Brier mejor? cerca de 5,7 %. No declarar saturaci?n definitiva: pocos folds, subsampling por grupos, variaci?n regional y un ?nico algoritmo. La prioridad es ampliar geograf?a y calidad de etiquetas, no duplicar im?genes de los mismos lugares.

## Hallazgo M2: GEOROC ya existe, pero est? excluido por la selecci?n de columnas

El archivo combinado actual tiene 2.986 muestras GEOROC con Li y coordenadas num?ricas, y el archivo experimental B contiene 2.744 GEOROC que pasaron QA completo. El lector en src/lithiumscope/model_2/data/sample_source.py escoge la **primera columna existente** entre Longitude (X) y Longitude; lo mismo ocurre con la latitud. Las columnas originales de Mamani09 existen en el CSV pero est?n vac?as para GEOROC. Resultado: actualmente procesa solo **706 Mamani09** y ninguna GEOROC. Su huella SHA coincide con el dataset actual, de modo que no es simplemente una cach? obsoleta.

Para B, un selector por **valor v?lido en cada fila**, con identidad por fila Sample/source_sample y procedencia bibliogr?fica, identifica hasta **3.420** muestras con Li y coordenadas: **2.744 GEOROC y 676 Mamani09**, distribuidas en **244 grupos**. Hay **178 grupos adicionales** respecto del M2 actual.

La etiqueta descriptiva q75 de Li es 21,6 ppm en M2 actual y 21,601 ppm en B: pr?cticamente igual. Aun as?, hay que congelar la definici?n operativa del objetivo antes de comparar generaciones y evitar nuevos criterios basados en el test.

### Precisi?n espacial y viabilidad Sentinel-2

Entre los 2.744 GEOROC de B, 2.566 tienen l?mites de coordenadas m?nimo/m?ximo reportados con amplitud nula, y 178 carecen de ambos pares completos. **Amplitud nula no demuestra precisi?n GPS** ni que la posici?n coincida con una zona de 640 m de ancho: las coordenadas deben verificarse por publicaci?n. Ninguna de las 2.744 muestras tiene a?n una nueva escena Sentinel-2 descargada y validada para este experimento.

Adem?s de corregir la resoluci?n por fila, se requieren comprobaciones de material de muestra, metodolog?a anal?tica, fecha/fuente, localizaci?n, accesibilidad de escenas, calidad espectral, duplicados espaciales e independencia entre folds. Los identificadores GEOROC deben preservar publicaci?n + nombre + coordenadas y no convertirse en ID sint?ticos si existe source_sample.

**Decisi?n M2:** investigar incorporaci?n de GEOROC B existente antes de integrar una base externa nueva; es potencialmente ?til por los 178 grupos nuevos. Esto no significa activar autom?ticamente las muestras ni descargar miles de escenas. La etapa siguiente deber?a ser un piloto **aprobado** con trazabilidad y posiciones verificadas.

## Recomendaciones y l?mites de ejecuci?n

1. M1: verificar hierro total y publicaciones por muestra/lote; confirmar con evaluaci?n independiente si el aumento real (+205 %) mejora significativamente y sin sesgo.
2. M2: corregir primero, si se autoriza, el selector de coordenadas e identidad por fila; auditar precisi?n de GEOROC y preparar un piloto reducido de im?genes Sentinel-2 antes de cualquier construcci?n a gran escala.
3. M2: comparar en test espacial congelado, con reglas de umbral y calidad constantes, y controlar nuevo dominio geoqu?mico / diferencias entre fuentes.
4. Solo si la ampliaci?n interna no cubre los d?ficits, investigar conectores GEOROC adicionales, EarthChem/PetDB y fuentes geol?gicas oficiales. PetDB 2.0 permite filtrado geoespacial y exportaci?n CSV/JSON; su disponibilidad no garantiza muestras compatibles.
5. Mantener todos los experimentos como desarrollo; **no** ejecutar entrenamiento final, **no** instalar nuevos modelos, **no** efectuar push ni agregar nuevas fuentes sin nueva aprobaci?n.

## Fuentes consultadas

- Total FeOT y conversi?n: https://geochem.nrcan.gc.ca/cdogs/content/sui/sui00557_e.htm
- Calidad y metadatos GEOROC: https://georoc.eu/georoc/disclaimer.asp
- Citaci?n y licencia GEOROC: https://georoc.eu/georoc/cite.asp
- Autocorrelaci?n en teledetecci?n: https://mlanthology.org/mlj/2022/karasiak2022mlj-spatial/
- Validaci?n geol?gica: https://www.mdpi.com/2075-163X/16/10/1066
- EarthChem PetDB 2.0: https://earthchem.org/news/petdb2.0
- Sentinel-2 L2A y SCL: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html

**Punto de parada: decisi?n del usuario para avanzar.**
