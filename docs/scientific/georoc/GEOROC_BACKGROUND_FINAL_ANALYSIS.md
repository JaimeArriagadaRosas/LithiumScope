# GEOROC — Informe final de ejecución exploratoria

## Alcance
Se consolidó la ejecución automatizada sobre los datos GEOROC prefiltrados, sin reentrenar M2 y sin considerar aprobadas las etiquetas geoquímicas externas.

## Inventario de salida
- 498 coordenadas únicas recorridas.
- 407 coordenadas con imagen Sentinel-2 aceptada y puntuación.
- 91 coordenadas sin escena aceptada (`NO_ACCEPTED_SCENE`); no confundir con fallos de cómputo.
- 1.307 filas de muestras originales en el subconjunto prefiltrado.
- 1.010 filas de muestras enlazadas a una coordenada puntuada: 249 Li alto y 761 Li bajo; 297 sin puntuación.
- 7.733 evaluaciones de escenarios (407 × 19): 5.291 puntuadas y 2.442 abstenciones esperadas por supresión de bandas.

## Evaluación *exploratoria* por fila de muestra
- ROC-AUC: **0,647943**.
- Average Precision: **0,358364**.
- Balanced Accuracy a 0,5: **0,504127**.
- Balanced Accuracy al umbral histórico 0,087522: **0,654128**.
- Distribución de puntuaciones: media 0,14665; mediana 0,11428; rango 0,02033–0,66842.

**Advertencia científica fundamental:** estos números no son una evaluación externa científicamente validada. Existen 1.010 filas geoquímicas pero solo 407 localizaciones espectrales; 21 ubicaciones contienen registros de ambas clases, que comparten idéntico score. Esta pseudorreplicación y la falta de verificación independiente por muestra impiden interpretar la ROC-AUC anterior como estimación imparcial de generalización. El umbral histórico optimizado sobre predicciones OOF tampoco debe usarse para afirmar desempeño externo certificado.

## Qué significa el resultado
La puntuación espectral contiene cierta señal asociativa descriptiva frente al umbral de Li (21,6 ppm) en el subconjunto GEOROC recopilado. Sin embargo, la clasificación con umbral de 0,5 está cerca del azar y la capacidad predictiva real sigue siendo incierta. No se infieren depósitos, reservas ni concentraciones futuras.

## Archivos
- `background_georoc_exploratory/final_sample_scores.csv`: cruce de todas las muestras y sus puntuaciones, incluyendo las sin escena.
- `background_georoc_exploratory/final_coordinate_scores.csv`: un registro por ubicación puntuable.
- `background_georoc_exploratory/final_exploratory_metrics.json`: métricas, balances, exclusiones y caveats.
- `background_georoc_exploratory/status.json`: finalizó con estado COMPLETE.
- `background_georoc_exploratory/errors.csv`: registro inicialmente vacío; no se reportaron errores de cómputo individuales.
- `src/lithiumscope/experiments/georoc/analyze_georoc_background.py`: análisis reproducible.

## Conclusión y cierre
**Procesamiento y análisis exploratorio terminados.** La validación externa *certificada* permanece no demostrada: requiere aprobación de etiquetas y representatividad superficial, más una evaluación de incertidumbre agrupada por sitio y por fuente. No se amplía la búsqueda bibliográfica ni se inicia un nuevo entrenamiento en esta etapa.
