# Cierre definitivo del intento de validación externa — GEOROC 20576

## Decisión
**NO VIABLE para medir la discriminación externa de M2 con la muestra revisada.** El resultado es una decisión científica sobre el conjunto seleccionado, no un fracaso de ejecución.

## Pruebas de elegibilidad y reducción del universo

| Filtro | Número |
|---|---:|
| GEOROC de la publicación 20576 | 219 |
| Li verificado numéricamente contra suplemento original | 210 |
| Li y coordenadas coinciden con suplemento (±0,01°) | 87 |
| Li coincidente y pasó filtros del estudio original | 24 |
| Cumple Li, coordenadas y filtros del estudio simultáneamente | **9** |
| Separadas 100 km o más del corpus Mamani09 completo entre las nueve | **9** |
| Método de Li, independencia bibliográfica y afloramiento-píxel demostrados | **0** |

Las nueve muestras conservadas tienen valores Li entre **7,48 y 10,30 ppm**, todos por debajo del umbral histórico de Li alto de **21,6 ppm**. En consecuencia, aun si sus datos fueran aceptados, forman un conjunto de **una sola clase** (Li bajo). ROC-AUC y discriminación entre clases no serían estimables; no es correcto elegir artificialmente un nuevo umbral con ellas.

## Control de alcance
No se continuó con imágenes Sentinel-2 para este conjunto porque el control de elegibilidad y la distribución de etiquetas hacen que no sea apto para la evaluación binaria externa propuesta. No se buscaron más fuentes ni se reentrenaron modelos.

## Artefactos
- `results/scientific_simulations/external_geochemistry_audit/georoc_20576_final_candidate_gate.csv` — nueve muestras y motivos.
- `results/scientific_simulations/external_geochemistry_audit/georoc_20576_final_candidate_gate.json` — resumen auditable y SHA256.
- `src/lithiumscope/experiments/georoc_20576_final_gate.py` — implementación reproducible.
- `docs/GEOROC_PRIMARY_SAMPLE_VERIFICATION.md` — concordancia original de 210 muestras.
- `docs/SCIENTIFIC_SIMULATION_FINAL_CLOSURE.md` — cierre del laboratorio científico original.

## Alcance pendiente separado
Para una validación de precisión externa real se requerirían muestras independientes de clases Li alto y Li bajo con método y ubicación superficial demostrados, o redefinir un resultado de evaluación geológica distinto de la clasificación binaria actual. Eso constituiría **otra investigación**, fuera del cierre acotado a la publicación 20576.

**Estado final de esta propuesta:** evaluación de elegibilidad completada; conjunto no apto, métricas de exactitud externa no calculadas. Esto cierra el intento sin extrapolar resultados ni ampliar indefinidamente el alcance.
