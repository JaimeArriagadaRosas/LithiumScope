# LithiumScope — Cierre de la etapa original de simulaciones científicas

> **Alcance cerrado:** ejecución, auditoría y síntesis de robustez de M1/M2. Se detiene aquí la expansión de investigación sobre GEOROC.

## Inventario verificado

| Campaña M2 | Total | Puntuadas | Abstenciones previstas | Inesperadas |
|---|---:|---:|---:|---:|
| 2023-2025 | 228 | 156 | 72 | 0 |
| 2025 seasonal/shifted | 228 | 156 | 72 | 0 |
| 2024 seasonal | 152 | 104 | 48 | 0 |
| 2024-2025 multiscene | 304 | 208 | 96 | 0 |
| same-scene spatial | 684 | 468 | 216 | 0 |

**M2 total: 1596 escenarios**; 1092 puntuados, 504 abstenciones previstas y 0 fallos inesperados. **M1: 90 escenarios**.

## Robustez espacial en la misma escena

36 parches, cuatro sectores, distancias 400 y 1000 m, cuatro direcciones; 684 evaluaciones M2 en la campaña espacial. Cambios absolutos medios del score basal: 0,081121 a 400 m y 0,093536 a 1000 m, con grandes variaciones locales. La ventana de 640 m hace dependientes algunos desplazamientos. No mide exactitud en litio.

## Robustez temporal y perturbaciones
Las campañas 2023–2025 y multiescena 2024–2025 comprobaron variación de score entre años, estaciones y sensores; los efectos atmosféricos, temporales y del satélite no están aislados causalmente. La sensibilidad pareada detectó efectos relevantes de brillo, ruido y pérdida de píxeles. No existe una relación monótona universal del score con distancia, año o estación.

## Validación cruzada del Modelo 2
El histórico de CatBoost corresponde a un entrenamiento distinto (umbral q75 global, selección de umbral OOF): media de cinco folds ROC-AUC 0,573398; AP 0,308807; BA a umbral 0,5 0,498309. Las particiones se pudieron reconstruir por grupos y tamaños, no se recuperaron los índices originales exactos.
El experimento aislado corregido (CatBoost 240 iteraciones, profundidad 5, sin búsqueda) obtuvo cinco folds agrupados, 66 grupos, cero solapamientos: media ROC-AUC 0,546935; AP 0,306160; BA 0,513007. q75 por entrenamiento de fold y umbral de operación aprendido solo con predicciones OOF internas. No comparar sus cifras como prueba del efecto de una única corrección, ya que cambió la configuración.
**Interpretación:** evidencia de capacidad discriminativa interna débil/modesta, no demostración de predicción fiable de litio ni precisión prospectiva externa.

## Investigaciones separadas y no requeridas para cerrar este alcance
GEOROC: 3.220 candidatos descargados y auditados, 1.450 tras filtros iniciales, 1.307 alejados al menos 100 km del corpus Mamani09 completo. La comprobación de 210 valores contra suplemento primario no certifica representatividad espectral de superficie. **Cero etiquetas geoquímicas externas aprobadas**, por lo que no se calcularon métricas externas. Se conservan sus archivos y documentos como línea de investigación futura, sin impedir el cierre de la batería original.

## Conclusión
**La etapa de simulaciones científicas y análisis de robustez queda cerrada como experimento reproducible, no como validación externa de exactitud**. No se harán más descargas, verificaciones de publicaciones ni campañas adicionales dentro de este alcance. La rama original y los modelos entrenados quedan intactos.

## Reproducibilidad
`final_scope_closure/m2_campaign_inventory.csv` y `closure_manifest.json` incluyen conteos y hashes por origen. Los detalles por experimento siguen en sus carpetas originales bajo `results/scientific_simulations/`.
