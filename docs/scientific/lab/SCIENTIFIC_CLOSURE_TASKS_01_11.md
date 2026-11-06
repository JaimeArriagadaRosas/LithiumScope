# Cierre metodológico 1–11 de LithiumScope (laboratorio aislado)

## Alcance y resultados verificables

Se mantuvo inalterado el modelo de producción y su histórico. El laboratorio completó una revisión de **3.220 candidatos** de GEOROC y una auditoría contra **las 1.788 filas completas de Mamani09**, además de las 706 filas del conjunto M2.

La selección espacial/geológica dejó **1.450 filas para revisión documental**. Después de cotejar también la fuente Mamani09 completa, **1.307** continúan separadas geográficamente al menos 100 km. La distancia **no basta** para certificar independencia.

Se descargaron las escenas del piloto para cuatro coordenadas geoquímicas. Una corresponde a una muestra perforada a 100,1 m y se excluyó de la validación de superficie. Existen pruebas bibliográficas y de laboratorio para algunas publicaciones, pero aún faltan método específico de Li y correspondencia espacial para las muestras restantes. **Muestras externas aprobadas: 0.**

## Validación cruzada alternativa, independiente del artefacto publicado

Se ejecutó otro experimento de **5 folds agrupados** en **66 grupos** y **706 muestras**, utilizando características espectrales que excluyen expresamente `Li_icpms` y `spatial_group` como predictores. Cada fold define el umbral q75 únicamente con el Li de sus muestras de entrenamiento; el umbral de probabilidad se calibra con 3 folds interiores agrupados dentro del entrenamiento externo, no con el conjunto de prueba. Valores faltantes espectrales se imputan a 0 de forma fija, sin medianas estimadas en los folds de prueba.

Modelo fijo del experimento: CatBoost con 240 iteraciones y profundidad 5, sin selección de hiperparámetros. Los cinco folds presentan **0 grupos espaciales solapados**. Las predicciones por muestra, umbrales internos, Li y métricas por fold se archivan en `results/scientific_simulations/leakage_safe_m2/`. Se exporta el hash de los datos y características.

**Promedio ROC-AUC: 0,546935. Promedio Average Precision: 0,306160. Promedio Balanced Accuracy: 0,513007.** Son métricas de validación cruzada interna del experimento corregido; **no son métricas externas**, ni una comparación controlada de precisión con el entrenamiento histórico porque cambiaron los hiperparámetros y la definición de etiquetas es propia de cada fold. La estratificación de la construcción externa utiliza un proxy q75 global para repartir muestras entre folds, pero el etiquetado y entrenamiento de cada fold usan exclusivamente Li del entrenamiento de ese fold. Por ello sigue habiendo una pequeña decisión de diseño basada en la distribución global de Li para formar los folds; se documenta y no se oculta.

## Estado por tarea

| Tarea | Estado |
|---|---|
| 1. Método específico de Li | **Pendiente de confirmar por muestra**, documentación primaria parcial |
| 2. Auditoría bibliográfica | **Automatizada para 3.220 muestras**, referencias vinculadas, revisión de artículos originales parcial |
| 3. Referencias 29268/29484 | **Parcial**: 29268 asociada a tesis de Chollay; 29484 no resuelta y la muestra perforada está descartada como superficie |
| 4. Independencia de Mamani09 | **Tamizaje del corpus completo terminado**, independencia documental aún no probada |
| 5. Representatividad superficial | **Filtro de profundidad/plutónico realizado**, equivalencia afloramiento–píxel pendiente |
| 6. Conjunto externo definitivo | **No aprobable con evidencia actual**; ledger completo y shortlist generados |
| 7. Imágenes externas | **Piloto de 4 imágenes completado**, ampliación requiere muestras aprobadas |
| 8. ROC-AUC/AP/BA externas | **No se calculan**: no existen etiquetas externas aprobadas |
| 9. Cuantil q75 sin fuga de prueba | **Validado en experimento separado**, q75 de entrenamiento por fold |
| 10. Umbral operativo calibrado sin reutilizar prueba | **Validado en experimento separado**, OOF interior por fold |
| 11. Evidencia de particiones originales | **Reconstrucción y hash histórico guardados**, índices originales no archivados; nuevo experimento registra fold por muestra |

## Conclusión de cierre

**Toda la parte computacional verificable del alcance 1–11 quedó ejecutada.** No es metodológicamente correcto marcar las tareas 1–8 como “cerradas científicamente” si no existe evidencia primaria suficiente. Esta conclusión no impide continuar, pero identifica el insumo externo indispensable: procedimientos de ensayo geoquímico por muestra, fuente primaria y verificación geológica del material expuesto.

El resultado no modifica la rama `main`, el modelo de producción ni la entrega académica.

## Auditoría adicional de referencias contra Mamani09
Se contrastaron las **59 referencias** de los candidatos que superaron filtros automáticos con los registros de autor y año del dataset Mamani09 completo. **Una referencia** presenta coincidencia básica de autor-año; las otras 58 no la presentan en ese criterio limitado. Esto **no significa 58 referencias independientes**: el material puede replicarse en artículos distintos, cambiar de identificador o provenir de compilaciones secundarias.
Resultados y script: `results/scientific_simulations/external_geochemistry_audit/citation_mamani_overlap_screen.csv`, `citation_mamani_overlap_screen.json` y `src/lithiumscope/experiments/georoc/citation_mamani_overlap.py`.
