# Alcance diagnóstico y evidencia científica de LithiumScope

## Evolución del trabajo

Durante esta etapa se abordaron dos objetivos relacionados, pero diferentes: **diagnosticar el comportamiento actual de los modelos** y **examinar hasta qué punto los resultados permiten sustentar afirmaciones científicas sobre su desempeño**. Para ello se realizaron simulaciones de robustez, revisiones metodológicas de la validación cruzada y comparaciones exploratorias con registros geoquímicos externos de GEOROC.

El diagnóstico buscó describir cómo responde el modelo ante cambios de posición, fecha, características espectrales y conjuntos de referencia. La revisión de evidencia científica buscó determinar qué conclusiones se encuentran respaldadas por esos experimentos y cuáles exigirían controles adicionales sobre procedencia de muestras, correspondencia con la superficie observada e independencia de los datos. Ambos enfoques aportaron información, aunque **no debe confundirse el diagnóstico de errores con la certificación definitiva de precisión geológica**.

## Evidencia reunida y limitaciones observadas

La batería original de simulaciones de M1 y M2 quedó ejecutada y consolidada. Para M2, el experimento de validación cruzada con etiquetas calculadas dentro del entrenamiento de cada partición alcanzó ROC-AUC medio de **0,5469**, AP media **0,3062** y Balanced Accuracy media **0,5130**; se trata de evidencia de validación interna con una configuración experimental distinta de la histórica.

La comparación exploratoria con GEOROC procesó **498 ubicaciones**, de las cuales **407** pudieron puntuarse. Estas se vincularon con **1.010 registros geoquímicos**. Al contar cada registro como observación, el ROC-AUC resultó **0,6479**. Sin embargo, múltiples muestras compartían una sola ubicación y un único puntaje espectral, con **21 ubicaciones que contenían registros de ambas clases de Li**. Al consolidar por ubicación usando la mediana de Li, el ROC-AUC descendió a **0,5102**, con intervalo percentil de bootstrap agrupado al 95 % de **0,4199 a 0,5971**; al excluir ubicaciones mixtas fue **0,5004**.

La diferencia revela una limitación concreta: la unidad puntual del análisis geoquímico y la unidad espacial del parche satelital no siempre son equivalentes. También se identificó cobertura desigual entre registros con y sin imágenes aceptadas, sensibilidad al umbral de clasificación y efectos de perturbaciones espaciales, temporales y espectrales. Estas observaciones permiten orientar mejoras posteriores sin modificar ahora el modelo.

## Qué queda sustentado hasta este momento

**Queda documentado y reproducible** que los modelos fueron sometidos a simulaciones y controles de validación interna, que la evaluación exploratoria de GEOROC se ejecutó y que las cifras de desempeño varían sustancialmente al corregir la unidad de análisis. El resultado por sitio, cercano a ROC-AUC de 0,5, **no demuestra discriminación robusta sobre los datos externos examinados**. Estos hallazgos respaldan un **diagnóstico de limitaciones actuales**, no una declaración de que el modelo esté geológicamente certificado.

**No queda certificado** que las puntuaciones correspondan a concentraciones reales de litio en cada parche, que toda muestra externa sea independiente de las fuentes de entrenamiento, ni que exista una precisión de generalización validada a escala geológica. Una certificación más fuerte requeriría datos independientes, georreferenciados y apropiados para la escala satelital, junto con controles de trazabilidad, sesgos de selección e incertidumbre geográfica.

## Proyección y uso en el informe final

La siguiente etapa puede utilizar los hallazgos para formular propuestas justificadas: definición coherente de etiquetas y escala espacial, tratamiento de muestras contradictorias por ubicación, validación agrupada por sitio y región, selección y trazabilidad de referencias geoquímicas y control de diferencias entre escenas. **Son propuestas, no modificaciones implementadas.** El alcance de esta etapa termina con la caracterización de fortalezas, debilidades y condiciones de aplicación; la eventual corrección o reentrenamiento corresponde a un trabajo posterior.

**Documentos de respaldo:** `SCIENTIFIC_SIMULATION_FINAL_CLOSURE.md`, `GEOROC_BACKGROUND_FINAL_ANALYSIS.md` y `GEOROC_M2_DIAGNOSTIC_LIMITATIONS.md`.
