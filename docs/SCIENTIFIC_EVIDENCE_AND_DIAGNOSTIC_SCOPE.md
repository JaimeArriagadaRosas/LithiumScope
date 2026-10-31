# Alcance del diagnóstico y evidencia científica de LithiumScope

## Propósito de la etapa

La etapa experimental tuvo como propósito **caracterizar el comportamiento y las limitaciones actuales de LithiumScope**, utilizando simulaciones controladas, verificaciones de validación interna y comparaciones exploratorias con datos geoquímicos externos. Paralelamente, se examinó el alcance de las conclusiones que estos ensayos permiten sostener. Estos dos enfoques se complementan: el diagnóstico identifica condiciones en las que el modelo presenta debilidades; la evaluación de evidencia establece qué interpretación es razonable para cada resultado.

La evaluación se realizó de acuerdo con las funciones previstas para los modelos. En particular, M2 produce una **puntuación espectral asociada a prospectividad** a partir de imágenes Sentinel-2: no efectúa una medición química directa, no cuantifica reservas y no certifica yacimientos. Esas capacidades **no formaron parte de los objetivos de M2**, por lo que su ausencia no representa una prueba fallida ni una limitación descubierta durante los experimentos.

## Ensayos efectuados y resultados observados

Se completaron simulaciones de robustez para M1 y M2, con perturbaciones espaciales, temporales y espectrales. En una evaluación interna experimental de M2 con etiquetas calculadas dentro de cada partición, se obtuvo ROC-AUC medio de **0,5469**, Average Precision media de **0,3062** y Balanced Accuracy media de **0,5130**. Esta configuración experimental es diferente de la utilizada para las métricas históricas y no constituye una comparación causal entre versiones.

Para examinar su comportamiento con información externa, se procesaron **498 ubicaciones** asociadas a GEOROC. Se obtuvieron puntuaciones para **407 ubicaciones**, relacionadas con **1.010 registros geoquímicos**. El ROC-AUC calculado por fila de muestra fue **0,6479**. Al considerar una sola observación por ubicación, con la mediana de Li como referencia de sensibilidad, descendió a **0,5102** (intervalo de bootstrap agrupado al 95 %: **0,4199–0,5971**). Al excluir ubicaciones con clases mixtas, el ROC-AUC fue **0,5004**. Se identificaron **21 ubicaciones con registros tanto de Li alto como bajo**, que comparten una puntuación satelital.

Estos resultados **no deben interpretarse como errores individualizados y confirmados del modelo frente a la realidad geológica**: las referencias de GEOROC presentan diferencias de escala y representatividad respecto de las imágenes. Sí permiten identificar un problema concreto de evaluación y aplicación: las conclusiones de discriminación cambian sustancialmente según la unidad de análisis. También se observaron diferencias de cobertura en escenas disponibles, sensibilidad a la elección del umbral y variabilidad frente a perturbaciones.

## Alcance de la evidencia obtenida

**Se ha demostrado en el alcance experimental realizado** que las simulaciones son ejecutables, que M2 produce resultados ante escenas externas y que sus métricas son sensibles al modo de agrupar muestras y ubicaciones. Se documentó además que, **en el conjunto examinado y bajo la agregación por ubicación**, no se observó discriminación robusta respecto de las etiquetas geoquímicas utilizadas: el intervalo de incertidumbre del ROC-AUC incluye 0,5.

**Queda pendiente para una evaluación más específica dentro de su función predictiva** comprobar su capacidad de generalización con referencias externas mejor vinculadas espacialmente a los parches, independientes y representativas del tipo de superficie evaluada. Ese punto describe el nivel actual de evidencia, **no un incumplimiento de objetivos** ni una obligación de conseguir una precisión perfecta. La procedencia y adecuación de las etiquetas también condicionan cuánto pueden atribuirse los desacuerdos observados al modelo.

**Fuera del alcance funcional, sin constituir fallas:** medir concentraciones químicas directamente desde el satélite, confirmar depósitos minerales, cuantificar recursos o reservas y garantizar resultados geológicos. No son requisitos de la arquitectura ni criterios de éxito de esta etapa.

## Propuestas fundamentadas para una etapa futura

Los hallazgos justifican estudiar una definición coherente de etiqueta por ubicación o afloramiento, criterios explícitos para registros contradictorios dentro de un parche, evaluación agrupada por sitio y región, trazabilidad de muestras externas y controles de diferencias temporales y espectrales. Estas acciones deben presentarse como **propuestas derivadas del diagnóstico**, no como modificaciones ya implementadas.

La etapa actual se cierra con una caracterización de comportamiento, incertidumbres y limitaciones observadas. **No se reentrenaron ni modificaron los modelos como consecuencia de este diagnóstico.**

**Documentos de respaldo:** `SCIENTIFIC_SIMULATION_FINAL_CLOSURE.md`, `GEOROC_BACKGROUND_FINAL_ANALYSIS.md` y `GEOROC_M2_DIAGNOSTIC_LIMITATIONS.md`.
