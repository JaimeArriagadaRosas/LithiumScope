# LithiumScope — aclaraciones y piloto de etiquetas GEOROC para M2

Fecha: 06/11/2026. Rama: feat/scientific-simulation-lab.
**Estado: experimento documentado. Ningún modelo nuevo ha sido aprobado para release.**

## Respuestas a las consultas y precisiones del plan

**¿GEOROC cambiará la función del Modelo 2?** No. El Modelo 1 utiliza variables geoquímicas para estimar concentración de Li. El Modelo 2 sigue usando **exclusivamente características espectrales de Sentinel-2** para calcular una puntuación de prospectividad. De GEOROC se usan el Li medido y las coordenadas para asociar una **etiqueta de referencia a una muestra con un parche satelital**. Los demás elementos químicos no son nuevos predictores de M2.

**¿Se etiqueta cada píxel?** No. Es una etiqueta del **punto geoquímico asociado al parche**, no una segmentación de todos sus píxeles. La medición puntual puede no representar exactamente la superficie capturada. Si la ubicación es aproximada, el par no se considera científicamente certificado.

**¿Aumentará el peso, tiempo o funcionamiento de M2?** Más pares válidos requieren buscar y almacenar escenas, extraer características y ajustar el modelo con más filas. Eso puede aumentar significativamente el tiempo de preparación y entrenamiento. Pero las seis bandas y las **126 características espectrales** son las mismas, y el tamaño del modelo guardado no necesariamente crece proporcionalmente a las muestras. La inferencia sigue siendo imagen → características espectrales → modelo → puntuación. Entrenar con más casos sí puede cambiar los parámetros, el umbral y los resultados.

**¿Más etiquetas garantizan precisión?** No. Hay que verificar Li, comparabilidad geológica, ubicación real, fecha, procedencia y correspondencia con imágenes; las imágenes sin Li externo confiable solo demuestran viabilidad o estabilidad. La etiqueta alta/baja continúa utilizando q75 calculado **solo a partir del entrenamiento de cada fold**. No se puede utilizar el test para definir el umbral.

## Alcance aprobado y protecciones

1. Continuar validación de M1 B, reconstrucción de hierro FeOT y evidencia en nuevas zonas, sin activar B automáticamente en producción.
2. Implementar **herramienta piloto separada y explícita** para M2, sin modificar el lector habitual, el menú, las descargas automáticas, los 706 pares históricos ni los modelos instalados.
3. Seleccionar de GEOROC B muestras WHOLE ROCK con Li, coordenadas, cita, identificador y balance de óxidos 94–102. Priorizar grupos geográficos distintos y una publicación por muestra piloto.
4. Utilizar el adaptador Sentinel-2 ya existente con límites estrictos; descargar y almacenar **únicamente los parches del piloto** dentro de resultados experimentales.
5. Ejecutar tests, controlar las seis bandas y el extractor de 126 características. Comparar puntuaciones de modelos existentes solo **de forma descriptiva** hasta que las etiquetas estén verificadas.
6. Documentar experimentos y commits locales; **sin push, sin release y sin entrenamiento definitivo** sin evidencia independiente común.

## Resultado M2: piloto técnico real

- GEOROC B cuenta con 2.744 muestras potenciales; **1.930 candidatos** están en grupos no presentes en el manifiesto M2 histórico.
- Se escogieron **seis muestras, seis grupos y seis citas distintas**, equilibradas descriptivamente: tres con Li bajo y tres alto respecto al valor histórico 21,6 ppm.
- Se descargaron **seis parches Sentinel-2 reales de seis escenas**, sin fallos. Cada parche pasó el control de las seis bandas.
- Se extrajeron **126 características espectrales por parche**, esquema exactamente compatible con CatBoost del release y Random Forest de noviembre. Ni Li ni columnas químicas se introdujeron en la matriz de predictores.
- Los seis archivos de parche ocupan **148.224 bytes** en total. Este valor no representa el coste de almacenar o procesar miles de nuevos ejemplos.
- La diferencia absoluta media entre las puntuaciones de los dos modelos es **0,243**; ese número solo muestra que responden de manera distinta, **no cuál es más preciso**.
- **Ninguna de las seis etiquetas está validada externamente** para uso de entrenamiento definitivo: los límites mínimo/máximo de GEOROC, incluso iguales entre sí, no garantizan precisión de campo suficiente para parches de 640 m.

Evidencia:
- src/lithiumscope/experiments/analysis/pilot_georoc_m2_labels.py
- tests/scientific/test_pilot_georoc_m2_labels.py
- results/scientific_simulations/vnext_20261106/m2_georoc_labelled_pilot/selected_candidates.csv
- .../m2_georoc_labelled_pilot/pilot_manifest.json
- .../m2_georoc_labelled_pilot/pilot_spectral_features.csv
- .../m2_georoc_labelled_pilot/paired_model_scores_descriptive_only.csv
- .../m2_georoc_labelled_pilot/patches/ (fuera de Git)

## Resultado M1: validación adicional en grupos recuperados

Además de la evaluación previa (A 1.120 vs B 3.420; RMSE medio 20,78 → 19,40 ppm), se evaluó por grupos espaciales una parte de GEOROC que **no existía en A**: **1.207 muestras en 144 grupos nuevos**. M1 A se entrenó con sus 1.120 muestras habituales; M1 B excluyó íntegramente los grupos de prueba en cada uno de cinco folds.

| Métrica en mismas muestras GEOROC nuevas | SVM A | SVM B |
|---|---:|---:|
| MAE, ppm | 10,792 | **9,729** |
| RMSE, ppm | 94,489 | **91,766** |
| Sesgo, ppm | −2,748 | −2,953 |

El MAE mejora alrededor del **9,85 %** y el RMSE aproximadamente el **2,88 %**. En bootstrap por grupos, el intervalo de mejora RMSE A−B al 95 % es [−0,420; +4,978] ppm, **incluye cero**. El intervalo para MAE es [+0,271; +2,172] ppm, pero continúa siendo exploratorio. En dos folds B tuvo peor RMSE que A. Algunos valores extremos elevan considerablemente el RMSE.

**No es un holdout externo independiente:** esas muestras nuevas proceden del mismo GEOROC cuyo protocolo B seleccionamos durante el desarrollo. Hay que comprobar referencias bibliográficas, convenciones FeOT/Fe2O3, exactitud espacial y representatividad. No promover B solo porque aumentó la cantidad de muestras.

Evidencia:
- src/lithiumscope/experiments/analysis/validate_feot_expansion_m1.py
- tests/scientific/test_validate_feot_expansion_m1.py
- results/scientific_simulations/vnext_20261106/stage4_m1_feot_verification/

## Qué falta y cómo se continuará

1. **M1:** auditar por publicación/lote el hierro total convertido y los posibles registros ambiguos; preparar una evaluación independiente para certificar la ganancia de B antes de cambiar el filtrado principal.
2. **M2:** comprobar la precisión real de coordenadas y procedencia por publicación, y que cada punto representa material superficial comparable con el parche Sentinel-2.
3. Ampliar gradualmente el piloto solo con etiquetas verificadas y nuevos grupos espaciales. Antes de descargar centenares/miles de escenas, estimar coste de almacenamiento, fallos y calidad real.
4. Solo con suficientes pares válidos, realizar entrenamiento experimental de M2 con **las mismas 126 variables**, conservar separaciones espaciales y q75 train-only, medir ROC-AUC, AP, Brier y robustez en test común congelado.
5. Comparar cualquier candidato M1/M2 con release histórico mediante evidencia independiente; no presentar cifras descriptivas de seis parches como prueba de mejora.
6. Mantener documentación, tests y commits locales. No ejecutar push, publicar release ni sustituir modelos activos.

Documentación adicional: docs/models/INTEGRAL_TRAINING_VNEXT_2026.md y docs/models/DATA_ADEQUACY_VERIFICATION_2026.md.


## Auditoría adicional de las publicaciones y del hierro de M1

- Los **2.744 registros GEOROC B** se distribuyen entre **110 identificadores de publicaciones**. Las cinco fuentes más numerosas representan el **35,6 %** de sus muestras; esto requiere controlar dependencia y posibles sesgos entre estudios, además de los grupos geográficos.
- **2.329 registros** usan hierro total FeOT convertido a equivalente Fe2O3T y **415** usan Fe2O3 original.
- De los **32 registros originales con Li y ambas expresiones de hierro**, **17** tienen una relación Fe2O3 / (1,111348 × FeOT) fuera del intervalo exploratorio 0,95–1,05. Eso **no prueba un error**, porque FeOT puede ser hierro total y Fe2O3 solo la fracción férrica. Exige revisar el método reportado por cada publicación y evitar sumar ambos valores.
- Se guardaron publication_coverage_QC_B.csv, dual_iron_reporting_review.csv y feot_publication_audit.json dentro de stage4_m1_feot_verification/.
- Las huellas SHA-256 de los datos originales combinados M1, manifiesto M2 y CSV crudo GEOROC siguen idénticas a las registradas antes del piloto.

**Resultado de cierre:** se demostró el funcionamiento técnico del camino experimental de etiquetas e imágenes Sentinel-2 y se obtuvo evidencia adicional favorable para M1 B. **No se certificaron aún las seis ubicaciones del piloto ni las 110 fuentes analíticas de GEOROC**; por ello no se justifica todavía promover un modelo nuevo ni descargar masivamente miles de imágenes.
