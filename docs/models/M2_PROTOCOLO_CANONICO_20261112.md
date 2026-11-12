# Modelo 2 — Protocolo canónico y comparación RF espacial

**Fecha:** 12-11-2026. **Estado:** desarrollo, no constituye aprobación de release. Implementación: `model_2/evaluation/canonical_protocol.py` y `model_2/methodology/canonical_rf_comparison.py`.

## 1. Acuerdos comprobados contra las fuentes reales

El manifiesto expandido contiene 783 observaciones: 706 históricas y 77 GEOROC provisionales. El filtro QA95 permite **767 muestras elegibles para entrenamiento**, pero **no excluye observaciones de los folds externos de prueba**. Las imágenes y muestras originales no se modifican. La representación espectral sigue siendo la de 126 características derivadas de seis bandas Sentinel-2. La política de características `direct48` conserva valor como ablación histórica, pero **no se selecciona automáticamente**: las pruebas comparativas del 12-11-2026 favorecieron 126 características con regularización frente a 48, en ROC-AUC y AP.

El protocolo de desarrollo usa los **cinco folds históricos**. El asignador anclado conserva exactamente los 706 índices de prueba originales y asigna las 77 filas adicionales a folds sin compartir identificadores de grupo espacial entre entrenamiento y prueba. Los umbrales de concentración se derivan del cuartil 75 de las muestras históricas del **entrenamiento de cada fold**, nunca de su test. Resultan aproximadamente 21,45; 21,00; 22,40; 21,40 y 21,90 ppm respectivamente. La etiqueta es binaria y puede variar por fold; **no** debe considerarse un umbral universal ya aprobado.

### Distancias geográficas entre folds

La separación de grupos espaciales es correcta, pero la proximidad física no está totalmente eliminada: **50/783 muestras externas al entrenamiento del fold** tienen al menos un sitio de entrenamiento QA95 a menos de 5 km (12, 18, 7, 6 y 7 muestras por fold). Ninguno de los vecinos más cercanos estaba a menos de 1 km. Esto **no confirma una fuga**, pero limita la interpretación de independencia espacial. Conservar los folds permite una comparación histórica justa; analizar después un embargo de 5 km como **sensibilidad separada**, sin cambiar las 783 observaciones de test ni sustituir silenciosamente los folds congelados.

## 2. Comparación controlada de sobreajuste

Se evalúan únicamente dos perfiles de **Random Forest**, conservando las 126 características, muestras, folds y presupuesto de entrenamiento:

- `legacy`: distribución Optuna original, sin modificar estudios anteriores.
- `regularized_v2`: profundidad 6/8/10/12; al menos 6–16 muestras por hoja; 250–650 árboles; selección de variables `sqrt/log2/0.5`.

Cada perfil recibe **16 trials Optuna en cada uno de los cinco folds externos** y utiliza tres particiones internas `GroupKFold` de regiones **exclusivas del entrenamiento externo**. Se conserva la misma semilla base y un máximo de dos trabajos de CPU por estimador, dentro de la capacidad disponible. El objetivo de Optuna es ROC-AUC interno. El modelo externo se ajusta solo con muestras de entrenamiento QA95 y con los hiperparámetros seleccionados internamente.

Se obtienen predicciones OOF **internas** adicionales para estimar la calibración sigmoidal y seleccionar el umbral que maximiza Balanced Accuracy. Ningún caso del test externo interviene en Optuna, calibración ni selección de umbral. Se puntúan dos variantes explícitas: `raw` y `inner_sigmoid`, evitando elegir una únicamente después de observar el test. Se registran ROC-AUC, AP, Brier, Balanced Accuracy, precision, recall, número de positivos, brecha aparente entrenamiento–test y probabilidades por caso.

Todas las predicciones externas de los cinco folds se comparan de forma pareada; se calcula incertidumbre bootstrap por grupo geográfico (400 repeticiones). El conjunto de diez casos de la demostración del PDF **no se utiliza** ni para seleccionar parámetros, ni para calibrar o comparar oficialmente candidatos.

## 3. Trazabilidad y no interferencia

La ejecución guarda `frozen_protocol.json` con SHA-256 de las fuentes, índices de folds, QA y modelo de referencia; los cinco folds y los perfiles tienen subcarpetas y bases Optuna separadas en:

`results/scientific_simulations/m2_canonical_rf_20261112/`

Los archivos `completed.json` conservan los parámetros elegidos, las etiquetas y las predicciones; permiten reanudar sin rehacer trabajos terminados. Cualquier deriva de la huella de los datos o diferencia en el presupuesto provoca un error en vez de mezclar ejecuciones.

Para ejecutar desde la raíz (Windows CMD):

```cmd
set "PYTHONPATH=%CD%\src"
"D:\Librerias\Python312\python.exe" -m lithiumscope.model_2.methodology.canonical_rf_comparison --root "%CD%" --output "results\scientific_simulations\m2_canonical_rf_20261112" --trials 16
```

## 4. Interpretación y límites

Esta comparación es **exploratoria**: los mismos folds ya se utilizaron para desarrollar y revisar la regularización, y no constituyen validación externa nueva, aun cuando los hyperparámetros se ajusten en folds internos. Las comparaciones internas pueden orientar una futura competencia de seis algoritmos, pero **no demostrar superioridad para release**.

Las diez muestras demostrativas tampoco son independientes: A1015-22 y A1015-25 son posibles duplicados de GEOROC a pocos metros, pendientes de verificar documentalmente. Para la fase confirmatoria será necesario un conjunto predefinido de sitios independientes, procedencia revisada, diversidad geográfica y suficientes positivos y negativos, comparado contra el CatBoost anterior.

**Acciones prohibidas en este ciclo:** activar o promover modelos, modificar `models/active_models.json`, alterar el menú interactivo, transformar el PDF, reescribir los 783 registros o usar casos externos para elegir un ganador.


## 5. Resultado ejecutado del protocolo canónico

Se completaron cinco folds espaciales, dos perfiles Random Forest y **16 trials Optuna por perfil y fold**, es decir, **160 trials en total**, con tres folds geográficos interiores por ensayo. Los artefactos se encuentran en `results/scientific_simulations/m2_canonical_rf_20261112/`, junto con `comparison_summary.json`, `fold_metrics.csv`, `paired_outer_predictions.csv` y los diez estudios Optuna aislados.

| Métrica OOF agregada | `legacy` | `regularized_v2` | Diferencia regularizado–histórico |
|---|---:|---:|---:|
| ROC-AUC sin calibrar (↑) | 0,637136 | 0,642447 | +0,005311 |
| Average Precision sin calibrar (↑) | 0,373072 | 0,371298 | −0,001774 |
| Brier sin calibrar (↓) | 0,214789 | 0,220127 | +0,005338 |
| ROC-AUC calibrado (↑) | 0,628196 | 0,633176 | +0,004980 |
| AP calibrado (↑) | 0,354376 | 0,350985 | −0,003391 |
| Brier calibrado (↓) | 0,180453 | 0,180482 | +0,000029 |

El bootstrap pareado por 143 grupos espaciales, 400 repeticiones, obtuvo para las puntuaciones sin calibrar:

- Δ ROC-AUC: +0,00531, intervalo 95 % **[−0,00936; +0,02485]**.
- Δ AP: −0,00177, intervalo 95 % **[−0,02261; +0,02473]**.
- Δ Brier: +0,00534, intervalo 95 % **[−0,00104; +0,01075]**.

**Todos incluyen cero.** En calibración sigmoidal interna ocurre lo mismo. La diferencia no permite afirmar superioridad científica. El ROC-AUC aparente medio de entrenamiento disminuye de **0,97758** a **0,95876**, y la brecha media entrenamiento–fold externo de **0,36452 a 0,33670**. Es decir, la regularización reduce el sobreajuste aparente, **pero no lo resuelve y no trae una mejora externa concluyente**.

### Sensibilidad a cercanía de 5 km

Se ejecutó `spatial_distance_sensitivity.py` sobre exactamente las mismas predicciones OOF, sin crear otro modelo ni modificar la población del experimento principal. `distance_sensitivity.json` y `outer_case_train_proximity.csv` conservan las distancias auditables.

| Subconjunto de test | Casos | Positivos | AUC histórico / regularizado, puntuación sin calibrar |
|---|---:|---:|---|
| Todos | 783 | 195 | 0,6371 / 0,6424 |
| Distancia al train ≥5 km | 733 | 174 | 0,6208 / 0,6278 |
| Distancia al train <5 km | 50 | 21 | 0,7660 / 0,7512 |

Los ejemplos cercanos muestran mejor discriminación que los lejanos, pero la muestra pequeña y su distribución de etiquetas impiden atribuir causalidad a la cercanía. Esto **no sustituye una validación espacial con embargo real**: se excluyeron observaciones cercanas únicamente para este diagnóstico de la tabla, no del conjunto de entrenamiento de los modelos.

### Decisión de esta fase

**No seleccionar `regularized_v2` como ganador comprobado, no crear release, no promocionar candidatos, no activar modelos.** Mantener ambos perfiles disponibles y el registro activo inalterado. No iniciar inmediatamente una competencia costosa de los seis algoritmos solo por la ligera diferencia de ROC-AUC. El siguiente paso de investigación debe verificar con mayor rigor procedencia e independencia geográfica, evaluar embargo de 5 km real, y considerar nuevas muestras verificables de regiones infrarepresentadas. La calibración ayuda al Brier en ambos perfiles, pero no es suficiente para mejorar su discriminación.

El perfil `regularized_v2` permanece configurado en el YAML como **opción experimental para nuevas ejecuciones**, no como evidencia de un ganador. Cualquier competencia futura deberá declarar explícitamente el perfil utilizado y no comparar resultados de distintas configuraciones como si fueran idénticos. El criterio de aprobación del release continúa bloqueado por falta de un conjunto externo independiente.

## 6. Validación de software y reproducibilidad

La suite completa de regresión terminó con **360 pruebas aprobadas, 0 fallos y 15 advertencias no bloqueantes**. Las ocho pruebas nuevas incluyen contratos de cobertura de test, rechazo de solapamiento, distancia geográfica, y un control donde modificar las etiquetas externas no altera las predicciones (dentro de tolerancias numéricas). Ruff y el control de máximo 300 líneas pasan. Ninguna muestra, modelo anterior, menú o PDF fue reescrito.

El protocolo se congeló en `frozen_protocol.json` con SHA-256 de datos, folds, calidad, bundle de referencia y registro de modelos activos. Todas las salidas son **resultados experimentales de desarrollo, no pruebas externas confirmatorias**.
