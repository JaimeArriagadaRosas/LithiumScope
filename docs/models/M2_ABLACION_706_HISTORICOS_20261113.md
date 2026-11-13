# Control experimental M2: 706 muestras históricas frente a 783

**Fecha:** 13-11-2026. **Estado:** próxima competencia M2 preparada, no iniciada. No implica elegir ni activar automáticamente un release.

## Pregunta experimental

¿Los 77 registros GEOROC provisionales de la ampliación contribuyeron a los retrocesos de clasificación, calibración o generalización? Se compararán las competiciones anteriores con **783 fuentes de datos** contra una nueva competición con las **706 originales**. Ambas conservarán las seis familias, regularización, QA95 train-only y criterios de clasificación. Este contraste es exploratorio sobre los mismos folds de desarrollo; **no demuestra causalidad ni independencia externa**.

## Cambio de ejecución

En `config/model_2.yaml`, dentro de `training.expansion`: **`enabled: false`**. La selección normal del menú **Entrenamiento > Modelo 2** llama a `require_model_2_expanded_manifest`, que con esta configuración se redirige al **manifiesto congelado** `data/processed/model_2/training_manifest.csv` sin descargar escenas ni reconstruir el conjunto ampliado.

El manifiesto se protege mediante SHA-256 **`5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1`**. El sistema exige exactamente **706 registros**. El manifiesto y las imágenes ampliadas de 783 se **conservan**; no se borran ni alteran. El Modelo 1 y el menú (opciones) no cambian.

**Prevalidación real, sin entrenar:**
- 706 muestras cargadas, 126 características espectrales de seis bandas, 66 grupos espaciales.
- QA95: **690/706 elegibles para TRAIN**, **16 excluidas SOLO del ajuste**; todas las 706 evaluables en TEST. Es importante: **no es reproducir idénticamente el entrenamiento del release de 2026-09**, porque aquel no aplicaba las mismas políticas QA/Optuna.
- Los cinco conjuntos TEST conservan **exactamente** los índices/agrupaciones del archivo congelado `config/model_2_frozen_spatial_folds_706.json`, con tamaños 156, 138, 138, 137 y 137.
- Los seis algoritmos y búsqueda de hiperparámetros permanecen vigentes; `calibration_policy: legacy` sigue activa.
- La nueva firma de ejecución cambia por la ruta/hash del manifiesto y por la configuración; la competencia de 783 no se reanuda accidentalmente.
- El Modelo 2 rechaza invocaciones con la ruta del conjunto expandido mientras esta modalidad esté activada.

## Reportes y diagnóstico

El QA externo original de 783 muestras se valida por **prefijo de 706**, incluyendo IDs, orden y hashes de las imágenes. No se reescribe la auditoría original.

Los informes ahora admiten **706 o 783** predicciones OOF verificadas por fila, manteniendo las matrices VP/VN/FP/FN, resultados por regiones/QA/OOD, métricas y ranking de tres zonas. La comparación histórica continúa reconstruyendo el protocolo antiguo sobre 706 sin puntuar el binario del release sobre sus datos de entrenamiento.

El diagnóstico del nuevo run de 706 compara también sus OOF con **las primeras 706 filas** del run inmediatamente anterior de **783** `training_20261113_114607_m0300`, con comprobación estricta de IDs, Li observado, etiquetas, grupo y fold; esto permite examinar qué cambió al retirar las 77 observaciones provisionales. No confundir ese contraste con una validación externa independiente.

**Prueba integral sin entrenar:** se reutilizaron temporalmente las predicciones OOF archivadas del 13/11, recortando exclusivamente la COPIA de prueba a 706 para verificar los cuatro reportes. Todas las tareas terminaron `complete`; el directorio temporal de prueba se eliminó. Esta prueba no representa un modelo reentrenado con 706.

## Ejecución y lectura prevista

La próxima vez, seleccionar **Entrenamiento → 2. Modelo 2**. Se debe ver:
- `Modelo 2: control experimental SOLO 706 históricos`.
- `Características espectrales reutilizadas: 706 muestras`.
- `Calidad científica SOLO TRAIN: 690 / 706`.
- `Muestras: 706`; 66 grupos y cinco folds históricos.
- Al terminar, nuevos resultados en una carpeta `results/model_2/runs/training_...`, con `confusion_matrices_all_algorithms.csv` y `diagnostic_manifest.json`.

Para retornar a 783 después del contraste, cambiar **solo `training.expansion.enabled: true`**, conservando las muestras y las auditorías de ambos experimentos.

**Criterio de interpretación:** comparar, sobre los mismos 706 casos TEST, ΔROC-AUC, ΔAP, ΔBrier, FP/FN y ranking. Si hay mejora, debe evaluarse también el papel del QA95 y Optuna; no se atribuirá automáticamente a la exclusión de GEOROC. No se promoverán releases sin evidencia independiente.
