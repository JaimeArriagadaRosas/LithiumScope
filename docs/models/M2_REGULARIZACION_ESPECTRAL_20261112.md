# Modelo 2 — Piloto de reducción del sobreajuste estrictamente espectral

**Fecha:** 12-11-2026. **Estado:** estudio exploratorio ejecutado, sin nuevo release ni reemplazo de modelo.

## 1. Alcance y separación de responsabilidades

El **Modelo 1** concentra la interpretación geoquímica y cuantitativa. El **Modelo 2** recibe parches Sentinel-2 de seis bandas y produce una estimación de candidatura mediante características **exclusivamente espectrales** (126 dimensiones derivadas de la imagen). Las concentraciones químicas originales siguen empleándose únicamente para construir etiquetas supervisadas y evaluar correlación con la respuesta, sin inyectarse como variables de entrada del predictor.

No se mezclaron las salmueras, costras salinas ni suelos adquiridos durante la campaña externa con la matriz espectral. No se descargaron nuevos datos ni se reescribieron fuentes; este estudio aprovecha exactamente los 783 parches existentes y los cinco folds espaciales congelados.

## 2. Técnicas comparadas

Todos los brazos utilizan Random Forest con la **misma configuración fija**, 250 árboles, profundidad máxima 8, mínimo 10 registros por hoja, `max_features=sqrt`, ponderación de clase equilibrada y semilla 42 + número de fold. Sin Optuna ni selección de técnicas a partir del test; el procedimiento en este documento fue predefinido antes de ejecutar las comparaciones.

| Brazo | Tratamiento |
|---|---|
| `control_regularized_126` | 126 características originales, regularización de árboles |
| `gain_augment_train_126` | Control + una variante de ganancia espectral por muestra **únicamente en entrenamiento** |
| `predeclared_direct_48` | 48 características espectrales directas predefinidas |
| `train_only_kbest_48` | Selección univariante KBest de 48 características **ajustada solo sobre el entrenamiento externo** |
| `gain_plus_kbest_48` | Ganancia espectral en entrenamiento + selección KBest en entrenamiento |
| `true_5km_train_embargo` | Control después de quitar del entrenamiento muestras a menos de 5 km de cualquier muestra de test; test intacto |

La aumentación aplica ganancia compartida entre las seis bandas de amplitud ±3 %, más pequeñas variaciones por banda de ±0,7 %, directamente sobre los **DN guardados**, antes de ejecutar el mismo preprocesamiento y extractor de 126 características. Conserva orden de bandas, máscara de píxeles válidos, identificación geográfica, etiqueta y archivo de origen. Cada copia tiene peso 0,5 y cada original peso 0,5: duplicar no crea sitios independientes. **No se afirma que esta transformación simule radiometría física completa:** los DN almacenados pueden requerir escala y offset de escena. Es una prueba de robustez de pequeña amplitud, no un aumento de evidencia externa.

Los folds espaciales anteriores ya se utilizaron para seleccionar políticas del proyecto. Este experimento no es una confirmación independiente; incluso los intervalos bootstrap son exploratorios y no están corregidos por comparaciones múltiples.

## 3. Resultados sobre las mismas 783 observaciones de test

| Brazo | ROC-AUC agrupado | AP agrupada | Brier | Brecha media AUC entrenamiento-test |
|---|---:|---:|---:|---:|
| Control, 126 | 0,65747 | 0,38781 | 0,21646 | 0,31570 |
| Ganancia espectral | 0,64546 | 0,37411 | **0,20980** | 0,35974 |
| Directas, 48 | 0,64776 | 0,38263 | 0,21720 | 0,31306 |
| KBest 48 | 0,65356 | 0,38707 | 0,22257 | **0,29052** |
| Ganancia + KBest 48 | 0,64626 | 0,37789 | 0,21474 | 0,34134 |
| Embargo real 5 km | **0,65879** | 0,37999 | 0,21609 | 0,31953 |

**Incertidumbre agrupada por 143 grupos espaciales** (200 réplicas bootstrap; delta del candidato frente al control):

| Brazo | Delta ROC-AUC (IC95 %) | Delta AP (IC95 %) | Delta Brier (IC95 %) |
|---|---|---|---|
| Ganancia | -0,01200 [-0,03749; +0,00651] | -0,01370 [-0,05625; +0,01144] | -0,00666 [-0,01105; -0,00044] |
| Directas 48 | -0,00971 [-0,06588; +0,03527] | -0,00518 [-0,05033; +0,02744] | +0,00074 [-0,01235; +0,01697] |
| KBest 48 | -0,00391 [-0,02992; +0,02479] | -0,00074 [-0,03291; +0,03313] | +0,00611 [-0,00295; +0,01207] |
| Ganancia + KBest | -0,01121 [-0,05120; +0,01762] | -0,00992 [-0,05149; +0,02079] | -0,00173 [-0,00853; +0,00587] |
| Embargo 5 km | +0,00133 [-0,01844; +0,01696] | -0,00782 [-0,02944; +0,00732] | -0,00037 [-0,00354; +0,00429] |

**Ningún intervalo ROC-AUC o AP excluye cero.** La pequeña mejora en Brier con ganancia no constituye mejora confirmatoria: los folds ya habían sido vistos durante investigación y hay múltiples variantes evaluadas. La aumentación aumentó la brecha aparente de generalización; no es recomendable adoptarla automáticamente. KBest redujo esa brecha sin mejorar de manera concluyente el ROC-AUC/AP y empeoró Brier puntual. El embargo real eliminó 16, 14, 7, 6 y 7 ejemplos de entrenamiento en los cinco folds; no cambió el número de casos evaluados.

El control utiliza **250 árboles con hiperparámetros fijos**, de modo que sus cifras no deben confundirse con las del protocolo canónico de 160 trials Optuna ni afirmarse una superioridad respecto a otros algoritmos sin ensayos pareados equivalentes.

## 4. Reproducibilidad y seguridad experimental

Código nuevo:
- `src/lithiumscope/model_2/methodology/spectral_augmentation.py` — perturbación reproducible de seis bandas; sin mutar parches.
- `src/lithiumscope/model_2/methodology/spectral_overfit_inputs.py` — caché versionada y huellas SHA-256 de **cada parche original**, IDs y orden.
- `src/lithiumscope/model_2/methodology/spectral_overfit_pilot.py` — seis brazos, métricas, embargo real y bootstrap pareado.
- `tests/scientific/test_m2_spectral_overfit.py` — pruebas del esquema, máscaras, determinismo, pesos y embargo.

Salidas experimentales, deliberadamente **sin seguimiento Git**:
`results/scientific_simulations/m2_spectral_overfit_20261112/`. Allí se encuentran `frozen_spectral_study.json`, `train_only_augmentation/`, `selected_features_by_outer_train.json`, `paired_outer_predictions.csv`, `fold_metrics.csv` y `study_summary.json`.

Comando CMD desde la raíz del clon:

    set "PYTHONPATH=%CD%\src"
    "D:\Librerias\Python312\python.exe" -m lithiumscope.model_2.methodology.spectral_overfit_pilot --root "%CD%" --output "%CD%\results\scientific_simulations\m2_spectral_overfit_20261112" --bootstrap 200

La ejecución compara las mismas 783 observaciones de test y no usa imágenes aumentadas en test. La detección de deriva revisa hashes de las fuentes del protocolo, el registro activo y la integridad de la caché de imágenes. **Ningún modelo ni release se activa**.

## 5. Decisión científica y siguiente paso

**No cambiar el Modelo 2 activo ni declarar eliminado el sobreajuste.** La regularización continúa siendo una base razonable, y el piloto sugiere que reducir 126 a 48 atributos puede moderar su brecha aparente sin ganancia concluyente en test. La ganancia aplicada a DN no es una estrategia ganadora en discriminación y no debe quedar habilitada por defecto.

Para avanzar con rigor: aislar regiones/sitios externos verdaderamente nuevos con etiquetas compatibles obtenidas independientemente; revisar la corrección radiométrica de los parches por escena y el control de nubes/terreno; diseñar perturbaciones en reflectancia calibrada cuando tengan justificación física; definir el protocolo y criterios **antes** de examinar el nuevo conjunto externo. Aun así, un resultado estable y modestamente predictivo puede ser la conclusión científica correcta.
