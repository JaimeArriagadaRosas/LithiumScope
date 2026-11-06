# LithiumScope — comparación de entrenamientos del 5 y 6 de noviembre de 2026

**Estado:** pruebas y análisis OOF completados, sin reentrenar ni cambiar modelos activos. Rama local feat/scientific-simulation-lab.

## Objeto y cautelas de la comparación

Se compara el **entrenamiento anterior del 5/11** con el **nuevo del 6/11**, no automáticamente con el release activo de septiembre. No existe aquí una evaluación externa congelada común de ese release.

M1: SVM RBF con transformación identidad. Los datasets son distintos: A, 1.120 muestras, y B, 3.420 muestras. Se emparejan registros con igual fuente, Li real y coordenadas redondeadas a cinco decimales, excluyendo identidades ambiguas. Ambos modelos predicen en OOF sin entrenar con su propio grupo de prueba; **610 de 1.004 muestras emparejadas cambiaron de fold**, así que la mejora es exploratoria y no un ensayo con splits idénticos.

M2: los **706 registros, las etiquetas, los cinco folds espaciales y la huella SHA del dataset son idénticos**. Por tanto, la comparación de Random Forest es mucho más estricta. Todos los resultados son de OOF, no de observaciones externas nuevas. Los intervalos de bootstrap remuestrean **grupos geográficos completos** 1.000 veces y siguen siendo exploratorios.

## M1 — mejora en las mismas observaciones

Run anterior: training_20261105_204029_m0300.
Run nuevo: training_20261106_111653_m0300.

Se emparejaron **1.004/1.120** observaciones de forma unívoca; quedaron 116 sin correspondencia inequívoca (duplicados de clave o falta de par).

| Métrica sobre las mismas 1.004 observaciones | Antes | Ahora | Ganancia |
|---|---:|---:|---:|
| RMSE (ppm) | 26,0913 | **21,4690** | **17,72 %** |
| MAE (ppm) | 8,3328 | **7,2576** | **12,90 %** |
| Sesgo medio predicción menos real (ppm) | −1,2284 | −0,7993 | Mejora |
| P90 del error absoluto (ppm) | 15,5661 | 14,1699 | Mejora |

Bootstrap por **99 grupos**: disminución de RMSE 4,622 ppm, IC exploratorio 95 % **[0,436; 10,116]**; disminución MAE 1,075 ppm, IC **[0,289; 2,267]**. En los 394 registros con mismo fold, RMSE 22,55→15,93; en los 610 con fold cambiado, RMSE 28,14→24,39. Se mejoran ambos subgrupos, pero esto no corrige la falta de un mismo protocolo independiente.

| Subconjunto | Muestras | RMSE antes | RMSE ahora | MAE antes | MAE ahora |
|---|---:|---:|---:|---:|---:|
| GEOROC | 349 | 42,69 | **34,41** | 13,88 | **10,91** |
| Mamani09 | 655 | **8,50** | 8,70 | 5,38 | **5,31** |
| Litio alto en cuartil común (≥22,825 ppm) | 251 | 50,19 | **40,87** | 17,13 | **14,22** |

**No comparar directamente RMSE global 25,68 (A) con 159,85 ppm (B):** proceden de datasets distintos; B incluye litio extremo hasta 4.160 ppm y su nuevo dominio sigue siendo difícil. En Mamani09 el RMSE empeoró ligeramente, dato que debe considerarse en la futura promoción.

**Decisión M1:** supera el criterio de mejora **exploratoria** de al menos 80 % de registros emparejados y reducciones de 5 % o más tanto en RMSE como MAE con bootstrap espacial positivo. **No certificado para release independiente**. Conservar el nuevo como candidato prometedor.

## M2 — ninguna mejora de discriminación frente al entrenamiento anterior

Runs: training_20261105_223618_m0300 y training_20261106_135029_m0300.

Las 706 puntuaciones de Random Forest anteriores y actuales coinciden (diferencia máxima **1,67 × 10⁻¹⁶**, ruido de coma flotante).

| Métrica global OOF | RF anterior = RF nuevo | Nuevo RF calibrado |
|---|---:|---:|
| ROC-AUC (más alto mejor) | **0,6394** | 0,6181 |
| Average Precision (más alto mejor) | **0,3868** | 0,3566 |
| Brier (más bajo mejor) | 0,2127 | **0,1817** |
| Balanced Accuracy a umbral fijo 0,5 | **0,5993** | 0,5000 |

Bootstrap por **66 grupos** sobre la diferencia calibrado vs anterior: cambio de ROC-AUC **−0,0213**, IC 95 % [−0,0552; +0,0024]; AP **−0,0303**, IC [−0,0773; +0,0087]; ganancia Brier **+0,0310**, IC **[+0,0090; +0,0459]**. El calibrado sí mejora probabilidades pero reduce la ordenación de ejemplos y, con umbral 0,5, no ofrece mejor Balanced Accuracy que una decisión trivial. No se calculó un umbral retrospectivo optimizado con datos de test, ya que sería optimista.

El CatBoost del entrenamiento anterior en este protocolo (no el release activo) tuvo ROC-AUC OOF global **0,6101** y AP **0,3294**. No confundir ROC-AUC global del RF 0,6394 con ROC-AUC medio entre cinco folds, 0,6121: son agregaciones diferentes.

**Decisión M2:** no supera el umbral orientativo de +0,03 ROC-AUC y +0,02 AP sin empeoramiento importante de Brier. No reemplazar el modelo activo por esta versión con la evidencia actual.

## Próxima mejora acotada que se podría estudiar para M2

1. Conservar los 706 ejemplos y los grupos de validación espacial existentes. **No volver a entrenar exactamente la misma competencia** esperando un mejor resultado.
2. Evaluar por separado discriminación (ROC-AUC y AP), calibración (Brier) y decisión (umbral de Balanced Accuracy), seleccionando calibración y umbrales **solo dentro de los folds de entrenamiento**, nunca a partir del test exterior.
3. Revisar el rendimiento por grupo y la concentración geográfica, especialmente el grupo de 156 observaciones. Comparar las variantes antes de decidir una nueva sesión de entrenamiento.
4. Si aun así no mejora, dejar la adquisición de etiquetas GEOROC e imágenes adicionales como propuesta futura opcional, no como requisito automático.

## Código y evidencias

Herramienta: src/lithiumscope/experiments/analysis/compare_vnext_trained_runs.py

Tests unitarios y de integridad: tests/scientific/test_compare_vnext_trained_runs.py

Para repetir la comparación **sin entrenar**:

    python -m lithiumscope.experiments.analysis.compare_vnext_trained_runs

Reportes y predicciones en results/scientific_simulations/vnext_20261106/stage5_training_comparison/:
- comparison_summary.json
- m1_paired_comparison.json
- m1_shared_records_paired_oof.csv
- m2_paired_comparison.json
- m2_identical_protocol_paired_oof.csv
- m2_paired_fold_metrics.csv

**Protecciones:** 0 entrenamientos, 0 descargas, 0 modelos activados, sin push y sin modificar los datos canónicos.
