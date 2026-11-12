# Modelo 2 — restauración del entrenamiento completo (12/11/2026)

Esta actualización corrige el reemplazo accidental de la competencia original por un ajuste CatBoost reducido. El Modelo 2 es experimental y no se publica individualmente: el release futuro, si se aprueba, integrará Modelo 1 y Modelo 2 corregido.

## Menú y entrenamiento

La opción **Entrenamiento → 2. Modelo 2 — Competencia de prospectividad espacial** vuelve a ejecutar seis algoritmos, con sus fases de Optuna, validación espacial anidada, checkpoints, gráficos, exportación de métricas y selección del ganador:

1. Random Forest
2. Extra Trees
3. HistGradientBoosting
4. XGBoost
5. CatBoost
6. SVM-RBF

La configuración de optimización conserva los presupuestos por algoritmo (entre 10 y 12 trials) y los cinco pliegues externos anclados a los 706 registros originales. Se genera un nuevo run si el cambio de configuración invalida el anterior; los checkpoints compatibles pueden reanudarse.

## Datos y QA conservados

- Fuente completa: 783 registros = **706 históricos + 77 nuevas muestras GEOROC**. Las 77 nuevas no se excluyen silenciosamente del entrenamiento; su proveniencia sigue marcada como provisional sin validación de terreno.
- La calidad se controla por la cobertura finita conjunta de las seis bandas originales. El filtro mínimo de 95% actúa **solamente sobre TRAIN** y se aplica a los seis algoritmos, incluyendo optimización interna y ajuste final.
- **767 elegibles para ajuste final: 690 históricas + las 77 nuevas**. Los 16 registros históricos con menos del 95% no se eliminan de los datos; los 783 registros originales se mantienen **todos en los folds exteriores de prueba**.
- Se verifican contra el inventario los ID, el orden, los SHA-256 y las fracciones de cobertura de las imágenes originales antes de cualquier entrenamiento.
- No se modifican los 126 predictores de valores digitales de la caché anterior. La reflectancia corregida y el nuevo selector STAC están disponibles para experimentos opt-in, pero no se activan automáticamente porque su impacto predictivo no fue favorable en todas las métricas.

## Evaluación y candidatos

Cada algoritmo conserva la comparación de ROC-AUC, AP, Balanced Accuracy, Brier y demás métricas, con el ranking del ganador basado en ROC-AUC y desempates definidos. Se añade un reporte diagnóstico de priorización entre tres sitios distintos de un mismo fold, medido sobre predicciones OOF de cada algoritmo; este reporte no altera la regla de selección del ganador.

La inferencia del nuevo candidato puede abstenerse ante una entrada de cobertura insuficiente. El release anterior permanece activo, sin modificaciones. La competencia genera un nuevo candidato NO activo y no publica releases.

## Reversión del error de gestión de candidatos

Se conservaron físicamente los cinco artefactos existentes:

- 20/09/2026: training_20260920_195933_m0300 — modelo del release, activo.
- 05/11/2026: training_20261105_223618_m0300 — estado pending restaurado.
- 06/11/2026: training_20261106_135029_m0300 — estado pending restaurado.
- 11/11/2026: training_20261111_170832_m0300 — estado pending restaurado.
- 12/11/2026: training_20261112_112450_m0300 — ajuste reducido experimental conservado, no representa una competencia completa.

Los archivos binarios de los tres candidatos restaurados se verificaron mediante SHA-256 durante el cambio de estado. No se borraron ni reentrenaron.

## Para ejecutar MANUALMENTE desde CMD

    cd /d "D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Scientific-Lab"
    "D:\Librerias\Python312\python.exe" main.py

Seleccionar Entrenamiento y luego opción 2. Esto inicia la competencia completa; la restauración y los tests NO la iniciaron.

## Advertencias científicas

La condición de superficie expuesta no está certificada para todas las imágenes y hay sitios con etiquetas geológicas contradictorias que conservamos por trazabilidad. Los 77 GEOROC tienen estado provisional; no se debe interpretar la evaluación interna como validación independiente ni el score M2 como probabilidad de viabilidad de extracción.
