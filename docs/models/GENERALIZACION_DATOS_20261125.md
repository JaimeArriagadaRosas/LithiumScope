# Optimización científica de datos — primera implementación

La versión anterior del 25/11 es referencia histórica. No se sobrescriben sus fuentes, modelos ni resultados.

## Preparación

Menú 5 > 1 > 3: EDA y snapshot determinista versionado en data/processed/model_N/prepared_versions.
Contiene matriz sin imputación aprendida, objetivo, grupos, fuentes, elegibilidad QA y auditoría por fila.
Las alertas IQR/MAD, duplicados y conflictos no eliminan automáticamente registros.
Se comprueba identidad de origen/configuración/código y hashes de artefactos.
El cache se reutiliza en el nuevo protocolo experimental, no todavía en la competencia normal.

## Screening espacial experimental

Desde raíz: python -m lithiumscope.experiments.generalization.screen --model both
Por defecto RF en cada modelo; --all habilita las doce familias.
Se recomienda configurar PYTHONPATH=src si no está instalado el paquete.
Por cada fold exterior se elige un tratamiento por tres folds interiores, sin mirar el TEST exterior.
Se compara contra la misma familia con parámetros por defecto, no contra el Optuna histórico.
Se conservan las filas de TEST, las etiquetas por cuantil TRAIN, y QA95 solo para ajuste.
Se guardan resultados, predicciones y procedencia sintética en results/generalization.
Los folds históricos ya fueron inspeccionados: esta es evaluación de desarrollo, no holdout independiente.

M1: referencia, salida no negativa, log1p, deduplicación TRAIN por fuente/grupo,
bootstrap de concentraciones altas, bootstrap espacial, ablación de coordenadas,
filtro de cobertura, regularización inicial (TabNet mediante límite de épocas), e interpolación SMOTER restringida a fuente, coordenadas, grupo, categorías y patrón de faltantes idénticos. Las relaciones derivadas se recalculan.
M2: referencia, regularización, ponderación de clases, over/undersampling,
SMOTE restringido al mismo grupo (interpolación de descriptores, no mediciones satelitales),
PCA48, información mutua K48 y filtro de correlación.
Toda transformación aprendida se ajusta en el TRAIN de cada fit; la referencia M2 usa preprocesamiento común.
Los márgenes SVM se evalúan por ranking, no como probabilidades calibradas.
No se activa ningún modelo, no se ejecuta Optuna y no se altera la competencia canónica.

## Pendientes de la fase

Auditoría documental de unidades y conflictos, nuevas observaciones reales,
validación composicional y evaluación empírica de SMOTER; SMOGN completo con ruido calibrado,
perturbaciones radiométricas físicamente justificadas y evaluación externa independiente.
Integración de tratamientos validados y caché en competencia oficial, ajuste de Optuna,
curvas de aprendizaje y evaluación de calibración/umbral, recuperación de informes M2.
No afirmar mejora predictiva sin resultados locales.
