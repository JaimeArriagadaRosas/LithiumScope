# Optimización científica de datos — primera implementación

La versión anterior del 25/11 es referencia histórica. No se sobrescriben sus fuentes, modelos ni resultados.

## Preparación

Menú 5 > 1 > 3: EDA y snapshot determinista versionado en data/processed/model_N/prepared_versions.
Contiene matriz sin imputación aprendida, objetivo, grupos, fuentes, elegibilidad QA y auditoría por fila.
Las alertas IQR/MAD, duplicados y conflictos no eliminan automáticamente registros.
Se comprueba identidad de origen/configuración/código y hashes de artefactos.
El cache se reutiliza en el nuevo protocolo experimental, no todavía en la competencia normal.

## Screening espacial experimental

Menú 5 > 5 > 3 > 1: piloto de ambos modelos; opción 2 de alcance para las seis familias.
Alternativamente, desde raíz: python -m lithiumscope.experiments.generalization.screen --model both
Por defecto RF en cada modelo; --all habilita las doce familias.
Se recomienda configurar PYTHONPATH=src si no está instalado el paquete.
Por cada fold exterior se elige un tratamiento usando los folds interiores configurados, sin mirar el TEST exterior.
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

## EDA ampliado (5 > 1 > 3)

Además del snapshot canónico, genera un plan verificable de tratamientos para cada TRAIN exterior e interior, con QA, semillas, matrices de resampling, selección de variables M2, hashes de matrices preparadas y procedencia. Se guardan las particiones, estadísticas antes/después y trazas, no las matrices completas de cada combinación. El screening comparte las mismas funciones y particiones; recalcula transformaciones por fit. No se ajusta ningún predictor. Los tratamientos no viables se marcan BLOQUEADO sin sustituirlos por otra técnica. Estado PARCIAL si algún tratamiento no es viable; BLOQUEADO si la referencia no puede prepararse. La selección predictiva sigue pendiente. La expansión anclada M2 no está habilitada en este protocolo; se conserva la cohorte histórica.

Enviar la salida de consola y el informe único `results/eda_verification/eda_verification_*.json` (ruta al final del menú). Consolida hashes, auditoría, particiones y tratamientos de ambos modelos, incluso errores parciales. No iniciar entrenamiento completo.

## Correcciones de auditoría: lotes, firmas y límites de aumentación

- M1: SMOTER limitado a ocho usos por padre (ambos roles) y cuatro por par, con selección equilibrada entre grupos disponibles. Si se agota la capacidad, se generan menos filas. Nuevo brazo rare_group_bootstrap, máximo cuatro copias por padre; KNN de cinco vecinos, escalado robusto e imputación dentro de cada fit, sin utilizar TEST.
- M1: auditoría de lotes originales con índices, rangos y verificación de concentraciones numéricas; no utiliza lote como predictor. Diagnóstico de soporte por intervalos numéricos y categorías observadas exclusivamente en TRAIN. No representa una garantía de extrapolación ni altera el objetivo.
- M2: signature_weighted da masa uno a cada firma original exacta; signature_soft comprime cada firma a hasta dos representantes binarios con pesos iguales a la proporción de clase. No cambia las etiquetas originales ni las filas de TEST. Ambas alternativas son experimentales; no son mediciones nuevas. Se reportan AUC/AP ponderados por firma como métricas complementarias, además del resultado por muestra. El EDA informa conflictos binarios y solapamientos exactos por partición. La ponderación se aplica al ajuste del predictor, no al preprocesamiento común.
- Las verificaciones de preparación pasan de 480 a 580 (M1: 12 x 30; M2: 11 x 20). Se preserva el flujo normal y se requiere validación manual antes de seleccionar tratamientos. Los tratamientos bloqueados por capacidad/QA no se convierten automáticamente en éxitos. Los informes de soporte son descriptivos y no deciden qué tratamiento gana.
