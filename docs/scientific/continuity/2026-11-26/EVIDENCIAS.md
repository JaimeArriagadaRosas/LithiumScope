# Inventario y restauración de archivos

El ZIP LithiumScope_respaldo_evidencias_2026-11-26.zip contiene 158 originales,
inventario completo SHA-256 y guía. **No está publicado en GitHub**.
El archivo debe conservarse en almacenamiento privado. El siguiente listado
es el mínimo para restaurar comparaciones y auditorías. Las rutas de destino
son relativas a la raíz del proyecto, no rutas de un disco específico.

| Original del ZIP | Destino relativo |
| --- | --- |
| run(2).json | results/model_1/runs/training_20261106_111653_m0300/run.json |
| oof_predictions_svm__identity.csv | results/model_1/runs/training_20261106_111653_m0300/tables/oof_predictions_svm__identity.csv |
| outer_fold_manifest.csv | results/model_1/runs/training_20261106_111653_m0300/tables/outer_fold_manifest.csv |
| fold_metrics_svm__identity.csv | results/model_1/runs/training_20261106_111653_m0300/tables/fold_metrics_svm__identity.csv |
| run(3).json | results/model_2/runs/training_20260920_195933_m0300/run.json |
| oof_predictions_catboost.csv | results/model_2/runs/training_20260920_195933_m0300/tables/oof_predictions_catboost.csv |
| fold_metrics_catboost.csv | results/model_2/runs/training_20260920_195933_m0300/tables/fold_metrics_catboost.csv |
| dataset_manifest.json | results/model_2/runs/training_20260920_195933_m0300/manifests/dataset_manifest.json |
| all_reference_paired_oof.csv | results/model_2/runs/training_20261112_112450_m0300/all_reference_paired_oof.csv |
| per_sample_scl_quality_783.csv | results/scientific_simulations/sentinel_methodology_20261112/local_scl/per_sample_scl_quality_783.csv |

Piloto: manifest(20261126-131226).json, fold_metrics(4).csv,
predictions(6).csv y trials.csv corresponden a
results/generalization/model_1_oasis_like/20261126T123613_24ef3f6c/
(manifest.json, fold_metrics.csv, predictions.csv y trials.csv).
Texto pegado(20261126-131209).txt es su log original.

EDA andino: candidate_view.csv, ingestion_audit.json, lithium_bands.csv,
lithology.csv, missingness.csv, overlap_audit.csv, raw_source_identity.json,
spatial_coverage.csv. Los nombres con (1) corresponden a granitos y riolitas.
Las vistas de EDA son análisis descriptivos; no copias productivas.
Consultar inventario ZIP para los directorios exactos de las tablas.
No recrear manifiestos de ejecución ausentes con datos inventados.

Campaña focalizada: manifest(20261125-232044).json, summary(4).csv,
predictions(4).csv, con evidencias restantes por identificar mediante hashes.
Campaña balanceada: manifest(20261126-123455).json, summary(5).csv,
predictions(5).csv, archived_predictions(1).csv, selection(1).json.
No inferir identidad solo por el sufijo de un adjunto; comprobar contenido
y manifiesto. ZIP incluye también tablas de competiciones, logs, pruebas,
cobertura, versiones preparadas, comparaciones y otros adjuntos anteriores.

## SHA-256 críticos

Ver archivo critical_sha256.txt. Los hashes identifican bytes originales,
no archivos sanitizados ni copias reconstruidas. No se necesitan más
OOF históricos para el objetivo práctico: el scientific_folds.json original
de septiembre M2 sigue ausente/opcional y no se debe fabricar. La referencia
reconstruida es suficiente para el análisis exploratorio, no validación externa.

## Restauración segura

Extraer ZIP en una carpeta privada fuera de Git. Verificar cada hash con
inventario.json. Copiar solo las rutas sugeridas y verificar que no exista
una versión diferente antes de escribir. Registrar los conflictos para
revisión manual. No modificar modelos activos ni metadata del release.
El ZIP contiene adjuntos, no backup completo de todo el disco. No usarlo
para sustituir datasets, modelos o manifiestos que no fueron entregados.

## Restauración automatizada

Desde el repositorio actualizado:
```powershell
python scripts/restore_scientific_evidence.py "RUTA_AL_ZIP"
python scripts/restore_scientific_evidence.py "RUTA_AL_ZIP" --apply
```
Se validan los 14 archivos críticos contra hashes versionados antes de copiar.
Las colisiones se registran como conflicto sin sobrescribir. No se extrae todo
el ZIP ni se modifican modelos. Pendiente de ejecución en el PC del usuario.
