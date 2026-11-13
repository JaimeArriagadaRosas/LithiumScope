# Modelo 2 v4 — Regresión y entrenamiento desde el menú habitual

**Fecha:** 13/11/2026. **Estado:** integración implementada, limpieza v4 ejecutada y regresión verificada. **No se inició ningún entrenamiento nuevo** desde el menú ni se publicó un release.

## 1. Qué hace ahora la opción normal de Entrenamiento → Modelo 2

El programa conserva las opciones existentes del menú, el acceso por `main.py` y el proceso usual que guarda rankings, predicciones OOF, modelos candidatos, diagnósticos y trazabilidad. No se creó ninguna opción adicional. El entrenamiento del **Modelo 2 usa ahora v4**:

- **Seis algoritmos obligatorios:** Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost y SVM-RBF; se preservan los seis Optuna y cinco folds espaciales con tres folds internos.
- **170 características espectrales verificadas:** 126 históricas (sin variación científica, tolerando únicamente redondeos de lectura CSV menores a 5e-13) más 44 derivadas de B05/B06/B07/B8A de **escenas Sentinel-2 reales**. Los parches históricos y su extractor v3 siguen intactos.
- **706 muestras históricas**, por mantener `training.expansion.enabled=false` en `config/model_2.yaml`. Es la misma cohorte del experimento v4 anidado del que se acordó partir. La matriz alternativa de 783 muestras con 77 GEOROC provisionales sigue disponible y auditada, **pero NO se incorporó silenciosamente al entrenamiento principal**. Tampoco se incorporaron los nuevos sitios GEOROC provisionales aún sin validación externa. Esta elección mantiene comparabilidad espacial y evita cambios de fuente no controlados.
- QA95 conjunto de seis bandas anteriores y cuatro nuevas aplicado **solo a TRAIN**: 690/706 elegibles. Se mantienen todas las 706 filas y etiquetas en TEST, con etiquetas de percentil 75 calculadas dentro del TRAIN de cada fold.
- El entrenamiento identifica los resultados como `feature_extractor_version=4`, `spectral_representation=historical_126_plus_real_S2_rededge_44`, incluye los cuatro assets red-edge en los metadatos y **NO activa automáticamente el candidato**. Un candidato v4 exige información de las cuatro bandas nuevas para inferencia; no puede tratarse como un modelo de seis bandas original.
- La firma de ejecución ahora depende de la **matriz completa v4** y de la configuración/código; una ejecución anterior de 126 características no es compatible y no se debe reanudar.

## 2. Limpieza selectiva del entrenamiento v4 anterior

Antes de borrar se verificó que **ningún entrenamiento Python v4 estaba activo**. La lista fue examinada en modo solo lectura y después se ejecutó una limpieza por rutas exactas:

| Ruta de resultados anterior | Archivos eliminados |
|---|---:|
| `comparison_706/` — E1 seis algoritmos | 48 |
| `comparison_783/` — E1 ampliado | 27 |
| `nested_170_706/` — E2 interrumpido, Optuna y checkpoints | 70 |
| `data_ablation_170_new_sites/` — no existía | 0 |
| `candidate_bundle_not_published/` — no existía | 0 |
| **Total** | **145** |

La eliminación queda certificada en `docs/models/M2_V4_STANDALONE_RESET_AUDIT_20261113.json`. El procedimiento de auditoría y limpieza con `--execute` explícito es `scripts/scientific/m2_v4_reset_training_artifacts.py`.

**Conservado sin cambios:** las matrices `cohort_706/training_v4_170.csv` (706 × 170) y `cohort_783/training_v4_170.csv` (783 × 170), sus `build_status.json` y SHA-256, imágenes reales (`patches/`), escenas (`stac_cache/`), QA, nueva adquisición GEOROC (`new_sites/`), registros de entrenamiento históricos ajenos a estas pruebas, fuente de M1, modelos del release anterior y `models/active_models.json`. Se conservó la demostración de ingeniería con el M1 nuevo y el M2 del release, que no es prueba de generalización.

Las métricas antiguas E1/E2 escritas en la bitácora histórica son **observaciones de un experimento descartado para selección**. Los outputs y checkpoints de ese entrenamiento fueron eliminados y no se reutilizarán en el siguiente entrenamiento del menú.

## 3. Regresión ejecutada sin entrenar

- Fuente v4 completa, integridad SHA-256, orden de muestras, clases y grupos verificados.
- 126 características históricas contrastadas contra v4, con redondeos numéricos de CSV únicamente.
- QA95 TRAIN-only y conservación de todas las observaciones TEST verificadas.
- Cinco folds espaciales preparados con las 170 características, sin entrenar un estimador.
- La opción existente `2` del menú fue simulada con la rutina real de despacho y con un sustituto del entrenador que **no hace fit**.
- Validado que el protocolo de seis familias no puede reducirse sin fallo explícito.
- El origen 783 es compatible con v4 si se habilita conscientemente la expansión en la configuración, sin necesidad de crear otras opciones.
- La firma de ejecución emplea la matriz de 170 dimensiones, no la fuente de 126.
- **Pruebas dirigidas: 30 aprobadas. Suite total: 444 aprobadas, 15 advertencias no bloqueantes. Ruff aprobado.**
- Esta revisión no resuelve científicamente todas las causas del sobreajuste; evita fallos de conexión del flujo y cambios inadvertidos en los datos. No se volvió a reproducir el entrenamiento completo, puesto que el usuario controla cuándo iniciarlo.

## 4. Cómo entrenar personalmente desde CMD

En la raíz del clon del disco D:

```cmd
cd /d "D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Scientific-Lab"
"D:\Librerias\Python312\python.exe" main.py
```

Elegir la opción **Entrenamiento** y luego **2. Modelo 2** como siempre. La consola anunciará `Modelo 2 v4: 170 características espectrales reales`, las 706 muestras históricas y los seis algoritmos.

La ejecución que el usuario inicie creará una carpeta NUEVA en `results/model_2/runs/training_...` y conservará sus propios logs y checkpoints si se interrumpe. **No se hizo push, no se activó un release ni se lanzó entrenamiento manual por cuenta del asistente.**
