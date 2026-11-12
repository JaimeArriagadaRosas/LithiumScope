# LithiumScope Modelo 2 — diagnóstico confirmado y protocolo experimental

Fecha: 12 de noviembre de 2026. Análisis ejecutado localmente sin sustituir modelos, alterar datasets ni publicar releases.

## 1. Diagnóstico de sobreajuste

El Random Forest final `training_20261112_123451_m0300` se ajustó a 767 de las 783 muestras (QA95), con 126 características espectrales. Evaluado sobre las mismas 767 observaciones de ajuste: ROC-AUC=0.990865, AP=0.970613, Balanced Accuracy a umbral operativo=0.946852 y Brier=0.070213. Predicciones espaciales OOF archivadas sobre las 783 muestras: ROC-AUC=0.653811, AP=0.384209, BA@0.5=0.598116, Brier=0.208927. Señal fuerte de sobreajuste, no un diagnóstico dominante de subajuste. Las dos cifras proceden de estimadores y definiciones de etiqueta ligeramente diferentes, por tanto la brecha no es un estimador exacto de error de generalización.

## 2. Auditoría de independencia y calidad de datos

Existen dos probables duplicados entre los diez casos de demostración y GEOROC: A1015-22 con 28.191 ppm a ~1.91 m de un registro de entrenamiento y A1015-25 con 14.079 ppm a ~3.79 m. También se observa A1015-21 a ~175 m con diferente Li. Los diez casos están a menos de 50 km de una observación GEOROC. Por tanto, la demostración no es un holdout independiente válido para escoger hiperparámetros o aprobar el release.

Otros riesgos del conjunto: 77 GEOROC provisionales no verificados en terreno, 25 ubicaciones históricas con etiquetas inconsistentes y 16 parches con QA95 insuficiente conservados en test. Se conserva el contrato inmutable de 706 muestras históricas y cinco folds espaciales. Ninguna fila se eliminó ni cambió en el entrenamiento original.

Artefactos: `results/scientific_simulations/m2_generalization_20261112/external_overlap_audit.csv`, `independence_summary.json`, `scientific_release_gate.json` y `experiment_protocol.json`.

## 3. Técnica de mitigación: regularización del RF espectral

Se probó primero la reducción predefinida de 126 a 48 características directas Sentinel-2 (seis bandas por ocho estadísticos espectrales). No mejoró el rendimiento espacial. Se descartó adoptarla como reemplazo directo. El experimento más prometedor fue la regularización de complejidad conservando las 126 características: `max_depth=8`, `min_samples_leaf=8`, 240 árboles, frente al candidato previo con profundidad 16, mínimo tres muestras por hoja y 354 árboles.

Los seis experimentos usaron exactamente las mismas 783 posiciones de test espacial, las mismas etiquetas específicas de fold y QA95 aplicado exclusivamente a train. Resultados OOF:

| Variante | ROC-AUC | AP | Brier | AUC media train |
|---|---:|---:|---:|---:|
| RF 126 original-style | 0.649555 | 0.381100 | 0.203806 | 0.990239 |
| RF 48 misma complejidad | 0.632644 | 0.360743 | 0.206851 | 0.989553 |
| **RF 126 regularizado** | **0.661251** | **0.393586** | 0.214011 | 0.961617 |
| RF 48 regularizado | 0.647619 | 0.378094 | 0.215670 | 0.953408 |
| RF 48 regularizado + buffer 5 km | 0.645500 | 0.376730 | 0.216076 | 0.955181 |
| RF 48 regularizado sin GEOROC en TRAIN | 0.643223 | 0.380411 | 0.216391 | 0.953874 |

Los resultados son exploratorios y las diferencias de 0.01 de AUC/AP no son por sí solas evidencia de superioridad estadística. El piloto no reproduce exactamente la optimización Optuna original por fold: emplea parámetros fijos del modelo final para comparaciones controladas. No se entrenó un nuevo candidato final.

Artefactos: `paired_fold_predictions.csv`, `per_fold_metrics.csv`, `train_only_nested_thresholds.csv` y `pilot_summary.csv`.

## 4. Calibración y umbral sin fuga del test exterior

Para RF regularizado con 126 características se ajustó calibración sigmoide y selección de umbral mediante OOF de tres grupos espaciales internos, exclusivamente dentro de cada outer train. Los umbrales por fold fueron de ~0.230 a ~0.280, nunca calculados con el outer test.

Resultado exterior agrupado: ROC-AUC=0.654500, AP=0.370708, Brier=0.179732. Con umbral entrenado dentro de fold: BA=0.595147. Se redujo claramente el error de calibración (Brier), pero se degradó AP y no se obtuvo una mejor decisión operativa global. Los umbrales son específicos del fold y NO deben copiarse directamente al bundle final.

Artefactos: `nested_full126_predictions.csv`, `nested_full126_train_thresholds.csv` y `nested_full126_summary.json`.

## 5. Selección del ganador y decisión de release

La competencia interna no autoriza promover automáticamente al candidato. Se exige revisar AUC, AP, Brier, Balanced Accuracy, recall, robustez por región, incertidumbre espacial y comparación pareada con el release anterior. Un test externo suficientemente amplio y no contaminado debe quedar fijado antes de la selección confirmatoria. Política inicial conservadora de auditoría: al menos 50 observaciones, al menos diez positivas y sin coincidencias sospechosas a 250 m; son mínimos operativos, no una garantía estadística de potencia.

El release gate local devuelve `approved=false`, con razones: independencia insuficiente, conjunto confirmatorio no congelado y métricas externas pareadas insuficientes. **Modelos activos y candidatos históricos quedan intactos.**

## 6. Fases siguientes y deuda arquitectónica

El protocolo independiente, auditoría de calidad, piloto de seis brazos, umbrales anidados y gate de release están implementados como módulos experimentales nuevos, sin modificar el menú interactivo ni la generación del PDF.

La revisión de código encontró 23 archivos versionados con más de 300 líneas (21 en `src`, dos de pruebas). La refactorización debe preservar interfaces, salidas y validaciones; no se considera completada únicamente por desplazar líneas de un archivo a otro. Se recomienda separar funciones monolíticas `run_model_2_competition` (650 líneas), `run_classification_cv` (378) y la coordinación de reportes, con tests de regresión y experimentos científicos congelados por fase.

**Estado científico:** sobreajuste confirmado; regularización mejora modestamente la discriminación espacial pero requiere validación adicional y ajuste de calibración. Sin aprobación de release.
