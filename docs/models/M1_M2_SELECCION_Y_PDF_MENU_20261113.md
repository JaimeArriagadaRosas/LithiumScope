# LithiumScope - Seleccion cientifica de los modelos y PDF del menu

**Fecha:** 13-11-2026. **Estado:** propuesta previa al release, no activada.

Esta decision se fundamenta en artefactos REALES de entrenamientos. El PDF se genero usando la funcion original del menu Prediccion > 4. Demostracion integrada automatica (no un PDF redactado manualmente).

**Modelos propuestos:** M1 SVM ampliado training_20261106_111653_m0300 y M2 CatBoost release training_20260920_195933_m0300.

## M1: tabla historica de versiones

| Version | Familia | Casos | RMSE global | MAE global | RMSE media folds | Estado |
|---|---|---:|---:|---:|---:|---|
| training_20260925_111245_m0300 | svm | 1120 | 25.68 | 8.13 | N/D | promoted |
| training_20261105_204029_m0300 | svm | 1120 | 25.68 | 8.13 | 20.99 | pending |
| training_20261106_111653_m0300 | svm | 3420 | 159.85 | 16.32 | 87.45 | pending |

### M1: comparacion de 1004 casos fisicos compartidos

| Metrica | Historico | Ampliado | Mejora |
|---|---:|---:|---:|
| RMSE ppm | 26.091 | 21.469 | 17.72% |
| MAE ppm | 8.333 | 7.258 | 12.90% |
| RMSE Li alto ppm | 50.190 | 40.875 | 18.56% |

**Razon de seleccionar M1 ampliado:** reduce RMSE y MAE sobre las mismas observaciones y ayuda en Li alto. PERO 610 de 1004 observaciones cambiaron de fold y el conjunto nuevo contiene maximos de 4160 ppm, por lo que el RMSE total de 159.85 frente a 25.68 del historico NO es directamente comparable.

**Limitaciones:** extrapolacion ante Li extremo, posibles grupos/sitios proximos y falta de holdout externo independiente. La mejora pareada es exploratoria.

## M2: tabla de todas las ejecuciones relevantes

| Version | Casos | Features | Ganador | AUC media folds | AP media folds | Brier media folds | Estado |
|---|---:|---:|---|---:|---:|---:|---|
| training_20260920_195933_m0300 | 706 | 126 | catboost | N/D | N/D | N/D | promoted |
| training_20261105_223618_m0300 | 706 | 126 | random_forest | 0.6121 | 0.3771 | 0.2118 | pending |
| training_20261106_135029_m0300 | 706 | 126 | random_forest | 0.6121 | 0.3771 | 0.2118 | pending |
| training_20261111_170832_m0300 | 783 | 126 | catboost | 0.6225 | 0.3808 | 0.1955 | pending |
| training_20261112_123451_m0300 | 783 | 126 | random_forest | 0.6245 | 0.3740 | 0.2083 | pending |
| training_20261113_114607_m0300 | 783 | 126 | random_forest | 0.6336 | 0.3813 | 0.1977 | pending |
| training_20261113_133905_m0300 | 706 | 126 | random_forest | 0.6162 | 0.3707 | 0.1841 | pending |
| training_20261113_200558_m0300 | 706 | 170 | random_forest | 0.6215 | 0.3906 | 0.1842 | pending |

**Advertencia:** la tabla historica contiene conjuntos y protocolos diferentes; no es prueba pareada.

### Competencia v4: los seis algoritmos

| Algoritmo | AUC folds | AP folds | Brier folds | Mejor de 3 zonas |
|---|---:|---:|---:|---:|
| random_forest | 0.6215 | 0.3906 | 0.1842 | 43.34% |
| extra_trees | 0.6121 | 0.3648 | 0.1864 | 42.54% |
| catboost | 0.5938 | 0.3562 | 0.2126 | 41.49% |
| xgboost | 0.5913 | 0.3713 | 0.2078 | 42.78% |
| hist_gradient_boosting | 0.5764 | 0.3723 | 0.2322 | 38.84% |
| svm_rbf | 0.5526 | 0.3107 | 0.1855 | 39.09% |

### Comparacion M2 en los mismos 706 TEST historicos

| Metrica | Protocolo release CatBoost REENTRENADO | Random Forest v4 | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.6274 | 0.6499 | +0.0224 |
| AP | 0.3700 | 0.3943 | +0.0243 |
| Brier (menos es mejor) | 0.1859 | 0.1862 | +0.0003 |

IC95 espacial agrupado delta AUC: [-0.0176, 0.0540]. Delta AP: [-0.0250, 0.0725]. Ambos incluyen cero; no se demostro superioridad general.

Referencia: se REENTRENO el protocolo del release sobre los mismos 706 casos; no es una evaluacion independiente del archivo ya publicado.

Diagnostico: 516 firmas espectrales distintas entre 706 filas; 190 filas redundantes y 25 firmas con etiquetas mixtas. QA95 elegibles 690 para TRAIN, TEST completo.

**Razon de conservar CatBoost del release M2:** la prediccion existente utiliza sus 126 caracteristicas correctas; se verifico que el predictor actual rellena con CEROS las 44 red-edge faltantes si se carga el bundle v4. Publicar ese bundle sin corregir inferencia produciria puntajes incorrectos. V4 es prometedor, pero carece de superioridad estadistica demostrada y validacion externa certificada.

## PDF del menu (render original ejecutado)

- Run: demonstration_20261113_212844_m0300.
- PDF: results/predictions/demonstration_20261113_212844_m0300/report.pdf (28 paginas).
- ReportLab original: lithiumscope.prediction.reporting.pdf.write_pdf_report, llamado desde el flujo integrado automatico del menu.
- Modelos que efectivamente generaron predicciones: SVM ampliado M1 y CatBoost historico M2.
- 10 casos con 10 predicciones M1, 10 M2, 10 parejas, figuras, mapas, Excel, manifiesto y snapshot de fuentes.
- Se comprobo que el SHA-256 de models/active_models.json fue identico antes y despues del render.

## Decision de release y condicion de publicacion

Se propone preparar un ZIP similar al historico, con dos modelos y metadata, release_manifest y hashes; adjuntar como segundo asset el report.pdf real producido por LithiumScope. NO crear tag, push, activar ni publicar antes de aprobacion expresa del usuario.

## Archivos de evidencia

- results/scientific_simulations/vnext_20261106/stage5_training_comparison/m1_paired_comparison.json
- results/model_2/runs/training_20261113_200558_m0300/tables/old_release_protocol_comparison_706.csv
- results/model_2/runs/training_20261113_200558_m0300/tables/old_release_protocol_intervals_706.csv
- results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/spectral_identity_summary.json
- results/predictions/demonstration_20261113_212844_m0300/model_selection_evidence.json
