# LithiumScope — cierre experimental del Modelo 2 (12/11/2026)

**Estado:** cerrado el ciclo de investigación y entrenamiento **experimental**; sin release, promoción ni sustitución del Modelo 2 activo. Un eventual lanzamiento conjunto Modelo 1 + Modelo 2 requerirá decisión posterior.

**Entrenamiento auditado:** training_20261112_123451_m0300
**Log de sesión:** logs/training/session_20261112T153420_370894Z
**Modelo ganador (no activo):** models/model_2/trained/training_20261112_123451_m0300/model.joblib
**Manifiesto original:** data/processed/model_2/vnext_expanded/training_manifest.csv
**Tablas:** results/model_2/runs/training_20261112_123451_m0300/tables
**Entrenamiento anterior:** training_20261111_170832_m0300
**Protocolo del release de septiembre reconstruido:** training_20261112_112450_m0300

## 1. Dictamen del cierre

- **Sí cambió el ganador.** El 11/11 ganó CatBoost; el 12/11 ganó **Random Forest**, que obtuvo el ROC-AUC medio más alto de la nueva competencia (0,624514) y la mejor selección de tres zonas entre los seis competidores (45,659 %).
- **Sí hubo mejoras descriptivas en métricas concretas.** Sobre los mismos 783 registros y folds, el nuevo Random Forest produjo AUC global 0,653811 frente a 0,624751 del CatBoost anterior; AP 0,384209 frente a 0,359586; Balanced Accuracy a 0,5 de 0,598116 frente a 0,538619. La priorización de tres zonas mejoró de 43,971 % a 45,659 % sobre los mismos 1.244 sorteos.
- **No hubo mejora uniforme ni concluyente.** Brier empeoró de 0,195245 a 0,208927. La AP media entre folds del nuevo ganador (0,374009) es menor que la del ganador anterior (0,380819). Los intervalos bootstrap espaciales de AUC y AP incluyen cero.
- **Sí mejoró el método.** Se mantuvieron los seis algoritmos, Optuna, checkpoints y datos completos; se añadió el filtro QA95 SOLO a TRAIN y se conservaron 783 observaciones para evaluación. Hay comparaciones pareadas, métricas estratificadas, tres zonas y reportes reconstruibles sin entrenar de nuevo.
- **Decisión:** conservar Random Forest como candidato experimental pendiente. No proclamar ganancia universal ni autorización de release. Este ciclo se considera cerrado sin entrenamientos adicionales automáticos.

## 2. Integridad del entrenamiento completo

Se preservaron **783 muestras: 706 históricas y 77 GEOROC provisionales**, sin alterar manifiestos, etiquetas, imágenes o caché de 126 características. El filtro de cobertura conjunta de seis bandas >=95 % dejó **767 elegibles para entrenamiento final**; las otras 16 sí participaron en las predicciones OOF de test. Cinco folds espaciales anclados, 143 grupos espaciales y tres folds internos para Optuna. Terminaron los **seis competidores** sin fallos y con sus métricas individuales, OOF, gráficos, ranking, Excel, dashboard, logs y checkpoints.

La regla para elegir ganador NO cambió: ROC-AUC medio por fold, con AP y Balanced Accuracy como desempates. El acierto de tres zonas se informa como métrica adicional y no decide al ganador.

## 3. Ranking de seis algoritmos — media por fold

| Lugar | Algoritmo | ROC-AUC ↑ | AP ↑ | Balanced Accuracy ↑ | Brier ↓ | Top-1 entre 3 ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Random Forest** | **0,624514** | **0,374009** | **0,578430** | 0,208252 | **45,659 %** |
| 2 | Extra Trees | 0,608866 | 0,362449 | 0,575905 | 0,206198 | 44,695 % |
| 3 | CatBoost | 0,599271 | 0,367696 | 0,545461 | 0,190696 | 44,775 % |
| 4 | SVM-RBF | 0,594458 | 0,364770 | 0,501887 | **0,182130** | 43,489 % |
| 5 | XGBoost | 0,594311 | 0,354517 | 0,542621 | 0,205119 | 44,132 % |
| 6 | HistGradientBoosting | 0,574907 | 0,348092 | 0,538360 | 0,211921 | 41,720 % |

Baseline por prevalencia: AUC 0,5, AP 0,248573, Brier 0,186713. El Brier más bajo corresponde a SVM-RBF, no al vencedor. El RF gana en discriminación, no calibración probabilística.

### Comparación contra el ganador CatBoost del 11/11

| Resultado | CatBoost anterior | Random Forest actual | Cambio |
|---|---:|---:|---:|
| ROC-AUC **medio de folds** | 0,622455 | 0,624514 | +0,002059 |
| AP **media de folds** | **0,380819** | 0,374009 | −0,006810 |
| Balanced Accuracy media por folds | 0,537469 | **0,578430** | +0,040961 |
| ROC-AUC **global OOF** (mismos 783) | 0,624751 | **0,653811** | +0,029060 |
| AP **global OOF** | 0,359586 | **0,384209** | +0,024623 |
| Brier **global OOF** | **0,195245** | 0,208927 | +0,013682 (peor) |
| Balanced Accuracy global, corte score 0,5 | 0,538619 | **0,598116** | +0,059498 |
| Acierto top-1 en iguales 1.244 ternas de sitios | 43,971 % | **45,659 %** | +1,688 puntos, 21 aciertos |

La media por fold NO coincide necesariamente con la métrica global agregada, porque los folds tienen tamaños y prevalencias distintos. Se mantienen ambas; la regla del ganador es la primera.

**Bootstrap pareado de 783 registros en 143 grupos espaciales, 320 repeticiones:** intervalos exploratorios de diferencia nuevo menos anterior: ROC-AUC [−0,013011; +0,088146], AP [−0,016859; +0,071033], Brier [−0,017124; +0,034189]. Todos incluyen cero, por tanto no permiten declarar superioridad robusta. Se reutilizaron datos para desarrollar y comparar; no son un test externo ciego.

### ¿Por qué cambió el ganador?

El CatBoost anterior obtuvo AUC medio 0,622455 y el Random Forest anterior 0,611241. Tras el nuevo entrenamiento, Random Forest subió a 0,624514 y CatBoost bajó a 0,599271. Puede contribuir la política QA95 y la optimización, pero **no se puede atribuir el cambio exclusivamente al filtro** sin una ablación adicional: no se abrió otra fase por este motivo.

## 4. Comparación contra el protocolo del release original de septiembre

La referencia metodológicamente defendible es el **protocolo CatBoost del release de septiembre reentrenado en los folds originales**, no el binario ya ajustado sobre registros que pudo haber visto. Para el contraste se verificaron 706 casos históricos iguales y un corte de etiqueta externo común **Li >=21,6 ppm**.

| Métrica en 706 test iguales | Protocolo previo reentrenado | Nuevo RF | Diferencia |
|---|---:|---:|---:|
| ROC-AUC ↑ | 0,627442 | **0,650201** | +0,022759 |
| AP ↑ | 0,370013 | **0,387217** | +0,017204 |
| Balanced Accuracy, score 0,5 ↑ | 0,540017 | **0,597666** | +0,057648 |
| Recall, score 0,5 ↑ | 0,123596 | **0,477528** | +0,353933 |
| Brier ↓ | **0,185858** | 0,212609 | +0,026751 (peor) |
| Mejor selección entre tres sitios históricos (1.246 ternas) ↑ | 42,135 % | **44,302 %** | +2,167 puntos |

Bootstrap en 66 grupos espaciales históricos, 320 remuestreos: IC 95 % exploratorio para la diferencia ROC-AUC [−0,006316; +0,046755], AP [−0,014986; +0,053920] y Brier [+0,006653; +0,041885]. En este contraste el empeoramiento de Brier sí resulta consistente dentro del procedimiento bootstrap, aunque las diferencias de AUC y AP no excluyen cero.

El entrenamiento previo usó 706 históricas y el nuevo incorporó las 77 GEOROC provisionales con QA95, y los umbrales internos de fold pueden diferir. **Son comparables en casos y reglas de evaluación, no en la composición de TRAIN.** Los valores antiguos del metadata publicado, como AUC 0,561697, no deben mezclarse directamente con esta reconstrucción OOF.

## 5. Utilidad de priorizar tres zonas

En la competencia de 783, RF eligió la ubicación con mayor mediana de Li entre tres sitios distintos del mismo fold en **568 de 1.244 ternas (45,659 %)**. El CatBoost anterior acertó **547 de las mismas 1.244 (43,971 %)**. Referencia aleatoria: aproximadamente 33,33 %.

Cuando alguna zona de la terna contiene Li alto, el RF acertó 47,995 % (748 ternas). Los sorteos reutilizan ubicaciones; NO constituyen 1.244 descubrimientos independientes. Modelo 1 estima ppm; el score de Modelo 2 es una evidencia espectral relativa para **priorizar investigación de zonas**, no probabilidad de yacimiento ni factibilidad de extracción.

Umbral operativo del RF candidato: **0,441563**, Balanced Accuracy OOF 0,621468. Fue seleccionado mediante las mismas OOF y su resultado es **optimista**; no es validación sobre casos externos.

## 6. Hallazgos científicos y limitaciones

- SCL: 613 observaciones aceptadas como **proxy óptico**, 170 con abstención/revisión. SCL no certifica roca y **no se aplicó como filtro obligatorio** a esas 170.
- Dominio espectral: 26 observaciones tienen >25 % de variables fuera del perfil percentil 1–99 histórico. Es diagnóstico, no motivo de borrado.
- **25 sitios históricos con etiquetas altas/bajas contradictorias** para el mismo punto bajo corte Li 21,6 ppm; no se modificaron etiquetas.
- **77 GEOROC provisionales**: fuentes identificadas, sin correspondencia de muestra de campo certificada.
- 111 escenas Sentinel-2. Compartir escena no demuestra por sí solo fuga, pero sí requiere cautela al inferir independencia regional/temporal.
- Las simulaciones con píxeles artificialmente faltantes son ensayos separados; no se confundieron con los 16 parches reales rechazados SOLO durante TRAIN por QA95.
- Es **validación interna exploratoria**, sin test ciego externo, sin certificación geológica ni validación de extracción económica.

## 7. Corrección de caracteres del menú

Origen identificado: las cadenas de texto de src/lithiumscope/cli/train_command.py contenían **mojibake literal en el código fuente**, por una conversión previa de UTF-8: aparecían caracteres de dibujo de cajas en lugar de las letras acentuadas, flechas y rayas. **No eran errores de sintaxis:** Python seguía ejecutando, pero mostraba textos corruptos.

Se corrigieron exclusivamente los textos visibles (—, ó, ú, é, →), además de añadir pruebas de regresión para el menú y mensajes finales. No cambiaron las rutas de entrenamiento, opciones, algoritmos, resultados ni logs.

## 8. Evidencias y estado final

Carpeta real del entrenamiento:
- results/model_2/runs/training_20261112_123451_m0300/tables/competition_ranking.csv
- tables/fold_metrics_*.csv y tables/oof_predictions_*.csv
- tables/winner_operating_threshold.csv
- tables/three_zone_competition_summary.json y tables/three_zone_competition_cases.csv
- tables/old_release_protocol_comparison_706.csv, tables/old_release_protocol_intervals_706.csv
- tables/scientific_diagnostics_summary.json, tables/scientific_stratified_metrics.csv
- tables/evidence_report_status.json (**complete** para los tres informes)
- run.json, dashboard.html, carpeta exports y logs originales.

Auditoría adicional **sin reentrenar**:
- results/scientific_simulations/m2_closure_20261112/closure_evidence.json
- paired_783_all_metrics.csv, bootstrap_comparisons_783.json
- paired_three_zone_summary_783.json, paired_three_zone_cases_783.csv
- analyze_reproducing.py y sus registros de ejecución.

**Conclusión:** cerrar la etapa de mejora metodológica y el experimento del Modelo 2. Se registran mejoras descriptivas en discriminación, Balance Accuracy y priorización, pero no una superioridad universal o estadísticamente concluyente; la calibración Brier empeoró. Conservar al nuevo Random Forest como candidato experimental **pending** sin reemplazar el Modelo 2 activo. El eventual release conjunto con un Modelo 1 nuevo se decidirá más adelante.
