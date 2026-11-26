# Resultados consolidados

## Identidad de modelos

| Grupo | Release | Candidato conservado |
| --- | --- | --- |
| M1 | SVM, training_20261106_111653_m0300 | RF, training_20261125_115932_m0300_826123c1 |
| M2 | CatBoost, training_20260920_195933_m0300 | RF, training_20261125_144535_m0300_6fa5d269 |

M1: SHA-256 del dataset compartido
94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94.
3420 filas, 33 predictores, 244 grupos, cinco folds espaciales. Se contrastaron
los OOF históricos y la campaña balanceada mediante row_index/row_id,
row_position/position, fuente, verdad, grupo y fold. 3420 coincidencias únicas.
El manifiesto exterior histórico coincide por SHA-256 con la referencia
archivada del experimento. No descartar 298 observaciones ambiguas de la clave
compuesta cuando la identidad explícita está verificada.

| M1 | RMSE medio | RMSE global | MAE global | R2 medio | R2 global |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVM release | 87.448705 | 159.846300 | 16.320391 | .246772 | .042980 |
| RF actual | 82.310168 | 152.302214 | 15.303595 | .251477 | .131184 |
| RF balanceo 2700 | 81.431717 | 151.648396 | 15.300003 | .303601 | .138627 |
| RF selección interna | 81.533788 | 152.221012 | 15.191764 | .311035 | .132110 |

Bootstrap agrupado exploratorio de 5000 remuestreos, diferencia RMSE candidato
menos release, IC95 reportados en la revisión previa: RF actual -7.544
[-12.676, .661]; RF 2700 -8.198 [-13.717, .115]; RF selección -7.625
[-12.727, -.126]. Reproducir y guardar las tablas e intervalos con el comparador
corregido. No inferir independencia estadística de folds reutilizados.
No afirmar que se han guardado modelos balanceados.

### Campañas completadas

- Focalizada 45 configuraciones: 1350 evaluaciones + 540 calibraciones,
  164160 predicciones de desarrollo. No logró superar simultáneamente las
  mejores referencias internas. RF/HGB seleccionaron controles; CatBoost
  redujo MAE, pero no mejoró RMSE global archivado. Variabilidad del control
  CatBoost debe distinguirse de efectos de tratamiento.
- Balanceo 21 configuraciones: 630 evaluaciones base + 591 ajustes expertos,
  82080 predicciones, sin errores, 30 particiones y 15 selecciones internas.
  RF 2700 mejora descriptivamente frente al control; sus intervalos
  contra el control incluyen cero. Selección interna reduce MAE y brecha
  R2 TRAIN-TEST. Los expertos no son recomendados con los datos actuales.
  Las cifras 3000/2700/2400/2043 son tamaños equivalentes de cohorte, no
  número literal de filas en cada TRAIN. Se conserva TEST completo.
- Concentraciones elevadas: fold 5 contiene nueve observaciones superiores
  al máximo TRAIN de 696 ppm. Siete concentran aproximadamente 95.82% del
  SSE del RF actual. No excluirlas para mejorar artificialmente las métricas.
  Los árboles no extrapolan bien a valores sin representación suficiente.
  Los resultados históricos de CatBoost no se deben ocultar: RMSE global
  anterior 137.54, R2 global .2914; no es el SVM publicado.

## M2

| M2 global, protocolos propios | CatBoost release | RF actual |
| --- | ---: | ---: |
| ROC-AUC | .561697 | .665983 |
| Average Precision | .276532 | .400357 |
| Brier | .214409 | .182237 |

Release: 706 muestras, 126 predictores, 66 grupos; hash
b04b1291a8a7cbf24538253e521a93f32ff8c765ff437bd7c2ba5666a095121e.
Candidato: 706, 170 predictores, 66 grupos; hash
5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1.
OOF publicado original tiene 3 columnas y carece de identidad por muestra.
No comparar por posición ni declarar equivalencia de folds. Comparación
descriptiva sí es posible. Comparación emparejada reconstruida a 21.6 ppm,
706 muestras, mismas claves y grupos: AUC .6274 -> .6570, AP .3700 ->
.3969, Brier .1859 -> .1842; intervalos incluyen cero. La referencia
reconstruida no es el modelo publicado. Reproducir esta comparación desde
all_reference_paired_oof.csv con controles de identidad.

## EDA candidatos GEOROC

| Fuente | Registros brutos | Vista compatible | QC | Sin alerta inicial | >696 | >1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Andean Arc | 27077 | 2998 | 2755 | 6 | 0 | 0 |
| Granite/Rhyolite | 59790 | 8234 | 7576 | 6849 | 69 | 53 |

69 muestras altas: 14 grupos a 0.5 grados, 13 referencias bibliográficas,
máximo 6201 ppm; todas intraplaca y fuera del dominio andino (revisión previa).
318 muestras de margen convergente superaron el cribado, ninguna >696 ppm.
No incorporar todas las muestras indiscriminadamente. Cribado inicial no
demuestra independencia, exactitud de mediciones, ausencia de duplicados
entre fuentes candidatas o equivalencia geológica. Cero muestras aprobadas.

## Piloto académico OASIS

ID 20261126T123613_24ef3f6c. 3263 observaciones reales únicas, 100 trials en
total (20 por fold), 5x5 KFold aleatorio. Recorte previo al QC: 4.2439–126.12
ppm; rango efectivamente evaluado 4.27–123.8 ppm. 157 filas menos que la
cohorte espacial de 3420. 130–144 grupos compartidos TRAIN/TEST por fold.
Se recalcularon las métricas de las 3263 predicciones; coinciden con manifiesto.

| Indicador | OASIS (681) | Piloto (3263) |
| --- | ---: | ---: |
| RMSE medio | 5.361 | 7.723715653 |
| RMSE desviación | .595 | .813801271 |
| R2 medio | .499 | .561373693 |
| R2 desviación | .086 | .024873980 |
| RMSE global | no reportado | 7.757604754 |
| MAE global | no reportado | 4.296776976 |
| R2 global | no reportado | .560445164 |

4.79 veces más muestras, 20 vs 100 trials/fold. Diferentes fuentes, dominio,
preparación y presupuesto. No afirmar superioridad integral sobre OASIS:
R2 descriptivo mayor y RMSE mayor. No comparar error de este dominio
recortado con RMSE espacial de dominio completo. No repetir piloto salvo
justificación específica de defensa.

Referencia académica: Alarcón Arias y Saud Rauld (2026), Predicción de
concentraciones de litio mediante modelos de aprendizaje automático aplicados
a datos geoquímicos y geoespaciales, secciones 7.3, 8.4, 8.6 y 9.1.1.
