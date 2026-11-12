# LithiumScope — informe experimental Fase C (2026-11-12)

Estado: ocho ablaciones ejecutadas y validación exploratoria de decisiones internas. La evidencia no es un nuevo test externo ni autoriza promoción de modelos.

## Protocolo reproducible
Se empleó un CatBoost fijo (160 iteraciones, profundidad 5, tasa 0,065, L2=5, semilla=42, CPU), sin búsqueda de hiperparámetros ni ajuste de pesos después de ver el test. Los cinco folds espaciales y los **706 índices de test históricos fueron idénticos en las ocho variantes**, con 177 etiquetas positivas construidas a partir del cuantil de Li histórico obtenido solo en el train de cada fold. Las 77 observaciones GEOROC nuevas son provisionales; en cada fold se incorporaron únicamente aquellas asignadas originalmente al train.

El fin es aislar el efecto de una intervención por vez. No se usaron los diez casos conocidos de demostración, no se descargaron datos, no se regeneraron imágenes, no se ajustó el modelo oficial y no se eligió un algoritmo o umbral basándose en los 706 resultados.

## Resultados pareados (datos históricos, métricas OOF exploratorias)
| Brazo | ROC-AUC | AP | Brier (menor mejor) | Recall a 0,5 |
|---|---:|---:|---:|---:|
| Solo 706 históricos | 0,6207 | 0,3827 | 0,1841 | 0,1695 |
| 706 + GEOROC provisionales | 0,6318 | 0,3766 | 0,1839 | 0,1695 |
| Adicionales + buffer 5 km | 0,6436 | 0,3641 | 0,1838 | 0,1130 |
| Adicionales + buffer 10 km | 0,6193 | 0,3507 | 0,1874 | 0,1017 |
| Excluir escenas de test del train | 0,6128 | 0,3738 | 0,2039 | 0,2203 |
| Excluir sitios mixtos solo del train | 0,6335 | 0,3680 | 0,1879 | 0,1469 |
| 48 predictores de banda directa | 0,6182 | 0,3542 | 0,1897 | 0,1243 |
| Ponderación inversa por sitio | 0,6090 | 0,3789 | 0,1875 | 0,1921 |

El contraste +GEOROC frente a 706 cambia AUC y AP en direcciones distintas. La exclusión de escenas reduce considerablemente el tamaño de train (en este piloto, mínimo 251 filas), por lo que sus variaciones de métricas no permiten atribuir causalmente el efecto a filtraciones previas. Un buffer de 10 km no reproduce el AUC obtenido con 5 km, lo que cuestiona una regla simple de mayor distancia igual a mejor discriminación.

## Incertidumbre agrupada y selección del ganador
Se aplicaron 320 remuestreos de los **66 grupos históricos** (intervalos percentiles exploratorios de 95%) a las diferencias pareadas de AUC, AP y Brier. No es inferencia confirmatoria: el conjunto histórico ya se ha utilizado para trabajo metodológico, hubo comparaciones múltiples y puede existir dependencia entre grupos y escenas.

| Contraste | Delta AUC | IC exploratorio 95% AUC | Delta AP | IC exploratorio 95% AP |
|---|---:|---:|---:|---:|
| 706 + GEOROC provisionales vs Solo 706 históricos | 0,0111 | [-0,0189, 0,0538] | -0,0061 | [-0,0467, 0,0285] |
| Adicionales + buffer 5 km vs 706 + GEOROC provisionales | 0,0118 | [-0,0316, 0,0486] | -0,0125 | [-0,0360, 0,0181] |
| Adicionales + buffer 10 km vs 706 + GEOROC provisionales | -0,0124 | [-0,0383, 0,0208] | -0,0259 | [-0,0591, 0,0129] |
| Excluir escenas de test del train vs 706 + GEOROC provisionales | -0,0190 | [-0,0769, 0,0441] | -0,0028 | [-0,0647, 0,1094] |
| Excluir sitios mixtos solo del train vs 706 + GEOROC provisionales | 0,0017 | [-0,0218, 0,0325] | -0,0086 | [-0,0378, 0,0312] |
| 48 predictores de banda directa vs 706 + GEOROC provisionales | -0,0136 | [-0,0563, 0,0320] | -0,0224 | [-0,0620, 0,0251] |
| Ponderación inversa por sitio vs 706 + GEOROC provisionales | -0,0227 | [-0,0593, 0,0218] | 0,0023 | [-0,0258, 0,0382] |

Todas las bandas de incertidumbre de AUC **incluyen cero**: los datos no respaldan una ganancia estadística robusta de una variante concreta. Eso no demuestra equivalencia entre los métodos. Tampoco se debe elegir el buffer ni las características a partir del máximo observado.

## Umbrales aprendidos exclusivamente dentro del entrenamiento
Se ejecutó un experimento adicional en dos brazos predefinidos, con tres folds internos agrupados espacialmente por cada fold exterior, sin examinar las etiquetas de test para fijar el corte. Se reutilizaron las puntuaciones exteriores verificando equivalencia numérica exacta entre el piloto de siete brazos y la corrida final de ocho.
| Métrica de decisión exterior | Históricos | Con buffer 5 km |
|---|---:|---:|
| Recall con score >= 0,5 | 0,1695 | 0,1130 |
| Recall con umbral OOF interno | 0,3729 | 0,4802 |
| Precisión con umbral OOF interno | 0,3708 | 0,3455 |
| Balanced Accuracy con umbral OOF interno | 0,5806 | 0,5879 |

El recall aumenta al seleccionar umbrales internos, pero la precisión sigue limitada. **No se realizó calibración probabilística**, ni comparación confirmatoria independiente. Los umbrales obtenidos no se aplicaron ni instalaron sobre ningún modelo activo.

## Hallazgos metodológicos y próximas intervenciones
1. Antes de elegir un candidato, auditar en origen los 25 sitios de etiquetas discordantes y el significado físico de cada muestra: representatividad del afloramiento respecto del parche Sentinel-2, calidad GPS y análisis geoquímico.
2. No inferir que las 635 observaciones de test que comparten identificador de escena son filtración efectiva. Hacer una evaluación por escena geográfica/fecha con control del tamaño de train y análisis de calibración del dominio.
3. Mantener benchmarks congelados y nuevas regiones/escenas fuera del desarrollo para una confirmación final. El conjunto histórico de 706 registros ya participa en el desarrollo de Fase C.
4. Priorizar selección de umbral y calibración verdaderamente anidadas con objetivos explícitos de recall y coste de falsos positivos, nunca fijar umbrales con los diez casos.
5. Explorar mejores descriptores de roca expuesta/máscaras SCL y una etiqueta por sitio verificable; comparar de modo pareado con datos nuevos y sin mezclar cambios de protocolo.

## Límites de la evidencia
Las diferencias de score entre brazos no estiman por sí mismas la probabilidad de descubrir depósitos. No se validó correspondencia de litio en terreno; se conservó la etiqueta operativa de q75. La exclusión de sitios mixtos se hizo solo en train, mientras el test mantiene su definición original. La reducción a 48 columnas estaba predefinida, no seleccionada según test. Los modelos de este estudio son pequeños candidatos temporales y **no se guardaron como modelos de producción**.

## Archivos auditables
- `experiment_design.json`
- `experiment_summary.json`
- `all_oof_arms.csv`
- `fold_metrics.csv`
- `exploratory_comparison.csv`
- `cluster_bootstrap_exploratory.csv`
- `site_aggregated_metrics.csv`
- `regional_metrics_descriptive.csv`
Los archivos `inner_frozen_thresholds_*.csv` y `decision_probe_summary.json` están en el subdirectorio del primer piloto de siete brazos; su alineación con los resultados finales fue comprobada numéricamente.

El código que reproduce la ejecución reside en `src/lithiumscope/model_2/methodology/ablations.py` y `decision_probe.py`. No se ha hecho push ni se ha alterado el menú o los modelos activos.
