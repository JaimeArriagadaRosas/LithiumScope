# LithiumScope — Fase E: unidad de sitio y calidad de los parches

**Fecha:** 12 de noviembre de 2026. **Estado:** piloto ejecutado; no se promovieron modelos y no hay evidencia de validación externa independiente.

## Pregunta científica

¿Es suficiente representar múltiples muestras geoquímicas mediante una etiqueta por sitio, excluir imágenes incompletas o restringir la representación espectral al centro de la imagen para mejorar la estabilidad del Modelo 2? Se evaluaron por separado los cambios y una combinación exploratoria.

La auditoría de Fase D detectó 25 sitios históricos con 77 observaciones que proporcionaban **exactamente el mismo parche** a pesar de tener etiquetas Li alta/baja incompatibles y 16 parches históricos con menos del 95% de píxeles finitos. Ninguna de estas alertas demuestra por sí sola que la medición geoquímica sea errónea ni establece un afloramiento físico.

## Protocolo experimental

Se mantuvieron **706 ejemplos históricos de evaluación, cinco folds geográficos congelados y 177 etiquetas positivas**, con umbral Li q75 calculado solo usando las muestras históricas del entrenamiento de cada fold. Se excluyeron de todas estas pruebas las 77 muestras GEOROC provisionales, porque la pregunta era aislar los efectos de representación y QA de la base histórica.

Para reducir coste, la referencia se reutilizó exactamente desde el piloto histórico de Fase C; se verificaron SHA-256 del manifiesto, caché, folds y archivo OOF. Se entrenaron únicamente 25 estimadores transitorios: cinco variantes nuevas × cinco folds, todos con CatBoost de 160 iteraciones, profundidad cinco, aprendizaje 0,065, regularización L2=5 y semilla 42. No se cambiaron hiperparámetros en función de resultados.

Las variantes por sitio deciden una clase usando solo **Li de TRAIN**: mediana numérica por sitio o mayoría estricta (los empates se excluyen solo del entrenamiento). Un sitio aporta una sola fila espectral si todos sus parches coinciden en SHA-256; una inconsistencia genera error bloqueante, no promediación silenciosa. Esta política es una hipótesis de etiquetado, no un certificado de identidad litológica.

En la variante de calidad, únicamente filas de entrenamiento con menos de 95% de valores finitos se retiran; las 706 observaciones de test se mantienen, incluso las 16 incompletas. Para el centro, se extraen 126 características sobre la ventana fija [8:24, 8:24] de cada imagen original de seis bandas; se aplica exactamente la misma transformación a train y test. No se utilizan los diez casos conocidos para elegir variantes.

## Resultados pareados sobre los mismos 706 test (OOF exploratorio)

| Variante | ROC-AUC ↑ | Average Precision ↑ | Brier ↓ | Recall con score >= 0,5 |
|---|---:|---:|---:|---:|
| Referencia histórica (706 muestras) | 0,6207 | 0,3827 | 0,1841 | 16,9% |
| Una observación por sitio; Li mediano de TRAIN | 0,5963 | 0,3550 | 0,1891 | 17,5% |
| Una observación por sitio; consenso estricto TRAIN | 0,6080 | 0,3576 | 0,1885 | 14,7% |
| Solo entrenamiento con >=95% píxeles finitos | 0,6447 | 0,4014 | 0,1807 | 19,2% |
| Sitio mediano y QA del 95% (interacción) | 0,6065 | 0,3343 | 0,1916 | 14,7% |
| 126 características de región central 16x16 | 0,5960 | 0,3543 | 0,1918 | 18,1% |

**La única variante con una mejora descriptiva simultánea en ROC-AUC, AP y Brier fue el filtro de calidad del TRAIN al 95%:** cambios AUC 0,0240, AP 0,0187 y Brier -0,0035. El recall a umbral fijo de 0,5 continúa bajo y no se optimizó aquí. **Estas diferencias no equivalen a una mejora confirmada de generalización.**

Las políticas de mediana y mayoría redujeron AUC/AP al pasar de múltiples muestras a una fila por sitio. No por ello se descartan como principios científicos; no se han evaluado etiquetas respaldadas por geología de campo ni una función objetivo que represente heterogeneidad real.

La ventana central de 16×16 también disminuyó las métricas agregadas en este ensayo: reducir el tamaño espacial sin una evidencia previa sobre exposición de roca no garantiza mayor correspondencia entre píxel y muestra.

## Incertidumbre pareada por grupo histórico

Se hicieron 320 remuestreos por grupos espaciales históricos (**66 grupos**), con intervalos exploratorios de 95% de diferencias entre la referencia y cada alternativa. El trabajo previo de fases A–D ya analizó esos datos; existen comparaciones múltiples y dependencia de escenas. No son pruebas confirmatorias ni un test externo.

| Variante vs referencia | ΔROC-AUC | IC95% AUC | ΔAP | IC95% AP | ΔBrier | IC95% Brier |
|---|---:|---:|---:|---:|---:|---:|
| Una observación por sitio; Li mediano de TRAIN | -0,0244 | [-0,0506, 0,0189] | -0,0276 | [-0,0624, 0,0185] | 0,0050 | [-0,0034, 0,0106] |
| Una observación por sitio; consenso estricto TRAIN | -0,0127 | [-0,0455, 0,0344] | -0,0251 | [-0,0638, 0,0271] | 0,0044 | [-0,0045, 0,0108] |
| Solo entrenamiento con >=95% píxeles finitos | 0,0240 | [-0,0086, 0,0697] | 0,0187 | [-0,0105, 0,0589] | -0,0035 | [-0,0119, 0,0020] |
| Sitio mediano y QA del 95% (interacción) | -0,0142 | [-0,0475, 0,0360] | -0,0483 | [-0,0785, -0,0009] | 0,0075 | [-0,0004, 0,0140] |
| 126 características de región central 16x16 | -0,0247 | [-0,0607, 0,0284] | -0,0284 | [-0,0757, 0,0239] | 0,0077 | [-0,0022, 0,0161] |

**Todos los intervalos exploratorios de ΔAUC incluyen cero**. El filtro del 95% también tiene intervalos de ΔAP y ΔBrier que incluyen cero. No procede promocionar ningún candidato por estas cifras.

## ¿Qué cambia cuando evaluamos por sitio?

La evaluación por sitio se calcula a posteriori como un análisis descriptivo adicional, con una puntuación mediana de cada sitio en el fold test y etiqueta de la mediana de Li usando el umbral del fold. No se utiliza para fijar la política de entrenamiento.

| Variante | Sitios test | ROC-AUC por sitio | AP por sitio | Brier por sitio |
|---|---:|---:|---:|---:|
| Referencia histórica (706 muestras) | 516 | 0,6198 | 0,3925 | 0,1911 |
| Una observación por sitio; Li mediano de TRAIN | 516 | 0,5963 | 0,3716 | 0,1959 |
| Una observación por sitio; consenso estricto TRAIN | 516 | 0,6011 | 0,3714 | 0,1965 |
| Solo entrenamiento con >=95% píxeles finitos | 516 | 0,6367 | 0,4133 | 0,1874 |
| Sitio mediano y QA del 95% (interacción) | 516 | 0,6070 | 0,3538 | 0,1967 |
| 126 características de región central 16x16 | 516 | 0,5945 | 0,3761 | 0,1958 |

Se detectaron **24 sitios de test con etiquetas alto/bajo mezcladas** al utilizar el umbral Li específico de su fold, y al menos **27 errores de clasificación inevitables** en esos sitios si la función solo recibe una misma entrada espectral por ubicación. La cota se calcula como suma de la clase minoritaria por sitio entre las etiquetas test; **no** es el error mínimo para una región externa ni implica que todas las etiquetas sean falsas. Se diferencia de los 25 sitios / 28 etiquetas minoritarias de la Fase D porque allí se usó un corte fijo de 21,6 ppm y aquí cortes propios de folds.

La reducción de etiquetas contradictorias no crea automáticamente una señal espectral capaz de distinguir litologías si el parche sigue mezclando materiales. Para decidir una política de sitio hacen falta mapas de afloramiento, posiciones GPS y evidencia geológica.

## Riesgo de los 16 parches incompletos

Los 16 registros históricos de cobertura espectral insuficiente permanecieron en los cinco test. La exclusión de QA afectó únicamente las muestras train y el número resultante se documenta en fold_metrics.csv.

Los valores de evaluación del subgrupo de 16 están basados en **solo 3 positivos**; no son suficientes para concluir que el filtro detecta o predice bien muestras de baja calidad.

La auditoría adicional de máscaras reveló que las **16 observaciones contienen solo 10 imágenes distintas**. En las 16, las **seis bandas comparten exactamente la misma máscara de no-datos por píxel**, lo que indica ausencia espacial correlacionada en lugar de una banda aislada. En 6 observaciones el centro 16×16 supera el 95% de cobertura finita; en otras 10 no. Por tanto, usar siempre el centro **no es una reparación general**.

La investigación siguiente debe revisar las huellas de píxeles no válidos y la geometría/fecha de adquisición para saber por qué esas ventanas quedan parcialmente sin datos, y si la imputación por mediana introdujo patrones espurios. La readquisición, si se autoriza, debe preservar originales y repetir un protocolo temporal predefinido. Ningún píxel fue interpolado en esta auditoría.

## Decisión metodológica

1. **Retener como hipótesis** el filtro de calidad del 95% para un futuro experimento pre-registrado; no activarlo ni escogerlo basándose en este mismo conjunto de desarrollo.
2. No imponer agregación automática por mediana o mayoría: estas dos políticas no resolvieron la relación espectro–Li en los 706 registros.
3. Mantener la unidad de sitio en el diagnóstico, calcular incertidumbre por grupo/escena y añadir evidencia de afloramiento antes de evaluar una nueva etiqueta.
4. Examinar el origen de los 16 parches incompletos; preparar reparación no destructiva con nuevas adquisiciones y comparar las 126 características, sin reemplazar las imágenes históricas.
5. Para confirmar una mejora, reunir etiquetas independientes y verificadas por superficie, separadas geográficamente desde el origen, y después repetir la comparación con los procedimientos previamente congelados.

## Evidencia reproducible

El runner opt-in es src/lithiumscope/model_2/methodology/site_patch_lab.py y produce experiment_design.json, central_16x16_features.csv, train_site_aggregation_decisions.csv, all_arms_paired_oof.csv, fold_metrics.csv, sample_quality_site_subgroups.csv, site_level_metrics.csv, identical_patch_label_error_bounds.csv, group_bootstrap_deltas.csv, paired_model_comparison.csv y experiment_summary.json.

Ejecutar desde la raíz:

    python -m lithiumscope.model_2.methodology.site_patch_lab --output results/scientific_simulations/m2_phase_e_20261112/pilot

El runner coteja manifiesto, 126 características, folds, línea base y hashes de imágenes originales con Fases C/D antes de entrenar los candidatos temporales. No utiliza datos GEOROC provisionales, no cambia la función de inferencia del release, no guarda modelos de producción, no hace push.
