# Modelo 1: diagnóstico comparado en cuatro fases — release histórico y candidato ampliado

**Fecha:** 13/11/2026. **Estado:** auditoría retrospectiva ejecutada sin activar modelos, sin Optuna, sin alterar menú ni datos.

**Versiones:** protocolo histórico del release M1 **1.120 observaciones** (OOF run `training_20261105_204029_m0300`, fuente SHA idéntico al release de septiembre) y candidato experimental M1 **3.420 observaciones** (OOF run `training_20261106_111653_m0300`).

**Artefactos de salida:** `results/scientific_simulations/m1_dual_diagnostics_20261113/`.

**Script reproducible:** `scripts/scientific/m1_dual_release_diagnostics.py`.

## 0. Contrato experimental, fuentes y control de integridad

El M1 **es regresión** de concentración `Li_icpms` en ppm. No sustituimos esa tarea por clasificación. Todos los resultados clasificados VP/VN/FP/FN son **auxiliares** para diagnosticar selección de muestras de Li alto.

Se reconstruyó la fuente de noviembre correspondiente al histórico **con el algoritmo de armonización original**, `iron_total_mode=original`, a partir de Mamani09 y GEOROC presentes en disco. SHA-256 **`2f99d38960c31a02b85aa5328c910c3aeab5ec1760cf6ef035922fbb56d41ea1`**, **identidad exacta** con el dataset usado por su entrenamiento. Se verificaron orden y valor de las 1.120 etiquetas OOF. Para el candidato se verificó SHA-256 **`94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94`**, 3.420 etiquetas OOF, 33 predictores.

Se reconstruyeron exactamente los cinco folds espaciales a partir de sus checkpoints y `GroupKFold`. No existe ningún grupo espacial presente simultáneamente en TRAIN y TEST de un mismo fold. El SVM de cada fold se volvió a ajustar sobre TRAIN usando sus **parámetros Optuna históricos guardados**, sin nueva búsqueda. Comparación numérica máxima entre predicciones recalculadas y OOF archivadas: **7,11×10⁻¹⁵ ppm** (histórico), **5,68×10⁻¹⁴ ppm** (candidato). **Se reproducen los diez folds SVM exactamente**.

**Importante:** un refit verificable del protocolo de 1.120 no debe llamarse validación directa del artefacto desplegado de septiembre; el hash de la fuente y la coincidencia de resultados proporcionan el puente trazable. La reutilización de CV de desarrollo no es un holdout externo nuevo.

## Fase 1 — Diagnóstico de sobreajuste (*overfitting*) y subajuste (*underfitting*)

### 1.1 Evaluación OOF de los seis algoritmos

Las métricas siguientes son **calculadas sobre todas las predicciones OOF agrupadas**, no promedios de las métricas individuales de los folds. Menor RMSE/MAE es mejor; mayor R² es mejor.

| Versión | Modelo | RMSE OOF | MAE OOF | R² OOF |
|---|---|---:|---:|---:|
| Histórico 1.120 | Random Forest | 25,743 | 8,341 | 0,0845 |
| Histórico 1.120 | XGBoost | 26,625 | 8,969 | 0,0207 |
| Histórico 1.120 | **SVM ganador** | **25,685** | **8,128** | 0,0886 |
| Histórico 1.120 | TabNet | 30,071 | 13,533 | −0,2492 |
| Histórico 1.120 | HistGradientBoosting | 25,065 | 9,733 | **0,1321** |
| Histórico 1.120 | CatBoost | 25,232 | 8,283 | 0,1205 |
| Candidato 3.420 | Random Forest | 154,933 | 16,957 | 0,1009 |
| Candidato 3.420 | XGBoost | **140,128** | 16,389 | **0,2645** |
| Candidato 3.420 | **SVM ganador** | 159,846 | **16,320** | 0,0430 |
| Candidato 3.420 | TabNet | 155,855 | 18,901 | 0,0902 |
| Candidato 3.420 | HistGradientBoosting | 157,040 | 23,996 | 0,0763 |
| Candidato 3.420 | CatBoost | 145,215 | 16,594 | 0,2102 |

**Importante:** SVM ganó ambas **competiciones** por la regla registrada de **menor RMSE medio de los cinco folds**, no necesariamente por menor RMSE sobre la unión de OOF. En el candidato, XGBoost alcanza el mejor RMSE y R² OOF **global**: diferencia de agregación a estudiar, no error de lectura. Elegir un algoritmo retrospectivamente por las cifras OOF agrupadas después de evaluar todos los modelos produciría sesgo de selección.

### 1.2 Brechas entrenamiento/validación (SVM de ambas versiones)

Se reentrenó para diagnóstico el **SVM de cada fold** sobre exactamente sus TRAIN históricos usando su hiperparámetro congelado; se evaluó en el mismo TRAIN y en su TEST espacial.

| Versión | MAE medio TRAIN | MAE medio TEST | Brecha TEST−TRAIN | R² medio TRAIN | R² medio TEST |
|---|---:|---:|---:|---:|---:|
| Histórico 1.120 | 3,691 ppm | 8,128 ppm | **+4,437 ppm** | ≈0,569 | ≈0,070 |
| Candidato 3.420 | 12,163 ppm | 16,320 ppm | **+4,157 ppm** | ≈0,286 | ≈0,247 |

**Evidencia:** los dos SVM muestran error MAE de TEST mayor que el de TRAIN en promedio. El histórico presenta una brecha de R² considerable; el candidato muestra R² bajos en TRAIN y TEST, compatibles con dificultad de capturar relaciones más allá de los valores típicos.

**Precaución:** TRAIN y TEST no tienen necesariamente la misma distribución de Li y regiones. Por eso la diferencia de RMSE de un fold puede ser **negativa** aunque el test tenga mayor dificultad en otros folds. No debemos diagnosticar «overfitting confirmado» solo porque el error TEST supere el error TRAIN, ni «underfitting confirmado» por un R² reducido.

**Diagnósticos especiales:**
- Histórico, fold 3: RMSE TRAIN **17,03**, TEST **36,69**; fold 4: TRAIN **10,29**, TEST **41,02**. Son diferencias considerables que sugieren fragilidad espacial / brecha de generalización.
- Candidato, fold 5: TRAIN **17,30**, TEST **353,62**, R² TEST **0,0178**. Una región de test con valores excepcionales altera enormemente el error global.
- Candidato, fold 1: RMSE TRAIN **159,79**, TEST **13,86**. No indica que el modelo «aprenda peor» en su propio entrenamiento; su TRAIN contiene extremos mucho más difíciles que el TEST de ese fold.

**Hipótesis H1:** el entrenamiento M1 puede mostrar sobreajuste localizado en algunos dominios espaciales. **Compatible con brechas observadas**, pero necesita distancias geográficas y validación independiente.

**Hipótesis H2:** el SVM ampliado presenta insuficiente capacidad/representación para la cola extrema de Li. **Fuertemente sugerido por errores de cola**, pero no debe denominarse underfitting universal sin revisar otras familias/funciones objetivo.

## Fase 2 — Curvas de aprendizaje espaciales completas

Se evaluó **cada ganador SVM en los cinco folds**. Para cada fold, se seleccionó el **25, 50, 75 y 100 % de los grupos espaciales de TRAIN**, con semilla fija y selección acumulativa, manteniendo **TEST completamente congelado y disjunto**. Se utilizaron los mismos hiperparámetros seleccionados en TRAIN completo del fold y NO se reoptimizó en cada fracción.

Promedios aritméticos entre los **cinco folds** (en ppm):

| Versión | Grupos de TRAIN | MAE TRAIN | MAE TEST | RMSE TEST |
|---|---:|---:|---:|---:|
| Histórico 1.120 | 25 % | 2,482 | 8,923 | 22,220 |
| Histórico 1.120 | 50 % | 3,051 | 8,747 | 21,653 |
| Histórico 1.120 | 75 % | 3,462 | 8,000 | 20,879 |
| Histórico 1.120 | 100 % | 3,691 | 8,128 | 20,986 |
| Candidato 3.420 | 25 % | 6,246 | 19,495 | 92,764 |
| Candidato 3.420 | 50 % | 10,715 | 18,552 | 91,733 |
| Candidato 3.420 | 75 % | 9,747 | 17,913 | 91,176 |
| Candidato 3.420 | 100 % | 12,163 | 16,320 | 87,449 |

**Interpretación:** el candidato suele reducir el error de TEST conforme aumenta la cobertura de TRAIN; esto sugiere que más datos pueden resultar útiles para su dominio. En el histórico, el mejor MAE TEST de estas fracciones se obtuvo en 75 %, no en 100 %: **no hay monotonía garantizada**.

**Limitaciones de la curva:** las fracciones incorporan grupos de distinta composición química y con distintas colas; el MAE de TRAIN puede aumentar al añadir casos difíciles. Los parámetros se congelaron antes de reducir TRAIN, lo que vuelve la curva un diagnóstico de tamaño condicionado a la configuración final, **no un estudio de reoptimización por tamaño**. Solo corresponde al ganador SVM; los otros cinco algoritmos tienen OOF evaluadas pero todavía **no** una curva TRAIN/TEST de cuatro fracciones. El registro detallado por fold está en `spatial_learning_curves_svm.csv`.

**Gráficos reutilizables de las curvas (promedios de cinco folds):**

![Curva de aprendizaje del release M1 con 1120 muestras](../../results/scientific_simulations/m1_dual_diagnostics_20261113/curva_aprendizaje_release_1120.png)

![Curva de aprendizaje del candidato M1 con 3420 muestras](../../results/scientific_simulations/m1_dual_diagnostics_20261113/curva_aprendizaje_candidate_3420.png)

**Hipótesis H3:** una ampliación controlada de grupos geográficos puede reducir el error medio del candidato en su TEST, aunque persista la falla de cola extrema. **Respaldada solo dentro de este diseño de desarrollo**, no probada externamente.

## Fase 3 — Matrices de confusión auxiliares: detección de Li alto

**Procedimiento antifuga:** para cada fold `j`, se calculó `q75_j = quantile(Li[TRAIN_j],0.75)`, **sin leer las etiquetas de TEST para elegir el umbral**. Con ese mismo `q75_j`:
- `y_real >= q75_j` es «alto Li» verdadero.
- `y_pred >= q75_j` es detección positiva a partir del regresor.
- Se cuentan VP/VN/FP/FN para cada algoritmo y se suman los cinco TEST OOF.

Los resultados son una tarea auxiliar de clasificación, **no reemplazan** los errores en ppm, no son probabilidades calibradas, y el umbral varía por fold por diseño.

| Versión | Modelo | VP | VN | FP | FN | Recall alto Li | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Histórico 1.120 | RF | 195 | 705 | 121 | 99 | 0,663 | 0,639 |
| Histórico 1.120 | XGBoost | 198 | 683 | 143 | 96 | 0,673 | 0,624 |
| Histórico 1.120 | **SVM** | **207** | 675 | 151 | **87** | **0,704** | 0,635 |
| Histórico 1.120 | TabNet | 56 | 677 | 149 | 238 | 0,190 | 0,224 |
| Histórico 1.120 | HGB | 187 | 658 | 168 | 107 | 0,636 | 0,576 |
| Histórico 1.120 | CatBoost | 211 | 688 | 138 | 83 | **0,718** | **0,656** |
| Candidato 3.420 | RF | 582 | 2.153 | 403 | 282 | 0,674 | 0,630 |
| Candidato 3.420 | XGBoost | 586 | 2.191 | 365 | 278 | **0,678** | 0,646 |
| Candidato 3.420 | **SVM** | 544 | 2.195 | 361 | **320** | 0,630 | 0,615 |
| Candidato 3.420 | TabNet | 254 | 2.128 | 428 | 610 | 0,294 | 0,329 |
| Candidato 3.420 | HGB | 566 | 2.120 | 436 | 298 | 0,655 | 0,607 |
| Candidato 3.420 | CatBoost | 571 | 2.235 | 321 | 293 | 0,661 | **0,650** |

**Nota:** el SVM histórico evaluó 294 casos «Li alto» (207+87), y el candidato 864 (544+320), sobre distribuciones diferentes; NO comparar recuentos absolutos como «mejor/peor». En la tarea auxiliar de alta concentración, el SVM candidato tiene recall **0,630** frente al histórico **0,704**; no evidencia de una mejora general en detección de Li alto. Otros algoritmos pueden ser más adecuados para detección auxiliar sin ser ganadores del ranking de regresión en ppm.

**Gráficos reutilizables de las matrices auxiliares del SVM:**

![Matriz auxiliar del SVM release 1120](../../results/scientific_simulations/m1_dual_diagnostics_20261113/confusion_auxiliar_release_1120.png)

![Matriz auxiliar del SVM candidato 3420](../../results/scientific_simulations/m1_dual_diagnostics_20261113/confusion_auxiliar_candidate_3420.png)

**Hipótesis H4:** la optimización por RMSE medio por fold no siempre selecciona el mejor detector de Li alto; se confirma descriptivamente que CatBoost histórico presenta mayor recall/F1 auxiliar que el SVM ganador por RMSE medio. Si la aplicación futura prioriza selección de muestras altas, habría que formalizarlo como **objetivo secundario predefinido**, no manipular la competición después de ver TEST.

## Fase 4 — Validación y diagnóstico integrado

### 4.1 Calidad, procedencia y sesgo

| Versión del SVM | Procedencia | Registros | RMSE OOF | MAE OOF | R² OOF |
|---|---|---:|---:|---:|---:|
| Histórico | Mamani09 | 676 | 8,454 | 5,380 | 0,340 |
| Histórico | GEOROC | 444 | 39,438 | 12,311 | 0,054 |
| Candidato | Mamani09 | 676 | 8,609 | 5,269 | 0,315 |
| Candidato | GEOROC | 2.744 | **178,402** | 19,043 | 0,042 |

El candidato incorpora **2.300 observaciones GEOROC adicionales** frente al histórico. Esto **no es** la ampliación de 77 muestras del M2, que constituye otro experimento.

El mayor error GEOROC está asociado al ensanchamiento de la cola, sin prueba causal de que las nuevas etiquetas estén equivocadas. En los 3.420 casos, la mediana de Li ≈15 ppm y el máximo alcanza 4.160 ppm; ocho casos superan 1.000 ppm.

### 4.2 Concentraciones extremas y concentración del error

| Indicador | Histórico 1.120 | Candidato 3.420 |
|---|---:|---:|
| Q99 de Li | 98,97 ppm | **240,0 ppm** |
| Máximo Li | 597,1 ppm | **4.160,0 ppm** |
| Observaciones con Li >100 ppm | 12 | 89 |
| Participación del mayor error cuadrático individual en SSE OOF SVM | 43,25 % | 19,38 % |
| Participación de los diez mayores errores cuadrados en SSE OOF SVM | 84,41 % | **97,35 %** |
| Sesgo medio SVM en el 25 % superior global de Li | −15,07 ppm | **−45,77 ppm** |

La desigualdad de errores es **muy alta en ambas versiones**, especialmente en el candidato; limita la interpretación de un único RMSE promedio. La coincidencia del Q75 no implica que la cola alta sea semejante.

### 4.3 Comparación sobre los mismos registros históricos: evidencia externa a esta corrida

Se reutilizó la comparación emparejada ya ejecutada, **no se calcularon nuevos resultados inventados**: `results/scientific_simulations/vnext_20261106/stage5_training_comparison/m1_paired_comparison.json`.

Sobre **1.004 observaciones emparejadas** del histórico y candidato:
- RMSE 26,091 → **21,469 ppm**, mejora exploratoria de ~17,7 %.
- MAE 8,333 → **7,258 ppm**, mejora exploratoria de ~12,9 %.
- Mediana error 4,043 → **3,883 ppm**.
- IC95 bootstrap geográfico del beneficio en RMSE **[+0,436; +10,116] ppm** y MAE **[+0,289; +2,267] ppm**.
- **610/1.004 observaciones cambiaron de outer fold** y el entrenamiento también utilizó distintos conjuntos; es comparación de desarrollo, **no prueba final en holdout congelado independiente**.

**Interpretación integrada:** es razonable llamar «mejora experimental en un subconjunto emparejado» al candidato. No está respaldado llamarlo «superior en todo el dominio», en particular para Li extremo.

### 4.4 Controles implementados o ejecutados

- Hash original y ampliado verificables; 1.120/3.420 etiquetas OOF exactamente alineadas con los conjuntos reconstruidos.
- Cinco folds espaciales identificados por checkpoints y reconstruidos con `GroupKFold`; cero solapamiento de grupos TRAIN/TEST dentro de cada fold.
- Predicción OOF del SVM refitada exactamente en **10/10 folds** a precisión de punto flotante, con parámetros de cada ejecución y sin nuevos trials Optuna.
- Curvas de aprendizaje en 5 folds×4 fracciones×2 versiones; TEST fijo y sin filtración de grupos.
- Matrices de confusión auxiliares VP/VN/FP/FN para **6 algoritmos×2 versiones**, con q75 calculado **solo en TRAIN**.
- Métricas OOF por algoritmo y por fuente; distribución de Li y contribución de los extremos al SSE.
- Robustez a ruido gaussiano del 5 % ya documentada en métricas de los entrenamientos previos; no se vuelve a inventar un nuevo resultado en este reporte.
- Modelos activos, configuraciones, menú, archivos históricos, PDFs y decisiones de release sin cambios.

### 4.5 Límites que permanecen

1. **No hay holdout geológico externo independiente** del ganador; las pruebas retrospectivas usan folds conocidos y repetidamente consultados durante desarrollo.
2. La curva TRAIN/TEST se reentrenó en los **dos SVM ganadores**, no en los seis regresores. Las matrices OOF sí cubren seis en cada versión.
3. El rendimiento por fuente del candidato puede ocultar desplazamientos geoquímicos/materiales; análisis de duplicados, distancias de exclusión y etiquetas de los extremos requieren auditar datos crudos.
4. Se seleccionaron hiperparámetros usando entrenamiento completo de cada fold antes de estudiar curvas fraccionadas; no se reoptimizó por fracción.
5. Ni la memoria ajena ni estos datos permiten declarar «porcentaje de aciertos» a partir de R² ni extrapolar universalmente a regiones sin muestreo.

## 5. Registro de hipótesis y cómo decidirlas

| Hipótesis | Estado hoy | Prueba de refutación o confirmación posterior |
|---|---|---|
| H01: las nuevas etiquetas GEOROC contienen errores de unidad o emparejamiento | **Sin verificar** | Trazar muestra, unidad, laboratorio, FeOT, lat/lon y material de 8 casos >1.000 ppm |
| H02: el candidato mejora rendimiento típico sobre históricos compartidos | **Compatible con comparación de 1.004** | Holdout geográfico congelado sin reutilizar en ajustes ni selección |
| H03: la cola extrema domina RMSE y explica cifras mayores | **Respaldada descriptivamente** | Reportar con/sin extremos **solo como sensibilidad**, nunca borrar TEST del resultado principal |
| H04: el SVM histórico presenta overfitting en ciertas regiones | **Señal parcial** | Repetir con buffers espaciales, distancias y análisis de incertidumbre regional |
| H05: el SVM ampliado subajusta sistemáticamente Li extremo | **Compatible, no universal** | Curvas de residuos estratificadas, MAE extremo y regresión robusta; evaluar RF/XGB/CatBoost sin selección retrospectiva |
| H06: más grupos de TRAIN ayudan al candidato | **Señal de curva** | Curvas con varias semillas y reoptimización anidada dentro del TRAIN de cada fracción |
| H07: un objetivo secundario de «Li alto» favorecería otro algoritmo | **Compatible con matrices** | Predefinir ponderación/recall, elegir solo con TRAIN y validar en geografía externa |
| H08: los resultados menores de RMSE de terceros reflejan al menos parcialmente su filtrado | **Plausible, no verificable directamente** | Reproducir su mismo dominio/splits/código si llegan a facilitarse; no inferir fraude/error |
| H09: la estrategia espacial explica métricas más conservadoras | **Plausible** | Contrastar aleatoria y espacial sobre nuestros mismos datos, sin usar random CV como «validación externa» |
| H10: las 77 muestras añadidas en M2 perjudicaron el release candidato M2 | **Experimento independiente, pendiente** | M2 solo 706 vs **release anterior**, manteniendo entrenamiento corregido; no mezclar con GEOROC M1 |

## 6. Decisión de release y trabajo ulterior

**Resultado de la auditoría:** hay mejoras objetivas en cobertura documental y existen ventajas experimentales del candidato sobre observaciones compartidas. **No hay prueba suficiente para activar un nuevo M1 como superior en todo el dominio**, especialmente en extremos. Conservar el release activo es una decisión de gestión de riesgo y trazabilidad, **no una declaración de que su precisión global es universalmente mejor**.

**Siguientes experimentos más informativos:** (a) auditoría de las 8 muestras >1.000 ppm con procedencia y unidades; (b) comparación pareada en holdout externo geográficamente disjunto; (c) curvas de aprendizaje de RF y XGBoost para estudiar si las debilidades del SVM son específicas del algoritmo; (d) repetición multi-semilla de selección de grupos TRAIN; (e) optimización train-only de un objetivo secundario de recall alto Li si se requiere operacionalmente.

## 7. Inventario de evidencias generadas

Carpeta `results/scientific_simulations/m1_dual_diagnostics_20261113/`:

| Archivo | Contenido |
|---|---|
| `audit_manifest.json` | Procedencia, SHA, diseño y estado |
| `regression_oof_six_algorithms.csv` | RMSE, MAE, R², sesgos, medianas y P90 OOF para 12 modelos |
| `train_test_gap_svm.csv` | Brechas TRAIN/TEST y verificación de refit en los 10 folds SVM |
| `learning_curves_svm.csv` | 40 ajustes espaciales (5 folds×4 fracciones×2 versiones) |
| `confusion_six_algorithms.csv` | Matrices agregadas y tasas para 12 algoritmos/versiones |
| `confusion_fold_train_q75.csv` | 60 matrices por fold, cutoff q75 train-only |
| `sample_errors_six_algorithms.csv` | 27.240 predicciones evaluadas con etiqueta auxiliar y error en ppm |
| `regression_by_source_svm.csv` | 4 estratos GEOROC/Mamani y versión |

**Fuentes:** datos y run files en el propio repositorio, con ruta SHA; memoria ajena «Memoria Litio (2)(5).pdf» para contraste contextual; documento compañero: `M1_DEFENSA_TESIS_METRICAS_DATOS_Y_RELEASE_20261113.md`.

**Ética de reporte:** las hipótesis anteriores se conservarán como tales hasta reproducir tests específicos; no transformar correlaciones, métricas de desarrollo o decisiones de ingeniería en afirmaciones de superioridad causal o validación independiente.
