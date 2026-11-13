# Modelo 1 de LithiumScope: defensa de métricas, datos, metodología y versiones

**Fecha de corte:** 13 de noviembre de 2026. **Uso:** defensa de tesis, estudio personal y trazabilidad de resultados.

**Fuentes:** memoria *Predicción de concentraciones de litio mediante modelos de aprendizaje automático aplicados a datos geoquímicos y geoespaciales* de Matías Marcel Alarcón Arias y Samir Nasim Saud Rauld (2026), PDF «Memoria Litio (2)(5).pdf»; artefactos verificables de LithiumScope.

**Principio:** distinguir estrictamente lo **documentado por terceros**, lo **observado en LithiumScope** y las **hipótesis por comprobar**. No se dispone del código, modelo entrenado, checkpoints ni OOF originales de los autores de la memoria.

## 1. Resumen para exposición oral

> Se tomó como referencia documental una investigación anterior sobre predicción de concentraciones de litio, sin disponer de su implementación ni de sus modelos. En LithiumScope se reprodujo conceptualmente la tarea de regresión sobre Li_icpms, se desarrolló una implementación propia y se amplió el alcance hacia validación espacial, trazabilidad, múltiples fuentes geoquímicas, análisis de incertidumbre y robustez. Las cifras de desempeño bruto no son directamente comparables, porque el estudio de referencia filtró la cola de concentraciones de litio y evaluó 681 muestras, mientras nuestros experimentos conservaron valores extremos y utilizaron 1.120 y posteriormente 3.420 observaciones. La ampliación mejoró resultados en registros históricos emparejados, pero todavía no justifica promover automáticamente el candidato por falta de una validación externa independiente y por la debilidad en concentraciones extremas.

Esta formulación **no sostiene que los compañeros hayan cometido errores ni que sus métricas sean falsas**. Reconoce una replicación conceptual de alcance extendido y sus límites de reproducibilidad.

## 2. Lo que dice efectivamente el PDF de referencia

- **Procedencia y objetivo:** predicción por regresión de `Li_icpms` en ppm con variables geoquímicas, geológicas y geoespaciales (introducción y metodología).
- **Procesamiento:** el PDF declara 2.635 registros iniciales, 787 con Li válido en una etapa y **681 registros finales** tras filtros; aplica percentiles **2,5–97,5 de Li** y SUM(no water) en **94–102** (pp. 51–53).
- **Validación:** K-Fold **5 externos × 5 internos**, mezclado aleatorio con semilla 42 para particiones, búsqueda Optuna; pp. 70–75. El apartado de reproducibilidad declara **100 trials por fold externo** y que la creación del sampler Optuna **no fijaba una semilla propia** (p. 110).
- **Resultados — Tabla 20, p. 135**, *promedio ± desviación estándar de los cinco folds*:


  | Algoritmo | R² medio | RMSE medio (ppm) |
  |---|---:|---:|
  | Random Forest | 0,499 ± 0,086 | 5,361 ± 0,595 |
  | XGBoost | 0,467 ± 0,087 | 5,528 ± 0,589 |
  | TabNet | 0,399 ± 0,150 | 5,828 ± 0,650 |
  | SVM | 0,348 ± 0,086 | 6,124 ± 0,536 |
  | API GPT | −0,157 | 6,934 |

  GPT utiliza **50 casos con otra estrategia** y no debe incluirse como fold equivalente a los cuatro anteriores.
- En pp. 136–144 se describen limitaciones de tamaño, cobertura espacial y generalización. La memoria no aporta resultados experimentales completos de brechas TRAIN/TEST, curvas de aprendizaje por grupos ni VP/VN/FP/FN para cada regresor.

**Reproducibilidad ajena:** el PDF permite conocer el diseño declarado y sus resultados, **no verificar que obtendríamos exactamente esos números**, porque faltan el código, modelos, fuentes exactas de versión, splits originales y salidas OOF ejecutables. La ausencia de estos artefactos disponibles para nosotros **no demuestra** que no existieran o que los autores no los conservaran.

## 3. Nuestra implementación y sus versiones verificadas

| Propiedad | Protocolo de referencia (PDF) | M1 release activo | M1 candidato ampliado |
|---|---|---|---|
| Fuente verificable | Memoria escrita, sin bundle/código disponible | `models/model_1/trained/training_20260925_111245_m0300` | `models/model_1/trained/training_20261106_111653_m0300` |
| Estado | Publicación académica de terceros | **Activo en el registro** | **Pendiente, no activado** |
| Muestras evaluadas | 681 | **1.120** | **3.420** |
| Fuente de datos | Dataset depurado del estudio | 676 Mamani09 + 444 GEOROC | 676 Mamani09 + 2.744 GEOROC |
| Grupos espaciales | K-Fold aleatorio declarado | GroupKFold de **100** grupos | GroupKFold de **244** grupos |
| Predictores | Químicos, geológicos, espaciales | **33** | **33** |
| Variable objetivo | Li continuo en ppm, extremos filtrados | Li continuo en ppm, **sin recortar colas por percentil** | Li continuo en ppm, **sin recortar colas por percentil** |
| Familias estudiadas | RF, XGB, SVM, TabNet + GPT aparte | RF, XGB, SVM, TabNet, HGB, CatBoost | Las mismas seis |
| Ganador del procedimiento | RF | SVM-RBF | SVM-RBF |
| Evidencia conservada | Tablas y figuras en PDF | Bundle, hash, métricas OOF, folds, checkpoints | Idem, más experimento con FeOT armonizado |
| Holdout externo independiente del ganador | No reproducible desde el PDF | No certificado | No certificado |

El registro `models/active_models.json` mantiene activo el **release de septiembre**, no el entrenamiento de noviembre. Para hacer una auditoría retrospectiva del release usamos el **run de 05/11/2026** `training_20261105_204029_m0300`: reconstruimos su fuente bit a bit con SHA-256 **`2f99d38960c31a02b85aa5328c910c3aeab5ec1760cf6ef035922fbb56d41ea1`**, igual a la fuente del release; las etiquetas/orden de OOF coinciden. **No se debe confundir un refit reproducido con demostrar que cada predicción OOF pertenece físicamente al binario publicado.**

Para el candidato se verificó SHA-256 de la fuente `94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94` y sus 3.420 etiquetas OOF. En ambos casos la versión evaluada se apoyó en predicciones almacenadas y se reconstruyó el SVM de cada fold a partir de sus parámetros congelados.

## 4. Por qué el RMSE y R² de terceros NO permiten declararlos mejores

### 4.1 Qué representa cada métrica

- **RMSE**, ppm: `sqrt(media((y_real−y_pred)²))`. Penaliza cuadráticamente errores grandes. Su escala **depende de la distribución de valores de Li**; no es una tasa de acierto ni porcentaje. La media de los RMSE de folds tampoco equivale necesariamente al RMSE OOF combinado.
- **R²**, sin unidades: `1−Σ(error²)/Σ((y−media(y))²)`. Un R² de 0,499 significa aproximadamente que explica el 49,9 % de la **variación de la variable objetivo respecto al predictor de media dentro de esa evaluación**, **no** que acierta el 49,9 % de las muestras. Puede ser negativo y cambia con la variabilidad del conjunto TEST.
- **MAE**, ppm: media del error absoluto, menos dominada por extremos que RMSE. **Mediana de error absoluto** y **P90** ayudan a distinguir muestra típica y cola difícil.
- **R² medio de folds**, **R² global de OOF**, **RMSE medio de folds** y **RMSE global de OOF** son agregaciones distintas; siempre explicitar cuál se muestra.

### 4.2 Distribución observada de las dos versiones de LithiumScope

| Estadística de `Li_icpms` | Release / histórico (1.120) | Candidato (3.420) |
|---|---:|---:|
| Mediana | 16,1 ppm | 15,0 ppm |
| Q75 | 21,8 ppm | 21,60 ppm |
| Q95 | 40,50 ppm | 44,0 ppm |
| Q99 | 98,97 ppm | **240,0 ppm** |
| Máximo | 597,1 ppm | **4.160,0 ppm** |
| Desviación estándar | 26,91 ppm | **163,40 ppm** |
| Filas con Li >100 ppm | 12 | **89** |
| Filas con Li >1.000 ppm | 0 | **8** |

**Hecho comprobado:** el candidato tiene una cola alta muchísimo más amplia, a pesar de compartir medianas similares. Los diez casos de mayor error cuadrático del SVM candidato concentran **97,35 %** de su SSE OOF (en el histórico, **84,41 %**). Son distribuciones de dificultad distinta.

**Hipótesis plausible, NO demostrada para el modelo de terceros:** su filtro del 95 % central de Li puede ayudar a explicar RMSE menores. No conocemos sus predicciones sobre extremos descartados; tampoco sería correcto afirmar que sus resultados son erróneos o inventados. Un buen experimento requeriría aplicar protocolos idénticos sobre un mismo conjunto de evaluación y regiones equivalentes.

### 4.3 Números propios, siempre con la misma definición

| Comparación descriptiva | Release histórico 1.120 | Candidato 3.420 |
|---|---:|---:|
| SVM RMSE medio de 5 folds | 20,986 ppm | 87,449 ppm |
| SVM R² medio de 5 folds | 0,070 | **0,247** |
| SVM RMSE OOF combinado | 25,685 ppm | 159,846 ppm |
| SVM R² OOF combinado | **0,0886** | 0,0430 |
| SVM MAE OOF | **8,128 ppm** | 16,320 ppm |
| SVM mediana de error absoluto OOF | 3,725 ppm | **3,657 ppm** |
| SVM P90 error absoluto OOF | **14,485 ppm** | 16,680 ppm |

Son cifras **verificadas**, pero referidas a dominios distintos. La mediana del error del candidato mejora ligeramente, mientras el error de cola extrema domina RMSE. El R² medio por fold mejora, mientras el R² global OOF empeora. Eso **no es una contradicción**, sino consecuencia de distribuciones y operaciones de agregación diferentes.

**Fallo de razonamiento que debemos evitar:** «más filas garantizan mejores métricas». Lo que aumenta es el soporte para estudiar un problema más heterogéneo; su rendimiento puede bajar, mejorar o permanecer igual.

## 5. Evidencia de mejora del candidato sobre observaciones compartidas

El estudio experimental `results/scientific_simulations/vnext_20261106/stage5_training_comparison/m1_paired_comparison.json` emparejó **1.004 observaciones físicas** entre ejecuciones:

| Métrica, mismos 1.004 registros | Histórico | Candidato |
|---|---:|---:|
| RMSE | 26,091 ppm | **21,469 ppm** |
| MAE | 8,333 ppm | **7,258 ppm** |
| Mediana error absoluto | 4,043 ppm | **3,883 ppm** |
| P90 error absoluto | 15,566 ppm | **14,170 ppm** |
| Sesgo (predicho − real) | −1,228 ppm | **−0,799 ppm** |

Reducciones descriptivas: **17,7 % de RMSE y 12,9 % de MAE**. En bootstrap agrupado exploratorio, la ganancia RMSE tuvo IC95 **[+0,436; +10,116] ppm**; MAE **[+0,289; +2,267] ppm**.

**Límite clave:** 394 casos compartían fold; **610 cambiaron de fold** y también cambió la muestra disponible durante entrenamiento. Por ello este estudio apoya una **mejora experimental sobre observaciones comunes**, **no** una superioridad independiente validada con un único holdout congelado.

## 6. Por qué mantenemos el release activo

1. Está identificado, es reproducible desde una fuente con hash y existe un artefacto publicado activo.
2. El candidato ampliado ofrece resultados alentadores en el subconjunto emparejado, pero **todavía no ha demostrado superioridad frente al release en evaluación independiente** bajo la misma condición de TEST.
3. El candidato tiene deficiencias importantes en la cola de concentración alta y sensibilidad espacial; promoverlo basándose solo en algunas métricas medias ocultaría ese riesgo.
4. No es obligatorio sustituir una versión desplegada por una candidata de investigación. La promoción exige trazabilidad, pruebas y un criterio explícito de riesgo.
5. **Mantener el release NO significa afirmar que su precisión es mejor en todas las regiones**; significa que no se ha acreditado suficientemente el reemplazo.

La decisión futura puede ser aprobar un nuevo release **con limitaciones declaradas** cuando haya evidencia sólida para su uso previsto; o conservar el vigente si esa evidencia no aparece. No debe afirmarse que ya existe validación geológica de campo o generalización universal.

## 7. Ventajas demostrables y límites frente a la memoria

| Eje | Comparación defendible |
|---|---|
| Volumen y alcance de datos | **Ventaja de cobertura:** 3.420 casos frente a 681, pero no prueba calidad superior de todas las filas |
| Diagnóstico por fuentes/cola y residuos | **Ventaja documental y experimental:** tablas OOF, grupos, errores extremos y pruebas de robustez |
| Control espacial | **Diferencia metodológica con menor riesgo de mezcla local:** GroupKFold espacial frente a K-Fold aleatorizado descrito |
| Trazabilidad | **Ventaja verificable:** hashes de dataset, checkpoint por fold, esquema, parámetros y OOF reutilizables |
| Matriz auxiliar de alta concentración | **Ventaja de documentación:** VP/VN/FP/FN por fold con q75 TRAIN-only, ausente en el PDF |
| Curvas y brecha TRAIN/TEST | **Ventaja de evidencia propia:** reproducidas para SVM histórico y candidato, no documentadas cuantitativamente en el PDF |
| Predicción en ppm sobre mismo TEST del trabajo ajeno | **NO acreditada:** faltan su modelo, datasets/splits y sus OOF |
| Superioridad predictiva global del candidato | **NO acreditada:** discrepancias espaciales, cola alta y falta de holdout externo |

**No confundir «más experimentos documentados» con «mayor precisión científica demostrada».** La memoria ajena es una referencia técnica, no un benchmark ejecutable bajo nuestro control.

## 8. Preguntas de defensa y respuestas preparadas

**¿Por qué sus compañeros lograron RMSE ≈5,36 ppm y ustedes 20–160 ppm?**

Porque no usamos el mismo conjunto ni el mismo protocolo. Su memoria describe 681 muestras con exclusión del 5 % extremo y particiones K-Fold aleatorias. Nuestra evaluación conserva extremos muy altos —hasta 4.160 ppm— y retiene regiones espaciales completas como test. Nuestros RMSE no miden la misma dificultad experimental. Esto **no autoriza afirmar** que nuestra precisión es superior a la suya.

**¿Qué demuestra que efectivamente reconstruyeron los datos históricos?**

El SHA-256 exacto de la fuente del release y el orden idéntico de los 1.120 valores de Li de los OOF archivados; además las predicciones de los cinco folds SVM se reprodujeron con diferencia numérica máxima de orden 10⁻¹⁴.

**¿Su R² no es demasiado bajo?**

El R² mide varianza explicada en la distribución evaluada, no «porcentaje de muestras acertadas». No ocultamos que es limitado en varios folds, especialmente en dominios con litio extremo. Por eso mostramos RMSE, MAE, cuantiles de error, regresión por origen y pruebas espaciales. El objetivo es apoyo preliminar, no sustituir ICP-MS.

**¿Por qué no activaron automáticamente el modelo de 3.420 muestras?**

Porque el conjunto nuevo es más heterogéneo; mejora sobre 1.004 registros emparejados pero su generalización a nuevas regiones y extremos sigue incierta. Mantener estable el modelo desplegado evita afirmar una mejora no demostrada.

**¿Por qué no hay un código reproducible de sus compañeros?**

Recibimos su memoria en PDF, no su repositorio ni los estimadores entrenados. Reproducimos el propósito y el procedimiento descrito tanto como fue posible; los límites impiden afirmar que verificamos o mejoramos su misma implementación.

**¿Hubo overfitting o underfitting?**

La auditoría de los cinco folds para ambos SVM encuentra brechas de generalización heterogéneas y baja capacidad explicativa de la cola alta. Existen señales de sensibilidad al dominio y subestimación de extremos. No corresponde diagnosticar ambos fenómenos como causa única sin examinar brechas, distribuciones y curvas; el detalle está en el segundo documento.

**¿Por qué una matriz de confusión en un modelo regresor?**

Como diagnóstico **auxiliar** de detección de concentración alta, sin alterar la salida continua en ppm. El q75 se calcula solo en entrenamiento para evitar la fuga del test.

## 9. Hipótesis registradas, no convertir en hechos antes de verificarlas

- **H-M1-01:** los extremos de Li y la heterogeneidad GEOROC explican parte importante del RMSE alto. **Respaldada descriptivamente**, no causalmente aislada.
- **H-M1-02:** la ampliación con FeOT equivalente mejora predicción sobre algunos registros históricos compartidos. **Respaldada exploratoriamente** por el emparejamiento de 1.004 casos; requiere TEST común fijo.
- **H-M1-03:** diferencias en estrategia de CV (aleatoria vs espacial) explican parte de las cifras del PDF. **Plausible, no probada** para terceros por falta de sus splits.
- **H-M1-04:** el candidato no recupera adecuadamente valores excepcionales de Li incluso cuando ajusta el resto. **Respaldada** por residuos extremos, requiere estudios específicos de la cola.
- **H-M1-05:** incrementar datos puede mejorar la mediana del error, pero empeorar la métrica global si se amplía el rango de Li. **Compatible con la evidencia**, no garantía general.
- **H-M1-06:** hay riesgo de dependencia espacial residual aun con celdas de 0,5°. **Sin descartar**; ensayar buffers/distancias y holdout externo.
- **H-M1-07:** errores de etiquetas/unidades/procedencia explican parte de la cola alta. **No confirmado**; auditar trazabilidad de los ocho casos >1.000 ppm antes de modificar datos.
- **H-M2-77 (línea separada):** las 77 muestras GEOROC/Sentinel-2 de M2 podrían perjudicar su calidad por emparejamiento espectral o etiquetas. **No concluida**; su prueba adecuada es el contraste 706 vs release previo, manteniendo correcciones y registrando cambios. **No confundir esas 77 muestras con los 2.300 nuevos registros GEOROC del M1**.

## 10. Reglas de integridad y material de consulta

No atribuir a terceros experimentos que no aparecen en su PDF, ni asumir sus datos o códigos inaccesibles como «malos». No presentar cross-validation reutilizada como externa. No afirmar que un coeficiente R² equivale a porcentaje de aciertos. No confundir el **run histórico de noviembre** con el **binario publicado de septiembre**, aunque compartan fuente. No sustituir el release sin autorización.

**Evidencia local:**

- `results/model_1/runs/training_20261105_204029_m0300/` (histórico 1.120).
- `results/model_1/runs/training_20261106_111653_m0300/` (candidato 3.420).
- `results/scientific_simulations/vnext_20261106/stage5_training_comparison/m1_paired_comparison.json`.
- `results/scientific_simulations/m1_dual_diagnostics_20261113/`: matriz, curva, generalización y auditoría de hashes.
- Código reproducible: `scripts/scientific/m1_dual_release_diagnostics.py`.
- Fuente ajena: PDF entregado «Memoria Litio (2)(5).pdf», especialmente pp. 51–53, 70–75, 110–112 y 135–144.

**Dictamen de defensa:** «LithiumScope supera el alcance documental y experimental demostrable de la referencia que tenemos, pero no afirmamos una superioridad predictiva directa sobre un modelo ajeno que no podemos ejecutar ni validar. Se desarrolló una réplica conceptual propia con mayor cobertura y diagnósticos científicos verificables; el candidato mejoró ciertos casos compartidos, mientras los extremos y el holdout externo siguen siendo límites explícitos».
