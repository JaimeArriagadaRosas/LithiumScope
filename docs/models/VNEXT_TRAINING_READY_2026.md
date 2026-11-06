# LithiumScope — listo para el próximo entrenamiento vNext

**Fecha:** 6 de noviembre de 2026. **Rama:** `feat/scientific-simulation-lab`. **Estado:** preparado y verificado, entrenamiento NO ejecutado.

## Decisión y responsabilidades

El usuario ejecutará personalmente el entrenamiento desde el menú interactivo. ChatGPT solo deja listas las modificaciones, controles, pruebas, documentación y commits **locales**, sin push, sin entrenar, sin instalar nuevos modelos, sin activar candidatos y sin publicar releases.

Se declara **cerrada la etapa 4** del plan original. La búsqueda de fuentes nuevas, la revisión bibliográfica extensa de GEOROC y el piloto de etiquetas de imágenes para M2 dejan de ser bloqueadores y quedan como posible trabajo futuro, según los resultados del entrenamiento siguiente.

## Modelo 1 — protocolo B listo

- GEOROC se adquiere mediante el descargador automático existente si falta el CSV, como antes.
- La ingestión incorpora `iron_total_mode: feot_equivalent` en `config/model_1.yaml`. Para filas con FEOT, reconstruye Fe2O3 total equivalente mediante `FEOT × 1.111348` **sin sumar dos formas del hierro**; donde no hay FEOT conserva el Fe2O3 original. Registra `Fe_total_provenance`.
- La ruta es reversible: `iron_total_mode: original` recupera la convención anterior. Ni la descarga cruda ni los demás algoritmos fueron alterados.
- Continúan las condiciones de material WHOLE ROCK, Li disponible, 10 óxidos y cierre de calidad 94–102 ppm equivalentes en porcentaje de óxidos (94–102 % en peso, no ppm); sin imputar datos ficticios.
- La prevalidación aislada, ejecutada **sin ajustar ningún modelo**, reconstruyó exactamente **3.420 muestras**, con **2.744 GEOROC + 676 Mamani09**, **244 grupos espaciales** y **33 predictores**. Los datos originales no se modificaron.
- Se mantiene competencia completa RF, XGBoost, SVM, TabNet, HGB y CatBoost, con CV espacial y resultados de diagnóstico.
- La ganancia de B observada en desarrollo, 20,78 a 19,40 ppm de RMSE, es **hipótesis**, no garantía de mejora del próximo entrenamiento ni de release.

## Modelo 2 — 706 muestras y calibración espacial

- No se incorporan etiquetas GEOROC ni nuevas imágenes. Se siguen utilizando exclusivamente seis bandas Sentinel-2 transformadas en las **126 variables espectrales ya existentes**.
- El menú reutiliza el manifiesto **congelado** de 706 muestras, y verifica tanto el número de filas como su huella SHA-256. **No vuelve a construir el dataset a partir de M1** al elegir M2, evitando una posible descarga masiva y la entrada accidental de los datos GEOROC nuevos de M1.
- Huella SHA-256 del manifiesto congelado: `5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1`.
- Se verificó la presencia local de **706/706 imágenes** y 66 grupos espaciales; los cinco folds exteriores permiten subdivisiones internas por grupos con ambas clases para calibración.
- Se añade `random_forest_spatial_calibrated` a la competencia, con RF de base regularizado (160 árboles, profundidad 9 y mínimo 4 muestras por hoja) y calibración sigmoidal dentro de los grupos de entrenamiento. La calibración no puede usar folds aleatorios de sustitución ni etiquetas del test exterior.
- El algoritmo sigue compitiendo con RF original, Extra Trees, HGB, XGBoost, CatBoost y SVM-RBF. La variante nueva tiene 4 pruebas Optuna para controlar coste; los demás algoritmos mantienen sus configuraciones.
- La aumentación espectral anterior NO se incorpora: redujo ROC-AUC/AP en desarrollo.
- El valor de Li que define «alto» sigue obteniéndose solo del conjunto de entrenamiento de cada fold exterior; el corte operativo final continúa marcado como exploratorio porque se selecciona en OOF y no en holdout externo.

## Cómo ejecutar el entrenamiento (usuario)

En la carpeta del repositorio:

```powershell
python main.py
```

Después elegir **1. Entrenar modelos → 3. Entrenar ambos**.

El orden será **M1 con GEOROC B**, después **M2 con las 706 muestras espectrales existentes**. El usuario inicia y controla la ejecución; este trabajo de preparación **no ejecutó la opción 3 ni otra competencia de entrenamiento**.

Los candidatos se guardan versionados sin activarse automáticamente. No usar la opción de carga/activación de modelos como parte de esta preparación.

## Después de que el usuario entrene

1. Registrar identificadores de ejecución, hashes, versiones, ganadores y métricas por fold.
2. Comparar M1 con el protocolo A y con el release anterior sobre pruebas equivalentes; reportar RMSE, MAE, sesgo por fuente/litio alto y rango de incertidumbre. Buscar aproximadamente ≥5 % de mejora de RMSE sin degradación por regiones. La estadística es exploratoria mientras no exista holdout externo verdaderamente independiente.
3. Comparar M2 con CatBoost histórico y con el RF nuevo: ROC-AUC, AP, balanced accuracy con umbral pertinente, Brier y robustez. Una mejora sustancial orientativa sería ≥0,03 ROC-AUC y ≥0,02 AP sin empeorar considerablemente Brier o estabilidad espacial.
4. Si M1 mejora y M2 no, conservar M1 como candidato de release y el M2 histórico sin sustituir. Si ambos mejoran convincentemente, considerar ambos para el siguiente release mediante revisión explícita. Si M2 no mejora, su ampliación con etiquetas validadas y nuevas imágenes queda propuesta **opcional**, no una tarea obligatoria.
5. No comparar métricas de validaciones incompatibles ni declarar generalización externa con el mismo conjunto que sirvió para seleccionar hiperparámetros.

## Evidencias previas y trazabilidad

- Plan original: `docs/models/INTEGRAL_TRAINING_VNEXT_2026.md`.
- Ensayos B y M2: `results/scientific_simulations/vnext_20261106/stage2_m1/` y `stage3_m2/`.
- Prevalidación de fuentes **sin entrenamiento**: `results/scientific_simulations/vnext_20261106/stage5_pretrain_readiness/pretrain_readiness.json`.
- El piloto bibliográfico GEOROC–Sentinel para M2 permanece como antecedente, fuera de este ciclo.
- Los archivos de resultados y datasets de las pruebas se mantienen locales; no se incluyen en commits ni se eliminan archivos históricos.

**Condición de cierre de esta preparación:** pruebas de código y prevalidación de datasets satisfactorias; ningún entrenamiento oficial ejecutado y ningún push.
