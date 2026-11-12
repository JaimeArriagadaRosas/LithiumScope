# LithiumScope — Ampliación controlada de datos espectrales del Modelo 2

**Fecha:** 6 de noviembre de 2026. **Rama local:** `feat/scientific-simulation-lab`. **Estado:** preparación y descarga de datos; NO se ha iniciado otro entrenamiento ni se han activado modelos.

## Decisión de alcance

El usuario decidió reservar **M1 `training_20261106_111653_m0300` (SVM con GEOROC B)** como candidato del próximo release y no volver a modificarlo. Sigue almacenado, sin sustituir el release activo de septiembre.

El entrenamiento de **M2 `training_20261106_135029_m0300` queda clasificado como experimento descartado**: el Random Forest ganador reprodujo exactamente las predicciones del día anterior sobre 706 muestras; la variante calibrada mejoró Brier pero empeoró ROC-AUC y Average Precision. Se preservan los modelos, pruebas y registros históricos para auditoría, sin promoción.

El trabajo nuevo se limita **exclusivamente a M2**. La siguiente ejecución será **`python main.py` → 1. Entrenar modelos → 2. Modelo 2**. El usuario inicia el entrenamiento: esta preparación no llama al entrenado ni instala un modelo.

## Diagnóstico de cobertura

Los 706 ejemplos M2 originales ocupan **66 grupos geográficos de 0,5°**; el mayor reúne **156 ejemplos (22,1 %)**. Este desequilibrio limita la generalización espacial.

Se identificaron **2.744 registros GEOROC B**, de los que **1.094** pasaron una revisión bibliográfica y espacial preliminar. Tras limitar Li a **0–100,6 ppm** y excluir grupos ya representados, **577 registros** quedan como candidatos, aunque constituyen **107 grupos nuevos únicos**. Para reducir dependencia de una publicación se selecciona como máximo una muestra por grupo y un cupo por publicación, con distribución geográfica multinacional y aproximadamente una etiqueta alta por cada dos bajas, según el cuartil 75 de los 706 originales.

**Aviso científico:** estas etiquetas tienen referencia bibliográfica recuperada y filtros geoquímicos, pero **no fueron validadas una por una con GPS de campo**. La ampliación puede mejorar o empeorar M2. Solo una comparación posterior permitirá determinarlo. No se representan mediciones de litio de roca como valores de todos los píxeles.

## Fuentes y transformación

1. **GEOROC**, existente en el disco y ya armonizado para M1. Se utiliza exclusivamente como fuente de concentraciones de litio y coordenadas, **no como nuevas variables del modelo espectral**. Si falta el cribado, la herramienta puede reconstruirlo a partir de GEOROC crudo y sus referencias; si falta el CSV crudo, llama al descargador GEOROC existente, utilizando una carpeta **exclusiva de M2**, sin alterar el dataset de M1.
2. **Sentinel-2 L2A**, vía Earth Search STAC (Element84), con seis bandas originales: red, green, blue, NIR, SWIR16 y SWIR22. Se extraen parches de **640 m**, 32×32 píxeles, y **126 características** utilizando el mismo extractor vigente. Se conserva el esquema de entrada de M2.
3. **Filtros:** roca íntegra y balance de óxidos de GEOROC B ya comprobados; referencias rastreables; sin perforaciones documentadas o localizaciones groseras; coordenadas válidas; Li dentro del rango histórico; grupos nuevos; máximo por publicación; nubosidad de escena ≤25%; nubosidad/sombra/nieve **dentro del parche según SCL** ≤25%; al menos 95 % de píxeles válidos; seis bandas y 126 características espectrales finitas.
4. La clasificación SCL se lee por vecino más cercano para preservar sus clases. A diferencia de la nubosidad de escena completa, se inspecciona la **zona exacta** del parche. Documentación: https://docs.sentinel-hub.com/api/latest/user-guides/cloud-masks/ y https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html.
5. **Descargas automáticas, reanudables y delimitadas**. Los parches y tablas se almacenan solo en rutas nuevas de M2; se reutilizan los que pasaron todos los controles y se registran los fallos. Nunca se sobrescriben las 706 imágenes originales.
6. En el siguiente entrenamiento **solo de M2**, el menú validará el dataset antiguo mediante SHA-256, completará las descargas pendientes si hicieran falta, combinará los pares aceptados con las 706 muestras y creará un archivo de características **separado**. Si no alcanza el mínimo seguro de **40 ejemplos nuevos**, abortará indicando el problema en lugar de entrenar silenciosamente el dataset anterior.
7. Los algoritmos de M2 compiten con datos ampliados. Se retira de la competencia ordinaria la variante RF calibrada espacialmente descartada, conservando su código para experimentos separados. El umbral de Li de cada fold continúa calculándose **solo con las muestras de entrenamiento**, y el evaluador excluye grupos compartidos entre train/test.

## Configuración y límites

Archivo: `config/model_2.yaml` → `training.expansion`.

- `enabled: true` activa la ampliación automática solo cuando se escoge M2 en el menú.
- `target_new_samples: 80` es el máximo buscado, **no una promesa** de 80 descargas de calidad.
- `minimum_new_samples: 40` limita el arranque si faltan pares supervisados suficientes.
- `max_per_publication: 10`; solo un registro por grupo espacial nuevo.
- `max_li_ppm: 100.6`, `max_scene_cloud_percent: 25`, `max_patch_cloud_fraction: 0.25`.
- El modelo resultante sigue clasificando alto/bajo litio a partir de bandas Sentinel-2. GEOROC aporta etiquetas, no características químicas adicionales.

## Directorios aislados

- Fuente cribada existente: `results/scientific_simulations/vnext_20261106/georoc_source_validation/georoc_B_screened_sites.csv`.
- Cola, estado, incidencias y resumen de descarga: `data/interim/model_2/vnext_georoc/`.
- Nuevos parches satelitales: `data/raw/model_2/vnext_georoc/patches/`.
- **Manifest ampliado exclusivamente para M2:** `data/processed/model_2/vnext_expanded/training_manifest.csv`.
- Caché de las 126 variables para la ejecución siguiente: `data/processed/model_2/vnext_expanded/spectral_features.csv`.
- El manifiesto original `data/processed/model_2/training_manifest.csv` (706 muestras) y el dataset/modelo M1 quedan inalterados.

Herramienta ejecutable para inspeccionar la selección sin descargas:

```powershell
python -m lithiumscope.model_2.data.expansion_vnext
```

Para adquirir antes del entrenamiento, con reanudación automática:

```powershell
python -m lithiumscope.model_2.data.expansion_vnext --acquire
```

Estas órdenes no entrenan modelos. Cuando el usuario elija **Entrenar solo Modelo 2**, la misma herramienta se ejecutará automáticamente con los límites configurados.

## Cómo determinar después si mejoró realmente

La prueba posterior debe comparar sobre observaciones y etiquetas comunes, no solo poner frente a frente dos medias calculadas sobre datasets distintos. Se revisarán ROC-AUC, AP, Brier, Balanced Accuracy, distribución geográfica y estabilidad por grupo, con referencia a la ejecución previa del 5 de noviembre. La nueva distribución puede cambiar los umbrales de fold, por lo que una comparación directa de scores sin igualar las condiciones puede ser engañosa. El umbral decisorio debe seleccionarse usando **datos del entrenamiento**, no optimizando el test externo.

**No se descargan nuevas variables para M1, no se toca el candidato de M1, no se crea release, no se hace push y no se ejecuta entrenamiento.**


## Preservación del protocolo de evaluación histórico

Para que el próximo resultado **sí pueda compararse** con el M2 del 5 de noviembre, se congeló la estructura de validación científica que ya usó sus **706 muestras**:

- Snapshot: `config/model_2_frozen_spatial_folds_706.json`, copiado de la ejecución histórica; guarda los índices de cinco folds.
- Implementación: `make_anchored_scientific_folds` en `src/lithiumscope/model_2/training/scientific_validation.py`, y activación solo cuando M2 recibe un manifiesto **ampliado**.
- Se conserva exactamente el fold exterior y el umbral de Li de cada prueba **para las 706 muestras originales**. Los ejemplos nuevos se distribuyen entre los cinco folds por grupo, nunca dividiendo un grupo espacial entre entrenamiento y prueba.
- El umbral alto/bajo de cada fold se determina usando **solo el conjunto histórico de entrenamiento** de ese fold. Las muestras nuevas no alteran retrospectivamente las etiquetas de evaluación anteriores.
- El próximo entrenamiento podrá compararse contra las puntuaciones OOF históricas **sobre esas mismas 706 muestras y mismos folds**. Se calcularán aparte métricas para las nuevas regiones, cuyos datos son provisionales.
- Se validó la construcción sin entrenar con 80 etiquetas candidatas: cinco folds, 16 ejemplos nuevos por fold en esa simulación, cero grupos compartidos entre train/test, umbrales originales conservados.

Esta preservación de folds no equivale a una evaluación externa nueva. Permite medir el valor incremental de las muestras adicionales sin confundir diferencias debidas a cambios de partición o etiqueta.

El run experimental rechazado queda identificado con el sidecar local `results/model_2/runs/training_20261106_135029_m0300/experimental_decision.json`; el historial no se borra ni se altera el release activo. El M1 nuevo permanece guardado sin cambios para futura decisión de release.


## Resultado final de la adquisición y prevalidación

La ejecución real terminó correctamente; el objetivo configurado de hasta 80 muestras era una cota deseada, no una condición para aceptar datos defectuosos.

| Indicador observado | Resultado |
|---|---:|
| Dataset M2 histórico conservado | 706 muestras / 66 grupos |
| Nuevos parches Sentinel-2 válidos | **77** |
| Manifiesto del próximo M2 | **783 muestras / 143 grupos** |
| Publicaciones GEOROC en las 77 aceptadas | **30** |
| Nuevas etiquetas con Li ≥21,6 ppm | **18/77** |
| Registros distintos rechazados por los filtros | 23 |
| Características espectrales verificadas | **126 por muestra**, todas finitas |

Distribución de las 77 incorporaciones: **Chile 29, Argentina 20, Perú 15, Ecuador 10, Colombia 2 y Bolivia 1**. La participación del grupo espacial original más grande baja de **22,10 %** a **19,92 %** por el aumento de cobertura, aunque sigue existiendo concentración y no se garantiza una mejora predictiva hasta la siguiente evaluación.

La caché de características ampliadas se generó para los **783** registros, sin ejecutar entrenamiento. El manifiesto histórico y los artefactos de M1 mantuvieron sus huellas SHA-256 originales:

- Manifiesto M2 histórico: `5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1`
- Dataset combinado del M1 candidato: `94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94`
- Modelo M1 candidato SVM: `a09b5f0c2018f68ae567bf89f0b4f36c4d88dbd55bc7c7cb665d2140dc2f4a77`
- Manifiesto M2 nuevo (783 filas): `764d9ab4d9f548e96f39b9a4bc843c7cb903fabdbbd0a982aa78027cc045e623`

Informe técnico reproducible: `data/interim/model_2/vnext_georoc/readiness_quality_profile.json`.

Se descartaron los fallos de descarga del flujo habitual para que al ejecutar nuevamente el menú no reintente una y otra vez las escenas que ya sabemos que fallan. Pueden reexaminarse deliberadamente con el argumento `--retry-failed`. **La aceptación de 77 parches SCL válidos no certifica por sí sola la exactitud de terreno de las coordenadas GEOROC.**

Se aumentó de forma acotada el presupuesto Optuna a **12 ensayos por defecto**, con 12 para Random Forest y 10 para Extra Trees, CatBoost, HistGradientBoosting y SVM-RBF, manteniendo la poda de ensayos. No se ejecutó ninguno de esos ensayos ni se realizó entrenamiento.

**El siguiente paso del usuario es directamente** `python main.py` → **1. Entrenar modelos → 2. Modelo 2**. Los 783 pares y su caché de 126 variables ya están preparados; el menú revalidará los ficheros y reutilizará las descargas correctas.
