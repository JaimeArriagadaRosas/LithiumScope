# Predicción e interpretación científica

El menú de predicción separa cuatro flujos:

1. **Modelo 1 — Predicción de concentración**: estima `Li_icpms` para datos tabulares compatibles.
2. **Modelo 2 — Prospectividad espacial**: calcula prioridad relativa para una imagen multibanda.
3. **Predicción completa**: solicita los inputs del científico para ambos modelos y genera una ejecución integrada.
4. **Demostración integrada automática**: utiliza el conjunto externo versionado en `examples/integrated_demo/`.

## Separación científica

Los modelos no se encadenan. La predicción del Modelo 1 nunca se utiliza como
feature de entrada del Modelo 2.

La integración ocurre **después** de que cada modelo produce su propia salida.
Cuando los casos pueden emparejarse por `case_id`, LithiumScope calcula:

- métricas externas cuando existe `Li_icpms` real;
- Pearson y Spearman entre Li real, Li predicho y score espacial;
- error absoluto del Modelo 1 frente al score del Modelo 2;
- concordancia entre la predicción de concentración relativamente alta y la
  señal espacial/espectral;
- aplicabilidad fuera de dominio de ambos modelos.

Una correlación se presenta como asociación, nunca como causalidad.

## Interpretación humana reproducible

La explicación en consola y en `interpretation.txt` es determinista. Se construye
a partir de valores observados, umbrales documentados, aplicabilidad y
concordancia. No intenta reconstruir un razonamiento interno del estimador.

El Modelo 1 responde a la estimación de concentración de una muestra compatible.
El Modelo 2 responde a prioridad de revisión espacial. Su score **no** es
concentración de litio ni probabilidad de yacimiento.

Cuando ambos modelos entregan evidencia relativa concordante, LithiumScope puede
identificar casos con mayor prioridad para revisión o nuevo muestreo dentro del
conjunto analizado. La salida sigue requiriendo validación geológica, geoquímica,
mineralógica y/o de laboratorio.

## Artefactos

Cada ejecución completa o demostración genera:

```text
results/predictions/<run_id>/
├── prediction_manifest.json
├── interpretation.txt
├── evaluation.xlsx
├── report.pdf
├── training_vs_external.csv
├── model_1/
│   ├── predictions.csv
│   └── diagnostics.json
├── model_2/
│   ├── predictions.csv
│   └── diagnostics.json
└── cross_model/
    ├── joined_cases.csv
    ├── correlations.csv
    ├── concordance.csv
    └── figures/
```

El manifest conserva hashes de modelos e inputs, run IDs, métricas, entorno de
ejecución y procedencia del conjunto de demostración.

## Informe PDF

Las opciones 3 y 4 generan un informe PDF reproducible y lo intentan abrir con
el navegador web predeterminado del sistema. El informe incluye:

- trazabilidad de los dos modelos y sus runs de entrenamiento;
- interpretación científica general;
- métricas externas y comparación contra entrenamiento;
- correlaciones y concordancias;
- todas las gráficas integradas;
- tabla resumen de los casos;
- una sección por caso con Li real reservado para evaluación, predicción M1,
  intervalo, aplicabilidad, score/prioridad M2, escena Sentinel-2 y las
  interpretaciones de M1, M2 e integración;
- advertencias de alcance, tamaño muestral y dominio.

El Li real de una demostración se utiliza exclusivamente como verdad de
referencia para evaluación: no se entrega como feature de entrada de los modelos.

## Publicación de una demostración

Una demostración con PDF puede empaquetarse localmente sin crear tags:

```bat
python -m lithiumscope.prediction.release_demo --run demonstration_YYYYMMDD_HHMMSS_m0300
```

El comando genera un ZIP y un archivo `.sha256` bajo
`results/release_candidates/`. El bundle sanea rutas locales de los artefactos
de texto y elimina hostname/PID del manifest de publicación. La creación de un
tag o GitHub Release sigue siendo una decisión manual posterior.


### Regenerar PDF de una ejecución anterior

Una ejecución histórica que todavía tenga `report.html` puede recibir el nuevo
informe PDF sin volver a descargar Sentinel-2 ni recalcular predicciones:

```bat
python -m lithiumscope.prediction.rebuild_pdf --run demonstration_YYYYMMDD_HHMMSS_m0300
```

El comando reconstruye el PDF desde los CSV, diagnósticos, manifest,
interpretación y figuras ya guardados, y registra `report_pdf` en el manifest.


## Validación científica de candidatos

Los entrenamientos nuevos de Modelo 1 prefieren validación espacial agrupada
cuando las coordenadas permiten construir suficientes grupos de 0,5 grados.
Los grupos se respetan tanto en los folds externos como en la optimización
interna de hiperparámetros. Si faltan coordenadas válidas o grupos suficientes,
el run registra explícitamente el fallback a KFold aleatorio.

Modelo 2 conserva ROC-AUC y Average Precision como métricas de ranking. La
conversión del score a clase operativa ya no depende obligatoriamente de 0,5:
los entrenamientos nuevos seleccionan un `operating_threshold` usando únicamente
predicciones OOF del conjunto de entrenamiento y optimizando Balanced Accuracy.
El threshold, su métrica y sus resultados quedan guardados en metadata y en el
bundle del modelo. Los modelos históricos que no contienen este campo mantienen
0,5 como fallback compatible.

La prioridad exploratoria `baja/media/alta` y la clase binaria operacional son
conceptos distintos. Ninguna de las dos convierte el score de Modelo 2 en
probabilidad de yacimiento o concentración de litio.


## Candidatos y promoción de modelos

Completar un entrenamiento ya no reemplaza automáticamente el modelo activo.
El ganador de la competencia se guarda como candidato fechado y recibe un
`candidate_status.json` con estado `pending`.

Desde **Métricas y resultados -> Evaluar/promover candidatos con demo
versionada**, LithiumScope compara el candidato pendiente más reciente contra el
modelo actualmente activo usando los diez casos externos versionados. Debido a
que este conjunto ya ha sido inspeccionado durante el desarrollo, se declara
explícitamente como **conjunto externo conocido de aceptación/regresión**, no
como validación ciega nueva.

Para Modelo 1 la promoción exige no empeorar RMSE, MAE ni R2 externos y mejorar
estrictamente al menos una de esas métricas. Para Modelo 2 se exige conservar o
mejorar ROC-AUC, Average Precision y Balanced Accuracy externas, con mejora
estricta en al menos una y el mismo umbral de Li usado para definir la etiqueta
de referencia. Solo un candidato que supera su gate cambia
`models/active_models.json`. La promoción local no crea tags ni GitHub
Releases.


## Experimentos de mejora posteriores al gate externo

Modelo 1 compara el target original para todos los algoritmos y una variante
`log1p` únicamente para Random Forest, SVM-RBF y CatBoost. La transformación se
ajusta dentro de cada fold y la predicción se devuelve automáticamente a ppm
antes de calcular RMSE/MAE/R2. Por tanto, la competencia sigue comparando todos
los candidatos en la escala científica original. Cada combinación
algoritmo/target usa checkpoints y artefactos separados.

Modelo 2 usa el extractor espectral 3. Además de las estadísticas ya existentes,
incorpora BSI, MNDWI, NBR2, un índice de suelo seco y descriptores de textura
por banda. Estos atributos describen contexto superficial y heterogeneidad del
parche Sentinel-2; no se interpretan como detección directa de litio. El cambio
de versión invalida automáticamente el cache anterior de features y queda
registrado en metadata y en el bundle entrenado.
