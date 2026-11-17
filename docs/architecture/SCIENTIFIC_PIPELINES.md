# Pipelines científicos de LithiumScope

Este documento resume el contrato general de M1, M2 e integración. Las cifras por ejecución y los protocolos históricos se encuentran en [docs/models](../models/README.md); no deben extrapolarse de un entrenamiento a otro.

## Modelo 1: regresión geoquímica

**Objetivo:** estimar `Li_icpms` en ppm a partir de variables conocidas de una muestra de roca.

`data -> QC y límites de detección -> features geoquímicas/geológicas -> preprocesamiento por fold -> competencia de regresores -> evaluación OOF -> modelo persistido`

- Pasos implementados en `src/lithiumscope/model_1/steps/`.
- Las seis familias usadas incluyen Random Forest, XGBoost, SVM-RBF, TabNet, HistGradientBoosting y CatBoost.
- Entrenamiento/competencia y métricas en `model_1/training/` y `model_1/evaluation/`.
- Métricas principales: RMSE, MAE y R² en el protocolo de evaluación correspondiente.
- Baseline de comparación: DummyRegressor, sin considerarlo candidato ganador.
- No utilizar `Li_icpms` como predictor ni comparar métricas globales de poblaciones diferentes como si fueran un test pareado.

## Modelo 2: priorización espectral

**Objetivo:** discriminar sitios relacionados con concentraciones de Li relativamente altas utilizando imágenes Sentinel-2 L2A y características espectrales, **no predecir un yacimiento explotable**.

`Li medido + coordenadas -> muestra/escena -> parche multibanda -> features -> etiqueta definida en TRAIN -> CV espacial -> selección, calibración y threshold -> score`

- Conjunto histórico operativo de referencia: 706 observaciones, 126 features de esquema previo.
- Experimento espectral v4: 170 features, incluidas bandas red-edge adicionales; **no equivale al release operativo**.
- Seis familias: Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost y SVM-RBF.
- ROC-AUC y Average Precision describen discriminación; Brier y umbrales requieren contexto de calibración y protocolo.
- Las etiquetas derivadas de cuantil se calculan sin incorporar el conjunto externo de evaluación.
- Los grupos espaciales deben respetarse también al ajustar hiperparámetros y calibración cuando corresponde.
- Los candidatos con más features no son automáticamente compatibles con el extractor operativo.

Los datos GEOROC usados de forma provisional requieren control de procedencia y calidad; su presencia en una prueba **no significa validación externa certificada**.

## Integración posterior

M1 y M2 se ejecutan por separado y se relacionan por identificador de caso o candidato. La integración permite examinar concordancias, divergencias, advertencias fuera de dominio y resultados, y producir la demostración automática de diez casos mediante el menú de predicción.

No existe una dependencia `predicción M1 -> feature M2`; no declarar una mejora de precisión de ambos modelos a partir de la mera integración.

## Salidas, identidad y publicación

Cada entrenamiento queda registrado por `run_id`, configuración, splits, fingerprints, métricas y artefactos. Una ejecución experimental o un algoritmo mejor rankeado no se publica sin revisar protocolo, inferencia compatible y criterios de release.

- [Contratos de entrada/salida](../data/INPUT_OUTPUT_VARIABLES.md)
- [Datasets y características](../data/DATASETS_AND_FEATURES.md)
- [Predicción integrada](../models/PREDICTION_WORKFLOWS.md)
- [Metodología M2](../models/MODEL_2_METHODOLOGY.md)
- [Reproducibilidad](../operations/REPRODUCIBILITY.md)
- [Publicaciones](../models/MODEL_RELEASES.md)
