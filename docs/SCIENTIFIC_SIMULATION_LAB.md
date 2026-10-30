# Laboratorio de simulaciones científicas (primera iteración)

## Alcance realizado
Módulo `src/lithiumscope/experiments/scenarios.py` para ejecutar escenarios de inferencia del modelo M1 ya entrenado, sin ajustar parámetros ni alterar los bundles. Incluye baseline, pérdidas aleatorias (5, 10, 25, 50, 75 y 100%), pérdidas por grupos de variables, ruido y factores extremos. Produce `scenario_metrics.csv` y `scenario_manifest.json`.

## Advertencias metodológicas
- Las perturbaciones sintéticas prueban estabilidad; no equivalen a validación científica externa.
- Errores MAE/RMSE solo deben interpretarse si `Li_icpms` procede de mediciones independientes y comparables.
- Un caso puede producir predicción numérica, pero carecer de validez geológica o estar fuera de dominio.
- Los escenarios extremos pueden violar balances geoquímicos. En próximas iteraciones deben introducirse perturbaciones físicamente plausibles.
- Los resultados obtenidos al revisar un conjunto no deben utilizarse para optimizar sobre ese mismo conjunto y volver a denominarlo evaluación ciega.

## Uso desde Python
```python
from pathlib import Path
import pandas as pd
from lithiumscope.persistence.load_model import load_latest_model
from lithiumscope.experiments.scenarios import evaluate_model_1

bundle, path = load_latest_model("model_1")
frame = pd.read_csv("examples/integrated_demo/cases.csv")
output = evaluate_model_1(bundle, frame, Path("results/scientific_simulations/m1"))
print(output)
```

## Próximas etapas no implementadas todavía
1. Perturbaciones reales sobre imágenes multibanda de M2, incluyendo bandas ausentes, nodata, nubosidad y cambios de escena.
2. Evaluación leave-one-region/salar-out con muestras independientes y sin contaminación espacial.
3. Conector de adquisición y catalogación verificable de capas y registros de SERNAGEOMIN/ArcGIS; no inferir disponibilidad de químicas a partir de geometrías.
4. Comparaciones científicas por material: muestras de roca frente a salmueras en conjuntos separados.
5. Informes y curvas de degradación con intervalos de confianza.
6. Gates de aptitud separados por modelo y dominio; nunca promoción automática.

## Iteración M2 implementada
El módulo `experiments/imagery.py` implementa escenarios de perturbación de parches multibanda: pérdida de píxeles, banda completa sin datos, ruido y brillo. El procesamiento respeta el esquema de bandas y extractor guardado en el bundle. Si una banda queda totalmente sin datos, marca abstención en vez de puntuar un parche artificialmente rellenado. En otros casos compara el cambio de score y el perfil de aplicabilidad.

Ejemplo (requiere un artefacto M2 entrenado localmente):
```python
from pathlib import Path
from lithiumscope.persistence.load_model import load_latest_model
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.experiments.imagery import evaluate_model_2_image
bundle, _ = load_latest_model("model_2")
image = load_imagery(Path("ruta/a/parche.npy"))
evaluate_model_2_image(bundle, image, Path("results/scientific_simulations/m2"))
```

## Estado de validación de esta copia
- 11 pruebas nuevas aprobadas.
- Ruff sin observaciones.
- No se ejecutaron predicciones reales: el clon remoto solo contiene archivos `.gitkeep` en `models/model_1/trained` y `models/model_2/trained`.
- Los modelos y datasets originales no fueron alterados, no se reentrenó y no se publicó nada.
- La integración con ArcGIS y las evaluaciones territoriales continúan pendientes.
