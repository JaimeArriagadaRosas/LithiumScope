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

## Piloto real del 30/10/2026
El laboratorio ya se integró como **opción 5** del menú. La opción 1 ejecuta M1 sobre `examples/integrated_demo/cases.csv` (10 muestras); la opción 2 recibe una imagen multibanda del usuario; la opción 3 muestra los CSV existentes. No realiza entrenamiento.

Se ejecutaron los 18 escenarios sobre M1/SVM `training_20260925_111245_m0300`. Resultados: `results/scientific_simulations/model_1/scenario_metrics.csv`. RMSE basal: 3.708 ppm; 25 % de pérdidas: 4.093 ppm y 100 % de casos fuera de dominio; 50 % de pérdidas: 5.917 ppm y 100 % fuera de dominio. Estos valores corresponden solo a los 10 casos de demostración, **no** a validación externa ciega.

Para M2/CatBoost se ejecutaron 19 escenarios con una **imagen artificial aleatoria** de 6 bandas y 32x32 píxeles para comprobar extremo a extremo el procesamiento, sin atribuir significado geológico a los scores: 13 escenarios puntuados y 6 abstenciones por banda completa sin datos. Archivo: `results/scientific_simulations/model_2_synthetic_smoke/image_scenario_metrics.csv`.

**Pendiente:** conseguir parches satelitales auténticos y sus referencias para medir robustez real de M2; extender los escenarios a múltiples imágenes y geografía. La instalación de modelos y las rutas usadas corresponden exclusivamente al clon, con `PYTHONPATH` dirigido a su carpeta `src`.

## Iteración: adquisición Sentinel-2 y ejecución masiva (30-10-2026)
- Se agregó `experiments/acquisition.py`: busca escenas Earth Search con el descargador preexistente de LithiumScope, guarda parches NPY, metadatos, manifiesto de procedencia y registro de fallos; valida píxeles sin datos y clases SCL mediante vecino más próximo.
- Se agregó `experiments/batch.py`: múltiples semillas M1, imágenes aceptadas M2 y resúmenes JSON; no modifica ni entrena modelos.
- Submenú ampliado: opción 4 adquiere parches; opción 5 ejecuta lotes M1 y M2 cuando haya parches aceptados.
- **Ejecución completada:** 90 combinaciones M1 = 18 escenarios por cinco semillas, cada combinación sobre diez casos conocidos. Salidas: `results/scientific_simulations/batch_m1/batch_m1.csv` y `summary.json`.
- **Verificación técnica:** 11 pruebas del laboratorio aprobadas, Ruff aprobado y compilación Python correcta.
- **Bloqueo de adquisición:** la API STAC respondió y una lectura real del asset rojo Sentinel-2 produjo una matriz 32x32, pero el primer intento automatizado de adquirir parches completos no finalizó; se detuvo. No se verificaron parches aceptados ni se completó el lote M2 con imágenes reales. Revisar latencia/timeout y aislar lecturas COG por banda antes de ampliar.
- No inferir precisión, calidad minera ni valores reales de Li a partir de scores sobre imágenes sin etiquetas de referencia. No usar estas imágenes para reentrenamiento en esta fase.
