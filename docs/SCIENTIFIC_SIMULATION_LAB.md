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

## Diagnóstico del bloqueo COG (30/10/2026)
- Se incorporó `raster_worker.py` y se reestructuró `acquisition.py` para aislar cada lectura de banda/SCL en un subproceso con límite temporal, registrar errores e ir guardando manifiestos parciales.
- Prueba real `atacama_2025`: SCL agotó el límite de 35 segundos; entrada documentada en `results/scientific_simulations/sentinel2_real/failures.csv`.
- Prueba independiente de asset B04.tif del 26/10/2025: agotó 45 segundos. Por tanto, el problema no está restringido a SCL; hay una dificultad de lectura remota de COG desde este entorno.
- 15 pruebas automatizadas aprobadas; Ruff sin observaciones. No se descargó una imagen completa válida y **no hay evaluación M2 real concluida**. Próximo paso: diagnosticar el acceso HTTP range a S3/GDAL y/o añadir fallback HTTP de ventanas o proveedor alternativo; no iniciar descarga masiva hasta validar una sola imagen completa.

## Solución de acceso COG y primer M2 real (30/10/2026)
- El acceso HTTP Range contra el bucket de Sentinel-2 respondió `206 Partial Content` en aproximadamente 1.4 s. La lectura GDAL remota seguía agotando 35-45 segundos.
- Se implementó `experiments/http_tiles.py` con `fsspec`, `tifffile` e `imagecodecs`: lee únicamente los bloques TIFF pertinentes mediante rangos HTTP y extrae una ventana 640 m / 32 píxeles. La implementación actual utiliza muestreo al vecino más próximo, **no** es idéntica al remuestreo bilineal del descargador del entrenamiento. Se debe estudiar el efecto de esta diferencia antes de sacar conclusiones sobre calidad predictiva.
- El trabajador `raster_worker.py` usa este método para URL HTTPS, manteniendo límites de tiempo por asset y el control de SCL.
- Primer parche completo aceptado: `atacama_2025`, escena Sentinel-2 `S2C_19KEQ_20251223_0_L2A`, SCL con fracción local inválida 0; seis bandas descargadas y guardadas en `results/scientific_simulations/sentinel2_real/`.
- Primer lote M2 real: 19 escenarios, 13 puntuaciones y 6 abstenciones por banda ausente. Score basal CatBoost 0.326944, fracción de features fuera de dominio 0.02381; **score de priorización relativo, no probabilidad de yacimiento ni validación de litio**. CSV y resumen en `results/scientific_simulations/batch_m2/`.
- Tests científicos: 15 aprobados; Ruff sin observaciones.
- Pendientes: validar remuestreo bilineal equivalente al entrenamiento, cuantificar efecto de imágenes en distintas fechas y ubicaciones, añadir comprobación radiométrica independiente, ampliar la muestra. No entrenar con estos parches sin su debida validación.

## Comparación geográfica y temporal M2, 30-10-2026
- Se corrigió `experiments/http_tiles.py` para interpolación bilineal de bandas continuas, con centros de píxeles, cuatro vecinos y omisión ponderada de nodata; la máscara categórica SCL conserva vecino cercano. Se agregaron pruebas sintéticas de interpolación. La equivalencia numérica exacta respecto a GDAL/rasterio aún requiere un contraste con referencia raster real (distintas resoluciones y bordes).
- Se ejecutó adquisición de ocho parches válidos: cuatro zonas (Atacama, Tara, Maricunga, Pedernales), años 2024 y 2025. Se regeneró Atacama 2025 para eliminar la mezcla del muestreo anterior; la copia anterior se conserva en `legacy_nearest`. Todos los parches del manifiesto activo usan el método nuevo.
- Se ejecutaron **152 evaluaciones M2** (8 parches × 19 escenarios), con 104 puntuaciones y 48 abstenciones previstas por banda completa ausente. Resultados en `results/scientific_simulations/batch_m2/batch_m2.csv`, resúmenes en `summary.json`, `geography_comparison.csv` y `geography_comparison.json`.
- Scores basales 2024/2025: Atacama 0.32958/0.31792; Tara 0.12629/0.13240; Maricunga 0.42148/0.42649; Pedernales 0.61951/0.65227. Son scores relativos, no probabilidades de depósito, y los parches no tienen mediciones de Li para evaluar exactitud.
- La opción 6 del laboratorio interactivo regenera la comparativa geográfica/temporal.
- M1 conserva 90 escenarios-semilla sobre 10 muestras conocidas; M2 evalúa 8 escenas reales. No se ejecutaron todavía los 1.000–3.000 casos originalmente propuestos ni una validación externa independiente.

## Validación GeoTIFF y ampliación 2023-2025 (30/10/2026)
- Comparación de remuestreo con `rasterio` sobre GeoTIFF sintéticos locales reales: dos pruebas bilineales (resoluciones 10 m y 20 m) y una prueba nearest sobre categorías SCL; tres aprobadas. Los límites de error para los gradientes fueron tolerancia relativa 1e-5 y absoluta 0.002. Esta equivalencia cubre únicamente los casos probados, no todos los efectos de bordes, formatos radiométricos o georreferenciación.
- Se adquirieron cuatro parches adicionales de 2023 (Atacama, Tara, Maricunga, Pedernales), 6 bandas, SCL aceptado; se reutilizaron ocho parches existentes de 2024-2025. En total, **12 parches aceptados**, cuatro ubicaciones y tres años.
- Se recomputó el lote M2 con 12 × 19 = **228 evaluaciones**: 156 puntuadas, 72 abstenciones cuando se eliminó una banda completa. Métricas CSV e informe geográfico/temporal actualizados en `results/scientific_simulations/batch_m2/`.
- Scores basales 2023/2024/2025: Atacama 0.318635/0.329580/0.317917; Tara 0.131538/0.126288/0.132395; Maricunga 0.441520/0.421482/0.426491; Pedernales 0.585484/0.619507/0.652272. Los scores NO son concentraciones ni probabilidades de yacimiento; no existe verdad terreno química asociada.
- Verificación: 21 pruebas científicas aprobadas, Ruff sin observaciones; rasterio reporta 9 avisos de deprecación pendientes en sus APIs.
- Pendiente: ampliar a más fechas/escenas con cobertura estacional y disponer de referencias geoquímicas ciegas y compatibles para medición real de exactitud. No se reentrenaron modelos.

## Piloto estacional y espacial (30/10/2026)
- Se añadió una ventana temporal opcional al descargador de evaluación y una matriz ampliable de desplazamientos geográficos (0.004 grados al norte/este aprox.). Cada campaña usa una carpeta distinta para no confundir escenas con el mismo nombre.
- Piloto ejecutado con Atacama verano 2025 (01/01-31/03), invierno 2025 (01/06-31/08) y coordenada desplazada 0.004° este. Tres parches Sentinel-2 aceptados y 57 evaluaciones M2: 39 puntuaciones y 18 abstenciones por banda íntegra ausente.
- Resultados en `results/scientific_simulations/extended_2025/`: `extended_manifest.csv`, `batch/batch_m2.csv`, `batch/summary.json` y comparación geográfica.
- Se implementó un plan de extensión a cuatro ubicaciones y desplazamientos, pero el piloto ejecutado aquí cubrió solo tres parches de Atacama; no deben presentarse los escenarios no ejecutados como resultados.
- Se creó catálogo de datos químicos externos en `data/external/independent_geochemistry/` (plantilla vacía, validación de material, unidades, método, origen, independencia declarada). Los candidatos requieren auditoría de solapamiento con el entrenamiento antes de llamarlos validación externa ciega. No se han adquirido datos geoquímicos independientes.

## Campaña ampliada de 2025 (30/10/2026)
Se ejecutaron 9 nuevas adquisiciones en Tara, Maricunga y Pedernales (verano, invierno y desplazamiento hacia el este), que se combinaron con 3 parches Atacama del piloto previo. Todos los **12 parches** fueron `ACCEPT` tras control local SCL (fracción inválida de 0 a 0.032227). Se ejecutaron **228 escenarios M2**, con 156 puntuaciones y 72 abstenciones por eliminación de una banda completa.

Resultados guardados localmente en `results/scientific_simulations/campaign_2025/campaign_manifest.csv`, `batch/batch_m2.csv` y `batch/summary.json`. Scores basales verano/invierno: Atacama 0.315828/0.382427; Tara 0.091350/0.063254; Maricunga 0.366533/0.425180; Pedernales 0.494263/0.329706. El conjunto de desplazamientos del este tiene scores 0.223356, 0.123555, 0.443666 y 0.596672 respectivamente.

**Limitaciones decisivas:** las escenas de desplazamiento se eligieron en diciembre y las basales de verano/invierno en marzo/agosto. Las diferencias por desplazamiento mezclan cambio de lugar y de fecha, por lo que no aíslan causalmente el efecto espacial. Además, se utilizó un parche por escenario/sector y no hay Li medido para esas posiciones. No interpretar los scores como evidencia de depósitos o validación externa. Como siguiente prueba, reutilizar exactamente la misma escena STAC para el punto original y sus coordenadas desplazadas.

Se documentaron GEOROC y recursos geoquímicos SERNAGEOMIN como candidatas de datos, no importadas a los modelos. Ver `docs/GEOCHEMICAL_EVIDENCE_SOURCES.md`.

## Ensayo espacial controlado y verificación de escenas, 30/10/2026
- Se verificó que los cuatro pares espaciales original/este de diciembre 2025 usan exactamente el mismo `scene_id`, eliminando el factor temporal entre cada par. El desplazamiento geodésico es aproximadamente 397 a 409 m. Scores original/este: Atacama 0.317917/0.223356; Tara 0.132395/0.123555; Maricunga 0.426491/0.443666; Pedernales 0.652272/0.596672. CSV: `results/scientific_simulations/controlled_spatial/all_verified_east_pairs.csv`.
- Se implementó `experiments/spatial_control.py`, capaz de generar 36 posiciones (cuatro centros × [centro, cuatro direcciones × 400 y 1000 m]), descargar solamente parches de la misma escena STAC recuperada por ID exacto, guardar manifiestos reanudables, evaluar los 19 escenarios por parche y exportar estabilidad en función de distancia y dirección.
- Piloto con los dos primeros parches nuevos de Atacama (400 m norte/este): 2/2 aceptados, scores 0.271165 y 0.222698 frente al centro 0.317917; cambios absolutos 0.046752 y 0.095219. Resultados en `controlled_spatial/spatial_robustness.csv`.
- Los 684 escenarios totales siguen siendo una capacidad planificada del laboratorio, **NO** un lote completo ejecutado. Los parches desplazados a 400 m pueden solaparse con el original de 640 m, por lo que las observaciones no son estadísticamente independientes.
- Tests espaciales y de interpolación ejecutados: 8 aprobados, Ruff sin observaciones. No se reentrenaron modelos ni se introdujeron mediciones químicas como verdad terreno.

## Batería espacial completa (30-10-2026)
- **36 de 36** parches aceptados, en Atacama, Tara, Maricunga y Pedernales. Cada región reutiliza exactamente una misma escena Sentinel-2 de diciembre de 2025, recuperada por su `scene_id`; ocho desplazamientos por centro: norte, este, sur y oeste a 400 m y 1000 m.
- **684 de 684 evaluaciones M2 ejecutadas** (36 parches × 19 escenarios): **468 con puntuación y 216 abstenciones previstas** por eliminación completa de banda. Artefacto consolidado: `results/scientific_simulations/controlled_spatial/all_684_scenario_metrics.csv`.
- Comparación espacial de scores basales: a 400 m (16 pares), cambio absoluto medio **0.08112097**, máximo **0.40815289**; a 1000 m (16 pares), media **0.09353551**, máximo **0.57885885**. Archivos `spatial_robustness.csv`, `spatial_robustness.json`, `spatial_manifest.csv`, `summary.json`.
- Para acelerar la ejecución se añadieron `spatial_batch_worker.py` y `spatial_batch_run.py`: una apertura HTTP por banda y escena para procesar múltiples posiciones, con reanudación y registro de fallos por región.
- Precauciones: estas son **32 comparaciones desplazadas alrededor de cuatro centros**, no 32 observaciones geoquímicas independientes. Las ventanas de 640 m pueden solaparse, y los scores son relativos y no equivalen a pruebas de precisión ni a validación geográfica externa. La distancia más larga no implica monotonía de la respuesta.

## Análisis pareado de sensibilidad, temporalidad y variabilidad (30/10/2026)
- Módulo nuevo `src/lithiumscope/experiments/sensitivity_analysis.py` (solo lectura). Reúne 684 observaciones de la batería espacial, 228 del historial 2023-2025 y 228 de la campaña estacional y espacial 2025. Para cada perturbación, compara el score con el baseline **de la misma imagen**, registra número puntuado/abstenciones, media, mediana, desviación estándar, percentil 90 y máximo del cambio absoluto.
- Resultados en `results/scientific_simulations/scenario_analysis/`: `ANALISIS_SENSIBILIDAD.md`, `scenario_sensitivity.csv`, `paired_per_image.csv`, `spatial_directions.csv`, `spatial_by_region_distance.csv`, `interannual_variability.csv`, `seasonal_differences.csv` y `analysis_manifest.json` con hashes SHA-256.
- Top de sensibilidad en 36 parches espaciales: brillo ×0.8, cambio absoluto medio 0.072112; brillo ×1.5, 0.071277; ruido 25 %, 0.065920; 75 % de píxeles faltantes, 0.057077. Las comparaciones NO son errores de precisión científica.
- Comparaciones espaciales: Pedernales tuvo medias 0.236078 a 400 m y 0.237119 a 1000 m; Tara 0.010392 y 0.020893 respectivamente. Comparación interanual de rango basal: Tara 0.006108; Atacama 0.011664; Maricunga 0.020038; Pedernales 0.066788. Variación invierno-verano: +0.066599 Atacama, -0.028096 Tara, +0.058647 Maricunga y -0.164557 Pedernales.
- No inferir generalización externa con estas escenas. Ventanas espaciales de 640 m pueden solaparse; año y estación también introducen cambios de fecha, sensor y superficie. El inventario de mediciones independientes sigue sin muestras nuevas certificadas; ver `GEOCHEMICAL_EVIDENCE_SOURCES.md`.

## Ampliación estacional 2024 y comparación 2024-2025 (30/10/2026)
- Se adquirieron **8 parches Sentinel-2 reales adicionales** (Atacama, Tara, Maricunga y Pedernales, verano e invierno 2024). Todos `ACCEPT` en calidad local SCL. Se ejecutaron **152 simulaciones M2**: 104 puntuadas y 48 abstenciones esperadas por banda íntegra ausente. Manifiesto y scores: `results/scientific_simulations/seasonal_2024/seasonal_manifest.csv` y `batch/batch_m2.csv`.
- Se añadió `experiments/seasonal_comparison.py` para reunir 2024 y 2025, comparar baselines por pares año-región y exportar trazabilidad de satélite, escena, fecha, SHA256 de las fuentes y variabilidad. Directorio: `results/scientific_simulations/seasonal_comparison_2024_2025/`.
- Diferencia invierno - verano del score por sector (2024 / 2025): Atacama -0.026536 / +0.066599; Tara -0.025543 / -0.028096; Maricunga +0.048241 / +0.058647; Pedernales -0.049085 / -0.164557.
- **Limitaciones:** solo un parche por año-estación-región; las escenas cambian en fecha, satélite y condiciones atmosféricas. El cambio de score NO equivale a un efecto estacional causal ni a precisión científica. La comparación 2025 incluye escenas procedentes de Sentinel-2A/B/C. Sin mediciones geoquímicas externas compatibles, la generalización geográfica independiente permanece sin demostrar.
