# Proceso autónomo GEOROC — adquisición y puntuación exploratoria

## Estado inicial
- **En ejecución en el equipo autorizado** mediante proceso Python independiente de la sesión interactiva.
- PID inicial: **13384** (comprobar `worker.pid`; no reutilizar si finaliza).
- **3.220 candidatos** GEOROC originales; para el lote se seleccionaron **1.307 registros** que pasan los controles automáticos previos y están separados al menos 100 km del corpus Mamani09 completo.
- Se agruparon por coordenadas redondeadas a cuatro decimales: **498 ubicaciones únicas**. Se conserva el vínculo completo muestra-coordenada en `sample_image_map.csv`.
- Hay **271 candidatos de Li alto (≥21,6 ppm)** y 1.036 de Li bajo dentro de los 1.307 registros preseleccionados. Esto muestra que GEOROC más amplio sí contiene ambas clases; el subconjunto anterior de nueve de la publicación 20576 no.
- **No se declara independencia geoquímica, calidad de etiquetado ni precisión externa.** Es una inferencia exploratoria masiva para obtener datos antes de revisar resultados.

## Ubicación
`results/scientific_simulations/external_geochemistry_audit/background_georoc_exploratory/`

Archivos: `coordinate_plan.csv`, `sample_image_map.csv`, `status.json`, `patch_manifest.csv`, `errors.csv`, `worker.pid`, `worker_stdout.log`, `worker_stderr.log`, carpeta `patches/`, carpeta `scores/`.

## Proceso
El trabajador busca una escena de 2025 por coordenada, verifica su capa SCL y obtiene seis bandas Sentinel-2, generando un parche de 640 m (32×32). Evalúa los 19 escenarios existentes de M2 sobre cada parche aceptado, sin reentrenar. Guarda un registro de estado tras cada coordenada. Los tiempos y posibles fallos dependen de la conexión de red y los servicios de imágenes.

El proceso es **reanundable**; si se interrumpe, repetir:

```powershell
Set-Location "D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Scientific-Lab"
$env:PYTHONPATH="$PWD\src"
python -u -m lithiumscope.experiments.georoc.background_georoc_exploratory
```

Para **solicitar detención ordenada**, crear un archivo vacío con este nombre completo: `background_georoc_exploratory\STOP`; el trabajador comprueba esa señal entre ubicaciones. La detención de Windows o un reinicio también detienen el proceso, pero permiten reanudarlo posteriormente.

Para **revisar progreso**:

```powershell
Get-Content "results\scientific_simulations\external_geochemistry_audit\background_georoc_exploratory\status.json"
```

Estados posibles: `RUNNING`, `COMPLETE`, `STOPPED_BY_USER`, `FATAL`. La presencia de `RUNNING` no demuestra por sí sola que el proceso siga vivo; verificar `worker.pid` con `Get-Process`.

**No hay envío automático de mensajes al chat**. El usuario puede volver a consultar el resultado y se leerán los archivos del proceso.

## Restricciones
Los datos geológicos son puntuales, mientras que una imagen Sentinel-2 describe reflectancia de una superficie extensa. No calcular métricas externas como ROC-AUC con candidatos no aprobados. No introducir las 1.307 filas como datos de entrenamiento. El lote puede incluir escenas temporalmente distintas de la fecha geológica de muestreo. Se limita a la rama experimental y a resultados locales, sin tocar `main` ni modelos.
