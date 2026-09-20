# Reproducibilidad de LithiumScope

LithiumScope trata cada entrenamiento como un experimento identificable mediante un `run_id`.

## Evidencia capturada por ejecución

Cada directorio `results/<modelo>/runs/<run_id>/` contiene:

- `run.json`: estado, eventos, ganador, métrica principal, commit de Git y versiones de librerías;
- `config/`: copia de los YAML utilizados;
- `manifests/dataset_manifest.json`: hash SHA-256 y estructura del dataset de entrenamiento;
- `tables/`: métricas, predicciones OOF y auditoría;
- `figures/`: gráficos;
- `exports/`: Excel y referencias al ganador;
- `dashboard.html`: visualización local.

## Estados

Las ejecuciones pueden quedar como:

- `created`;
- `running`;
- `completed`;
- `partial`;
- `failed`;
- `cancelled`;
- `crashed`.

Una ejecución con algoritmos fallidos se considera `partial`, aunque exista un ganador de esa ejecución. Un proceso que termina sin pasar por el cierre controlado puede recuperarse posteriormente como `crashed` sin borrar sus checkpoints.

## Seeds y entorno

Antes de entrenar se fija la seed configurada para Python, NumPy y, cuando está disponible, PyTorch. También se capturan:

- versión de Python;
- sistema operativo;
- commit Git;
- versiones de dependencias ML;
- CPU disponibles y presupuesto de workers;
- GPU CUDA cuando corresponde.

## Baselines

Modelo 1 incluye un `DummyRegressor(strategy="mean")` y Modelo 2 un `DummyClassifier(strategy="prior")`.

Los baselines se reportan junto con la competencia, pero **no son candidatos a modelo ganador**. Su propósito es demostrar cuánto valor agrega el aprendizaje frente a una estrategia trivial.

## Controles científicos

LithiumScope valida automáticamente:

- que `Li_icpms` no aparezca entre los predictores;
- que las predicciones out-of-fold estén completas y sean finitas;
- que los grupos espaciales de Modelo 2 no crucen train/test cuando se utiliza validación agrupada;
- que imputación, encoding y escalamiento permanezcan dentro de los pipelines ajustados por fold.

## Aplicabilidad

Los modelos guardan rangos empíricos de entrenamiento (percentiles 1–99) para poder advertir cuando una inferencia utiliza variables alejadas del dominio observado.

Esta advertencia no reemplaza una evaluación formal de incertidumbre o extrapolación geológica.


## Observabilidad

Cada proceso crea un log de sesión con eventos `SESSION_START` y `SESSION_END`. El arranque valida que ese archivo exista y tenga contenido.

Los logs de errores se crean de forma lazy: una sesión sin errores no deja un `errors_*.log` vacío.

Los warnings de Python se enrutan al logging de LithiumScope y se deduplican. El detalle permanece en el log de sesión, mientras que la consola conserva una única línea de estado reescribible para operaciones largas.

Al finalizar se muestra un resumen con:

- warnings únicos y totales;
- cantidad de errores;
- ruta del log de sesión;
- ruta del log de errores solo cuando realmente existe.
