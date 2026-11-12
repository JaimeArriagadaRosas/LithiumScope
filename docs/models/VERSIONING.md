# Estrategia de versionado y tags

No se crea un tag automáticamente después de entrenar.

## Identidad de un entrenamiento

Los entrenamientos nuevos utilizan una identidad local basada en fecha, hora y zona horaria:

`training_YYYYMMDD_HHMMSS_m0300`

El nombre se conserva durante reanudaciones compatibles. Una interrupción no crea por sí sola un nuevo entrenamiento.

## Reanudación

LithiumScope distingue dos conceptos:

- **source commit**: commit Git exacto desde el que se originó la ejecución y que queda registrado para trazabilidad;
- **training fingerprint**: firma científica utilizada para decidir si un checkpoint sigue siendo compatible.

El training fingerprint se calcula a partir de:

- modelo;
- SHA-256 del dataset de entrada;
- configuración científica relevante;
- código que puede modificar preprocessing, features, validación, tuning o entrenamiento.

El logger, la consola, dashboards, documentación y otros cambios de presentación no forman parte de esta firma.

Si encuentra una ejecución compatible en estado `cancelled`, `partial`, `running` o `completed`, reutiliza esa ejecución.

Los folds finalizados se guardan como checkpoints. Al reanudar:

- un fold completo se reutiliza;
- un algoritmo completo se reconstruye desde sus checkpoints;
- una ejecución ya completada se reutiliza sin generar otro modelo final.

Si cambia el dataset, la configuración científica o el código científico, la firma cambia y se crea una ejecución nueva. Un cambio exclusivamente de logging, consola o documentación no obliga a descartar checkpoints compatibles.

El commit Git sigue guardándose por separado. Para publicar un release conjunto se mantienen las comprobaciones de procedencia definidas por el release manifest.

## Cuándo una ejecución es candidata

Una ejecución solo se considera candidata a versión cuando:

1. su estado es `completed`;
2. existe un ganador;
3. ningún algoritmo requerido falló;
4. existe hash SHA-256 del dataset;
5. se conoce el commit Git utilizado.

Para versionar LithiumScope con ambos modelos, Modelo 1 y Modelo 2 deben además haber sido generados desde **el mismo commit**.

## Flujo propuesto

1. partir de un clon limpio;
2. ejecutar preboot desde cero;
3. entrenar ambos modelos;
4. si se interrumpe, volver a ejecutar y reanudar la misma identidad compatible;
5. revisar logs, dashboards y Excel;
6. usar `3. Métricas y resultados → Evaluar candidato local para futuro tag`;
7. revisar el `release_manifest.json` generado;
8. solo después crear el tag GitHub.

## Qué representa el tag

El tag fija principalmente:

- código fuente;
- configuraciones;
- hashes de datasets;
- run IDs fechados;
- algoritmos ganadores;
- métricas.

Los datasets grandes no deben introducirse directamente al historial Git. Para una publicación posterior, los modelos entrenados y manifests pueden adjuntarse como assets de un GitHub Release o almacenarse en un repositorio de artefactos, manteniendo sus hashes en el release manifest.


## Demostraciones publicables

Las demostraciones integradas se identifican por su run fechado:

`demonstration_YYYYMMDD_HHMMSS_m0300`

No representan una versión del producto y no utilizan etiquetas `v1`, `v2`
o similares. Una demostración puede empaquetarse con:

```bat
python -m lithiumscope.prediction.release_demo --run demonstration_YYYYMMDD_HHMMSS_m0300
```

El paquete incluye el PDF científico, Excel, CSV, diagnósticos, manifest, log,
figuras e inputs de la ejecución, más un `release_manifest.json` saneado.
También se genera un checksum SHA-256. El comando no crea tags ni publica en
GitHub automáticamente.
