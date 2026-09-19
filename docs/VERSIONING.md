# Estrategia de versionado y tags

No se crea un tag automáticamente después de entrenar.

## Identidad de un entrenamiento

Los entrenamientos nuevos utilizan una identidad local basada en fecha, hora y zona horaria:

`training_YYYYMMDD_HHMMSS_m0300`

El nombre se conserva durante reanudaciones compatibles. Una interrupción no crea por sí sola un nuevo entrenamiento.

## Reanudación

LithiumScope calcula una firma a partir de:

- modelo;
- SHA-256 del dataset de entrada;
- configuración relevante;
- commit Git.

Si encuentra una ejecución compatible en estado `cancelled`, `partial`, `running` o `completed`, reutiliza esa ejecución.

Los folds finalizados se guardan como checkpoints. Al reanudar:

- un fold completo se reutiliza;
- un algoritmo completo se reconstruye desde sus checkpoints;
- una ejecución ya completada se reutiliza sin generar otro modelo final.

Si cambia dataset, configuración o commit, la firma cambia y se crea una ejecución nueva.

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
