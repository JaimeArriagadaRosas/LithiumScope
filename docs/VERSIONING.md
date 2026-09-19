# Estrategia de versionado y tags

No se crea un tag automáticamente después de entrenar.

## Cuándo una ejecución es candidata

Una ejecución solo se considera candidata a versión cuando:

1. su estado es `completed`;
2. existe un ganador;
3. ningún algoritmo requerido falló;
4. existe hash SHA-256 del dataset;
5. se conoce el commit Git utilizado.

Para versionar LithiumScope con ambos modelos, Modelo 1 y Modelo 2 deben además haber sido generados desde **el mismo commit**.

## Flujo propuesto

1. ejecutar una prueba completa;
2. revisar logs, dashboards y Excel;
3. usar `3. Métricas y resultados → Evaluar candidato local para futuro tag`;
4. revisar el `release_manifest.json` generado;
5. solo después crear el tag GitHub.

## Qué debería representar el tag

El tag fija principalmente:

- código fuente;
- configuraciones;
- hashes de datasets;
- run IDs;
- algoritmos ganadores;
- métricas.

Los datasets grandes no deberían introducirse directamente al historial Git. Para una publicación posterior, los modelos entrenados y manifests pueden adjuntarse como assets de un GitHub Release o almacenarse en un repositorio de artefactos, manteniendo sus hashes en el release manifest.

## Convención inicial

El manifiesto local propone nombres del tipo:

`lithiumscope-training-YYYYMMDD`

La convención puede cambiarse antes del primer tag estable.
