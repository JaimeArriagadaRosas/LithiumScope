# Plan de implementación integral (histórico)

**Estado vigente:** consultar [HANDOFF.md](HANDOFF.md), [matriz de requisitos del repositorio](requirements/repository-requirements.md) y [trazabilidad científica](traceability.md). Los estados y restricciones de publicación anteriores a la primera CI son registros históricos; ahora el usuario autorizó push en la rama de trabajo y commits intermedios con skip ci.

Base: e12b877699fe97f301e2c87d07f4833084e1306f. Rama feat/integral-scientific-refactor.
Entorno nuevo: LithiumScope. Prohibido escribir en LithiumScope-Scientific-Lab o en seminario.
Sin promociones, releases ni entrenamientos definitivos. No publicar main.

| Fase | Trabajo | Aceptación | Estado |
|---|---|---|---|
| 01 | Clon, identidad, entorno, inventario y baseline | HEAD, rama, dependencias, JUnit y fallos previos registrados | Parcial |
| 02 | Revisión íntegra de los tres diagnósticos; asignar 57 criterios y arquitectura | Cada tarea con componente, test y evidencia | Parcial: índice generado |
| 03 | Tests, CI, logging por sesión y resultados | Regresión, concurrencia, aislamiento, trazabilidad | Iniciada |
| 04 | Core y resources independiente | Config efectiva, presupuesto y benchmarks | Pendiente |
| 05 | Acquisition y datasets | Procedencia, contratos, independencia M1/M2 | Pendiente |
| 06 | M1: preparación, EDA, validación, entrenamiento, evaluación, diagnóstico, metodología, inferencia | 27 criterios trazados; protocolos controlados, compatibilidad | Pendiente |
| 07 | M2: muestras, Sentinel-2, calidad, features, etiquetas, CV, calibración, evaluación, inferencia | 30 criterios trazados; umbrales separados, compatibilidad | Pendiente |
| 08 | Experiments y metodología | Experimentos aislados, sin promoción automática | Pendiente |
| 09 | Prediction y exportación | Inferencia no duplicada, resultados de solo lectura | Pendiente |
| 10 | Reporting en raíz | PDF sin cálculos científicos nuevos; compatibilidad | Pendiente |
| 11 | CLI y preboot | Menú 1-5/0, ENTER, demo directa y funciones históricas | Pendiente |
| 12 | Validación integral | Límite 300, directorios, CI, rendimiento, regresión y evidencia | Pendiente |

## Política de aceptación por tarea
Reproducir defecto o fijar contrato con test; implementar un cambio aislado; ejecutar prueba específica y regresión; registrar evidencia, commit y estado. No atribuir mejora de métricas sin comparación equivalente. No cambiar protocolos simultáneamente sin ablaciones. Mantener modelos, manifiestos y resultados publicados.

## Estado inicial y limitaciones
- Identidad Git y cuenta activa JaimeArriagadaRosas verificadas; sin cambios globales.
- Clon anterior excluido. El nuevo entorno usa Python 3.12 y .venv independiente.
- Instalación y baseline lanzados antes de cambios de código; comprobar results/tests/baseline/exit-code.txt y junit.xml.
- Diagnóstico E1/E2 vacío (0 bytes); no completar por inferencia.
- Los tres diagnósticos prioritarios están indexados; lectura completa y contraste aún pendientes. No iniciar reubicaciones de paquetes ni correcciones científicas hasta completar revisión y baseline.
- El inventario de origen y la matriz son referencias iniciales, no evidencias de resolución.
- Los logs y resultados de pruebas son locales. No añadir datasets ni artefactos masivos a Git.

## Estado acumulado tras el bloque de implementacion

| Fase | Estado verificado / trabajo restante |
|---|---|
| 01 | Completa: clon, identidad, entorno y baseline |
| 02 | Parcial: criterios indexados y revisados; antecedentes y complementos, trazabilidad por criterio pendientes |
| 03 | Parcial avanzada: logs por sesion, pruebas y CI configurado; cobertura 51%, CI remoto y reorganizacion de tests pendientes |
| 04 | Parcial: resources y logging centralizados; benchmark y separacion restante del core pendientes |
| 05 | Parcial: inventario historico y validacion fisica; importacion independiente, procedencia y acquisition pendientes |
| 06 | Parcial: restricciones fisicas, edad fija y EDA descriptivo; faltan contratos, EDA desarrollo y metodologia completa |
| 07 | Parcial: validacion fisica y EDA descriptivo; faltan identidad, integridad cache, calidad y metodologia completa |
| 08 | Parcial: acceso a simulaciones y catalogo; arquitectura de experiments pendiente |
| 09 | Parcial: consultas read-only; separacion de prediction/exportacion pendiente |
| 10 | Pendiente: mover reporting a raiz y verificar PDF |
| 11 | Parcial avanzada: menu y laboratorio, confirmaciones y pausa; preboot y detalle final pendientes |
| 12 | Parcial: regresion 469/11/0 y limite de 300 lineas; aceptacion cientifica, rendimiento y limite por directorio pendientes |

Prioridad siguiente: lectura restante, importacion historica independiente con hashes, contratos de calidad y procedencia, cache M2, EDA por particion y reorganizacion por responsabilidades. No iniciar EDA definitivo hasta completar criterios de entrada y pruebas.

## Estado actualizado tras reorganizacion y auditoria espacial

- Fases 05-07: importacion local con hashes completada para las instantaneas disponibles; preparacion y validacion reorganizadas; EDA preliminar real disponible. Quedan adquisicion original, contratos por fuente, calidad multibanda, holdout y entrenamiento/ablaciones.
- Fases 09-10: reporting en raiz y calculos centrales separados; inferencia M2 compartida, contratos de aplicabilidad. Falta verificar bundles y PDF con modelos reales.
- Fase 11: arranque real verificado con core y sin extras ML opcionales; EDA local sin adquisicion automatica.
- Fases 02-04,08,12: lectura completa de antecedentes y complementos, trazabilidad, rendimiento, reorganizacion de directorios restantes, simulaciones y validacion cientifica pendientes.
- No push ni workflows remotos. No promocion a partir de demostracion. No se realizaron entrenamientos definitivos.
