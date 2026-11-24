# Plan de implementación integral

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
