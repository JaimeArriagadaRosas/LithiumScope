# Principios de diseño y mantenimiento

## Enfoque

LithiumScope se desarrolla mediante **Clean Code aplicado a módulos Python**, con separación de responsabilidades y contratos científicos. El diseño no depende de programación orientada a objetos: predominan funciones, módulos y pipelines, aunque existen clases y estimadores externos donde tienen sentido.

SOLID es una **fuente parcial de criterios de diseño**, no el nombre de la arquitectura ni una garantía de calidad por sí mismo.

## Reglas operativas

1. **Cohesión:** cada módulo debe responder por una tarea comprensible y testeable; dividir archivos grandes solo si mejora claridad y mantenimiento.
2. **Responsabilidad única:** el menú despacha, los pasos transforman datos, los runners validan y el código de reportes presenta evidencia.
3. **Dependencias explícitas:** los modelos reciben datos, configuración y contratos; no dependen directamente de la interfaz ni de GitHub.
4. **Contratos de datos:** nombres, unidades, orden de features, shapes y umbrales científicos se validan y versionan.
5. **Determinismo verificable:** seeds, splits, configuraciones, datos, versiones y fingerprints quedan documentados.
6. **Separación train/test:** imputación, calibración, selección de parámetros y escalamiento deben ajustarse en el fold correspondiente.
7. **Resultados auditables:** no reemplazar silenciosamente `run_id`, OOF, métricas o comparaciones históricas.
8. **Simplicidad:** evitar capas, configuraciones y scripts duplicados sin responsabilidad propia.
9. **Pruebas proporcionadas:** tests de contrato, integración y regresión para los cambios que alteran el comportamiento.
10. **Calidad científica:** un mejor ranking no equivale a una mejora validada en datos independientes.

## Qué se conserva de SOLID

| Principio | Lectura válida en este proyecto |
|---|---|
| S: Single Responsibility | Funciones y módulos con responsabilidades delimitadas |
| O: Open/Closed | Nuevos algoritmos registrados sin rehacer el pipeline de datos |
| L: Liskov | Adaptadores de estimadores deben respetar la interfaz usada por su runner, cuando aplica |
| I: Interface Segregation | Contratos de regresión y clasificación independientes |
| D: Dependency Inversion | Orquestación basada en contratos/factories, no en un algoritmo fijo |

Los principios L, I y D **no deben presentarse como cumplimiento universal comprobado de SOLID**. Son criterios de diseño observables en componentes concretos. No hay un porcentaje, tamaño máximo de archivo o prueba estructural que certifique SOLID.

## Mapa de responsabilidades

- `steps/step_XX_*.py`: transformaciones secuenciales; `pipeline.py`: coordinación.
- `training/factory.py`: selección de estimadores; `cv_runner.py`: validación cruzada.
- `training/competition.py`: comparación; `evaluation/`: métricas y figuras.
- `datasets/`: acceso, adaptación, identidad y verificación de fuentes.
- `persistence/`: serialización e identificación; `results/`: lectura y exportación.
- `prediction/reporting/`: informes y presentación; `cli/`: interacción.

El control de tamaño de archivos de `scripts/check_code_size.py` y ciertos tests estructurales sirven como **guardas concretas**, no como definición general de Clean Code o SOLID.

## Prácticas científicas relacionadas

La arquitectura debe ayudar a controlar leakage, compatibilidad de modelos, incertidumbre, generalización y procedencia. Consultar [pipelines](SCIENTIFIC_PIPELINES.md), [reproducibilidad](../operations/REPRODUCIBILITY.md) y [decisiones de modelos](../models/README.md).
