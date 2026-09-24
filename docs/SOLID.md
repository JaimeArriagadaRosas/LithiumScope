# Diseño SOLID en LithiumScope

LithiumScope no pretende demostrar los principios SOLID mediante un número mágico de líneas. La arquitectura los utiliza como restricciones de diseño y añade algunos tests estructurales para evitar regresiones obvias.

## S — Single Responsibility

- `steps/`: una transformación por archivo.
- `pipeline.py`: ordena transformaciones; no entrena modelos.
- `factory.py`: construye algoritmos; no ejecuta CV.
- `cv_runner.py`: ejecuta validación; no decide el ganador.
- `competition.py`: orquesta y rankea; no contiene implementaciones de modelos.
- `artifacts.py` / `plots.py`: producen evidencia de resultados.
- `persistence/`: guarda y carga modelos.
- `cli/*_command.py`: traduce acciones del usuario a casos de uso.

## O — Open/Closed

Agregar un algoritmo nuevo requiere crear su módulo y registrarlo en `factory.py`/config, sin reescribir el pipeline de datos.

## L — Liskov Substitution

Los algoritmos usados por una competencia mantienen la interfaz `fit`/`predict` o `fit`/`predict_proba` esperada por el runner correspondiente.

## I — Interface Segregation

Modelo 1 y Modelo 2 tienen contratos diferentes: regresión y clasificación/prospectividad. No se fuerza una interfaz única que mezcle responsabilidades científicas distintas.

## D — Dependency Inversion

Los orquestadores dependen de factories y contratos, no de una única implementación concreta. La CLI tampoco importa lógica interna de cada algoritmo.

## Tests estructurales

`tests/test_architecture.py` protege tres decisiones explícitas:

1. no introducir notebooks;
2. mantener `menu.py` como despachador pequeño;
3. mantener los `trainer.py` como puntos de entrada delgados.

Estos tests complementan, pero no reemplazan, revisión de diseño y tests funcionales.


## Límites adicionales de responsabilidad

La segunda etapa arquitectónica agrega responsabilidades transversales sin trasladarlas a los modelos:

- `datasets/manifest.py`: identidad y trazabilidad de datasets;
- `core/experiment_tracker.py`: ciclo de vida de una ejecución;
- `core/scientific_checks.py`: invariantes contra leakage y validación incompleta;
- `core/reproducibility.py`: seeds y fingerprint del entorno;
- `core/resources.py`: presupuesto común de CPU;
- `results/catalog.py`: lectura/comparación de ejecuciones;
- `results/release.py`: preparación local de candidatos de versión.

Un algoritmo no debe conocer Git, dashboards, manifests ni tags. Del mismo modo, el tracker no debe conocer detalles de Random Forest, XGBoost o Sentinel-2.

## Tamaño de archivos

LithiumScope no utiliza el número de líneas como definición de SOLID. Los tests que limitan algunos archivos solo protegen decisiones concretas (por ejemplo, que `menu.py` siga siendo un despachador). Para los módulos de dominio se priorizan cohesión, acoplamiento y facilidad de prueba.


## Laboratorio de inspección

El laboratorio separa explícitamente estas responsabilidades:

- `tools/pipeline_inspect.py`: CLI/orquestación solamente;
- `tools/pipeline_inspection_sources.py`: steps de fuentes, armonización, concatenación y deduplicación;
- `tools/pipeline_inspection_models.py`: inspección de M1 y candidatos M2;
- `tools/pipeline_inspection_reporting.py`: tablas y resúmenes;
- `tools/lab_preboot.py`: disponibilidad del entorno y fuentes brutas del laboratorio;
- `tools/gpu_probe.py`: diagnóstico de aceleradores;
- `datasets/georoc_filtered_acquisition.py`: coordinación de adquisición GEOROC;
- `datasets/georoc_query_contract.py`: contrato científico de la extracción;
- `datasets/georoc_query_flow.py`: navegación de la consulta remota;
- `tools/georoc_query_models.py`: representación/parser HTML;
- `tools/georoc_query_payload.py`: construcción de payloads del formulario;
- `tools/georoc_query_html.py`: navegación HTTP mínima;
- `tools/georoc_query_export.py`: materialización y validación de la exportación.

Para este subsistema se adopta además una guarda estructural de 300 líneas por archivo. No define SOLID de forma general; sirve para impedir que el laboratorio vuelva a concentrar parsing, HTTP, reporting, preboot y orquestación en un solo módulo.
