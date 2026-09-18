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
