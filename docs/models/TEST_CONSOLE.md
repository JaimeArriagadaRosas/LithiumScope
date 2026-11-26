# Validación local: presentación y evidencias

El menú del laboratorio 5 > 3 > 1 ejecuta la misma suite con cobertura.
El observador de pytest registra colección, comienzo y fin de cada prueba y
resultados por fase. La consola muestra avance real, área, incidencias y
un aviso cada ocho segundos durante pruebas lentas o recopilación de evidencia.
Los estados finales proceden del JUnit, no de estimaciones del progreso.

Se conservan summary.json, junit.xml, coverage.xml, .coverage, pytest.log,
events.jsonl y progress.jsonl. El resumen presenta conteos, cobertura de líneas,
integridad Git, grupos, pruebas fallidas y omitidas y cinco pruebas más lentas.
Si falta JUnit o cobertura, se declara explícitamente. No se modifica la
selección ni la ejecución científica de los tests.

Para consumidores que necesitan el JSON por stdout:
`python scripts/run_tests.py --json --coverage -- -q tests`.

Los dos contratos de regresión se alinean al nuevo menú y al anexo dinámico,
conservando la prohibición de exponer las opciones retiradas y de inventar
evidencias científicas cuando faltan archivos.
