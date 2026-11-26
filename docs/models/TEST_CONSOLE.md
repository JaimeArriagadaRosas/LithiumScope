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

En una terminal interactiva (CMD/PowerShell) el progreso sobrescribe una sola
línea mediante retorno de carro y se limpia antes del resumen. Se limita al
ancho de la terminal para evitar saltos de línea. Si stdout está redirigido
a un archivo o capturado por otro proceso, se mantiene el progreso por líneas
para facilitar su lectura. Los eventos y el resumen final no cambian.
