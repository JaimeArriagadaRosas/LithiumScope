# Validación local: presentación y evidencias

El menú 5 > 3 > 1, **Validación del software**, ejecuta una sola suite de pytest
con cobertura, sin filtros automáticos. Las opciones 2 y 3 quedan reservadas
para comparaciones y evaluación de tratamiento de datos.

## Flujo

1. Sintaxis, lint, dependencias, imports, arranque y límites de tamaño.
2. Auditoría de la arquitectura de tests (inventario, catálogo, fixtures y marcadores).
3. Recopilación y ejecución de todos los tests, conservando los errores de colección.
4. Comprobación de artefactos protegidos, identidad Git y cobertura.
5. Reconciliación de los casos recopilados y JUnit; evaluación del catálogo de criterios.
6. Resumen y exportación de evidencias.

El observador registra colección, comienzo y fin de cada prueba y resultados por
fase. La consola mantiene una sola línea dinámica en CMD/PowerShell, muestra
categoría/responsabilidad y avisa de pruebas lentas cada ocho segundos.
Si stdout está redirigido, se conserva el progreso por líneas.

Las ocho categorías son excluyentes; los ámbitos M1, M2, EDA, training, slow y
local_artifacts se solapan. Los resultados provienen de JUnit, no del progreso
estimado. No se certifica una suite incompleta o con identidad no comprobada.
Las asociaciones con criterios no equivalen a aceptación científica.

## Evidencias

En results/tests/<run_id> se generan:

- summary.json: identidad, estado, comprobaciones, conteos y agregaciones.
- junit.xml, coverage.xml y .coverage: evidencia original cuando se genera.
- architecture.json: auditoría estática de la suite (modo integral).
- criterion_evidence.json: asociaciones de los 166 criterios, por familia.
- verification_report.md: informe legible para revisión, sin duplicar los logs.

Los logs pytest.log, events.jsonl, progress.jsonl y las comprobaciones previas
quedan en logs/sessions/<session_id>/tests/<run_id>. No se inventan conteos,
cobertura ni resultados cuando falta evidencia.

Para consumir JSON por stdout:
`python scripts/run_tests.py --json --preflight --coverage -- -q --continue-on-collection-errors tests`.

Las ejecuciones parciales mediante -m/-k siguen disponibles para desarrollo,
pero no acreditan la validación integral.
