# Cierre de implementación: arquitectura de tests y laboratorio de diagnóstico

Rama de trabajo: feat/training-refinement.

## Alcance

- Conservación de los 296 módulos anteriores, distribuidos por propósito de sus
  aserciones. Las rutas originales constan en test-relocation-purpose.json.
- Fixture compartida en tests/conftest.py, ocho categorías primarias y seis
  marcadores transversales. Las pruebas de infraestructura añadidas respetan
  el layout canónico y se asocian a los criterios de arquitectura relevantes.
- Preflight estático, una única ejecución pytest, reconciliación de colección
  y JUnit, cobertura real, protección de artefactos e identidad Git.
- Informes de categoría, responsabilidad, ámbitos y 166 criterios.
- Sin cierre automático de criterios científicos por resultados de tests.

## Navegación aprobada

5 > 3, Ejecución de tests y simulaciones:

1. Validación del software
2. Comparaciones de modelos
3. Evaluación de tratamiento de datos
0. Volver

La opción 1 no ajusta ni activa modelos productivos. La opción 2 selecciona M1,
M2 o ambos, procesa OOF guardado y comprueba la identidad de escenarios
completos antes de reutilizarlos. Un manifiesto sin identidad verificable no se
reutiliza: la generación de escenarios pendientes requiere confirmación.
No se reconstruye evidencia OOF mediante perturbaciones de inferencia.
La opción 3 selecciona M1, M2 o ambos; mantiene evaluación general
completa/piloto y la campaña focalizada M1 aprobada, separada de la general.
El acceso OASIS y la orquestación Ejecutar todo fueron retirados del menú
activo; se preservan las funciones científicas independientes.

## Verificación pendiente en el equipo del investigador

1. Actualizar la rama de trabajo sin hacer merge a main.
2. Iniciar LithiumScope y ejecutar 5 > 3 > 1.
3. Revisar results/tests/<run_id>/verification_report.md, summary.json,
   architecture.json, criterion_evidence.json, junit.xml y coverage.xml
   (cuando esté disponible), y los logs de la misma ejecución.
4. Corregir fallos, errores de recopilación, omisiones e incidencias de
   integridad antes de certificar la regresión integral.
5. Verificar navegación, alcance M1/M2, cancelación y comparación guardada
   en 5 > 3 > 2 y 5 > 3 > 3 con los artefactos locales, sin ejecutar nuevos
   experimentos no autorizados.

Los cambios en GitHub no prueban que pytest o los modelos hayan funcionado
en el entorno local. No se ejecutaron workflows en esta implementación.
La auditoría del árbol y catálogo no sustituye a la ejecución de la suite.
