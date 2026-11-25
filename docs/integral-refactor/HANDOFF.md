# Continuidad del desarrollo integral — punto de entrada obligatorio

Repositorio JaimeArriagadaRosas/LithiumScope, rama feat/integral-scientific-refactor.
Base e12b877699fe97f301e2c87d07f4833084e1306f; primera publicación b0c385788bb48b1c5079a4a1c9bfbad17ac019c2.
Primera CI aprobada: https://github.com/JaimeArriagadaRosas/LithiumScope/actions/runs/45511841300 (7/7).
Desde ese punto, commits intermedios con [skip ci] exclusivamente en esta rama, sin desactivar Actions globalmente.
Al cierre retirar la opción temporal y ejecutar CI normal; no modificar main, releases ni modelos activos sin autorización.

## Conteo y estado

| Diagnóstico | Total | Cerrado con evidencia acotada | Abierto |
|---|---:|---:|---:|
| M1 | 27 | 0 | 27 |
| M2 | 30 | 0 | 30 |
| Repositorio, instrucciones textuales | 109 | 6 | 103 |
| Total textual (requisitos superpuestos) | 166 | 6 | 160 |

Consola L39-1261: especificación adicional abierta, no incluida en el total textual.
57/57 científicos tienen implementación principal incorporada; la aceptación científica continúa abierta (documentos, datos, pruebas, experimentos y evaluación independiente). No equivale a una aprobación científica.
La arquitectura no tenía numeración original: IDs ARQ-001..109 introducidos para controlar el cierre.
Los requisitos se superponen, pero a solicitud del usuario se cuentan por unidad para medir implementación principal: 138/166 (83,1 %), 28 parciales. Véase implementation-status.md. No confundir con cierre científico ni con cobertura.
E1/E2 está vacío (0 bytes) y el checklist general del seminario contiene otras tareas académicas fuera de este alcance.
Última suite local en b0c3857: 647 aprobadas, 7 omitidas, 8 warnings, 0 fallos, 57% cobertura de líneas, 670 archivos <=300, sin directorios Python >20.
No se atribuye CI aprobada a commits posteriores omitidos por [skip ci]. Los 166 criterios cuentan ahora con módulos de pruebas relevantes asociados en criterion-tests.json; no implica cobertura exhaustiva ni cierre.

## Lectura inicial en un chat nuevo

1. Este HANDOFF.md y requirements/repository-requirements.md (109 instrucciones originales con ID, evidencia contextual y siguiente comprobación).
2. traceability.md (57 diagnósticos científicos, estado y evidencia), requirements/m1-diagnostic.md, requirements/m2-diagnostic.md (secciones de diagnóstico originales con redacciones de privacidad), requirements/console-contract.md.
3. implementation-status.md (conteo por criterio de implementación, separado de aceptación); phase-status.md (lote más reciente, pendientes antes de solicitar la regresión); progress.md (resultados, hashes, sesiones, commits); module-relocation.json (rutas actuales vs históricas); sources.md (hashes de originales); controlled-experiment-design.md y prospective-holdout.md.
4. plan.md conserva historia de planificación; sus estados iniciales no son la fuente de estado actual. Usar esta matriz y actualizar por criterio, no inferir cierre de la cantidad de tests.
5. Revisar HEAD remoto antes de editar, usar cuenta Jaime y cambios agrupados con tests específicos. Guardar en Git los cambios de código y documentación, no los datasets/artefactos binarios/credenciales.

## Desglose del tercer diagnóstico
- Menu interactivo: 6 instrucciones (2 verificadas).
- Arquitectura M1: 15 instrucciones (0 verificadas).
- Arquitectura M2: 20 instrucciones (0 verificadas).
- Renderizado de PDF: 9 instrucciones (1 verificadas).
- Arquitectura General: 2 instrucciones (1 verificadas).
- Datasets: 6 instrucciones (0 verificadas).
- Experiments: 7 instrucciones (0 verificadas).
- Core: 8 instrucciones (0 verificadas).
- Algoritmos: 3 instrucciones (0 verificadas).
- Recursos: 12 instrucciones (1 verificadas).
- test: 21 instrucciones (1 verificadas).

## Prioridades ordenadas de cierre

- Arquitectura y datos: resolver ARQ de datasets, experiments, core y prediction; conservar aliases serializados reales y CLI, verificar dependencias circulares, no separar por límite sin responsabilidad. M1/M2 independientes y reporting solo render.
- EDA: adquisición autorizada solo cuando falten datos, preparación común, validación de hashes, EDA compatible reutilizable y automático en entrenamiento, análisis por fuente/fold sin contaminar TEST. No decidir a partir del TEST.
- M1: verificar censura/unidades/duplicados y Mamani09, procedencia GEOROC, variantes de predictores y preprocesamiento; TabNet espacial requiere PyTorch instalado y ajuste real; seis familias, Optuna, baselines, robustez coherente, curvas, incertidumbre y errores extremos. No declarar superioridad sobre la memoria OASIS con protocolos distintos.
- M2: QA y georreferencia, SCL local ausente, v4 170 predictores verificados offline, v5 164 descriptivos no activado, comparar representación/QA/expansión de manera controlada. Separar Li cutoff vs score, folds internos/calibración espaciales, todas las familias, umbral y gate; no inferir depósito de una puntuación.
- PDF: verificar informe histórico y renderer separado sin nuevos cálculos, revisar gráficos y estructura; no render definitivo sin permiso. Inferencia histórica 20/20 exacta, no sustituye validación independiente.
- Recursos/logs/tests: benchmark sintético y CPU/GPU probes disponibles, ampliar medición comparativa real y concurrencia, correlación de eventos, regresión/coverage/E2E; mantener pruebas útiles. No aumentar número de tests sin cubrir comportamiento.
- Experimentos: protocolo preespecificado en controlled-experiment-design.md, curvas 100 particiones listas; ejecutar diagnósticos controlados solo con autorización y datos disponibles; no promover por OOF. Evaluación externa auténtica pendiente, no se puede simular como independiente.
- Cerrar cada criterio individualmente con test, resultado, ubicación de evidencia y commit, o registrar bloqueo real. Mantener historial de los estados. Repetir matriz CI al finalizar.

## Diagnóstico temporal y trabajo sin Desktop Commander

El recolector 5 > 4 está incorporado desde a5036ee y ampliado en 46a21ea: sintaxis completa, Ruff, arquitectura, pytest aislado con cobertura y JUnit, evidencia asociada por los 166 criterios, inventario y preflight. Todavía no se ejecutó sobre esta rama con los cambios nuevos. Revisar phase-status.md y agotar las implementaciones posibles desde GitHub antes de pedir intervención. Entonces el usuario ejecutará git pull y 5 > 4 y compartirá development_report.json revisado. El informe debe recopilar estado/hashes, versiones, QA, preflight, recursos, cobertura y artefactos faltantes; no transmitir secretos, datos crudos, rutas personales, logs completos, modelos, ni ajustar/activar modelos. Si hace falta, pedir logs específicos o prueba nueva. Quitar opción temporal y paquete de desarrollo al terminar.

## Restricciones y bloqueos

Los archivos locales ignorados por Git (datasets históricos, parches, releases, auditorías y logs) no se trasladan por git push. Las rutas relativas y hashes en progress.md son referencias a evidencia local, no archivos que el conector pueda recuperar automáticamente. Los originales completos permanecen en la carpeta local de seminario; se incorporaron únicamente sus secciones de requisitos relevantes, con redacciones, y anclas de consola. No se publicaron gráficos base64, datos, binarios ni consola cruda. Los 57 criterios científicos conservan el estado parcial hasta verificar aceptación individual.
No escribir en el clon histórico LithiumScope-Scientific-Lab. No usar conexión remota salvo necesidad específica del usuario. No ejecutar entrenamientos definitivos, promoción, publicación, ni render final sin autorización. No presentar el desempeño histórico como nueva evaluación. No inventar evidencia ausente.
