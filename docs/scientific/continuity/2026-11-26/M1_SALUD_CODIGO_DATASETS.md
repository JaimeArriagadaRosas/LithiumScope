# Salud del código — datasets

Estado: implementación arquitectónica, pendiente de validación local integral.
Base 1fbd354ae90300c219ab4b393466cc3b2a1eebb5. Usuario aprobó la propuesta
y pidió arquitectura primero, validación al final, sin workflows.
No ejecutar entrenamiento, recuperación espacial, adquisiciones ni simulaciones.
No modificar main, modelos guardados, configuración científica o datos.

## Cambios
- Inventario exacto de Git: 17 archivos Python en raíz (16 compatibilidad
  y __init__.py); después, solamente __init__.py. Se agrupan shims
  históricos en _legacy/ sin importarlos de forma anticipada; se agrega
  esa ruta a __path__ de su paquete. Es una capa temporal de compatibilidad,
  no implementaciones duplicadas. Se conserva identidad de módulos y
  nombres históricos para importación y deserialización, pendientes de prueba.
- GEOROC: paquete adapters/georoc; ingestión, esquema, transformación,
  química/censura/hierro, selección de materiales y filas, normalización,
  postprocesamiento y Dataverse. Auditoría de ingestión en provenance.
  Orden de operaciones y parámetros científicos conservados por extracción.
- Harmonization: coordenadas y combinación en módulos propios; __init__
  solo expone API. Deduplicación y reconciliación no cambian.
- Management: registro/estado y sources; M1 divide configuración/fuente
  congelada, adquisición, ensamblaje y coordinación. M2 frozen trasladado
  sin cambiar sus reglas. Mantener nombres históricos y dependencias
  inyectables de tests.
- Lectores CSV, Excel y API tabular separados; validación de encabezados
  en validation/tabular. Se conserva excepción de cabeceras del snapshot
  Mamani, codificaciones, metadatos, argumentos y comportamiento de consola.
- Validation: química, GEOROC y tabular; coordenadas y rutas generales
  se mantienen. Provenance: separar archivos enlazados y limpiar temporales
  de manifiestos ante fallo de reemplazo.
- Logging: inicio/fin de ingestión, resumen de recuentos, DEBUG por archivo,
  ERROR con traza ante fallo de chunk, warning preexistente por columnas
  ausentes. Namespace de normalización de faltantes bajo datasets.
  Mantener logger central y evidencias detalladas; no cambiar niveles
  globales ni añadir otro sistema. No se añadió logging a cada validador puro.
- Relocation map, auditor estático y pruebas de rutas/fingerprint adaptados
  a los paquetes canónicos y shims. Las huellas de código cambian, por lo
  que no reanudar checkpoints incompatibles ni sobrescribir evidencia.
- Tests/datasets: 19 módulos originales reagrupados en ocho categorías,
  tres archivos nuevos de pruebas (especialización/aliases/pickle,
  ingestión/transformación/logs, atomicidad del manifest). 30 archivos
  Python contando ocho __init__, 22 módulos test. No eliminar tests
  anteriores; verificar colección al final.

## Inventario y pendientes
Árbol de implementación 7696d308478939d299b1fedb454a2c3aaec00016:
101 archivos Python en datasets, incluidos shims y paquetes. Raíz 1.
Ningún directorio de datasets ni tests/datasets supera 20 archivos Python.
El mayor directorio funcional es adapters/georoc (14), los shims
de raíz suman 17 incluyendo __init__. No se debe interpretar incremento
de número de archivos como incremento equivalente de lógica científica.
No se han ejecutado pytest, workflows ni entrenamientos; no afirmar
cobertura actual, paridad científica ni compatibilidad certificada.
Meta propuesta datasets >=80% de líneas, revisar cobertura de ramas
críticas; no rebajar cobertura global. Revisar consumidores residuales
de aliases y artefactos históricos antes de retirar shims definitivamente.
El exceso de raw_recovery.py y tests/scientific/m1_expansion pertenece a
otro módulo y se mantiene pendiente.

## Cierre y menú 5
Opción 5 del menú principal = Laboratorio de diagnóstico; submenú de
tests y simulaciones tiene tests completos que invocan scripts/run_tests.py
con cobertura. No modificar ahora. Tras aprobación explícita del usuario
al terminar todos los módulos, mejorar esa opción: comprobación de
arquitectura/tamaño, imports, lint, suite completa, cobertura global y
por módulo, informe agrupado con fallos, logs y artefactos para corregir
por partes. No ejecutar simulaciones costosas por defecto. Validación
local tras git pull, no workflows hasta autorización. Después cerrar
rama y crear rama de entrenamiento desde ella, sin merge a main.

Todos los commits de esta intervención incluyen [skip ci] para evitar
CI automática por push.
