# Progreso de implementacion

- Base: e12b877699fe97f301e2c87d07f4833084e1306f.
- Commit documental: 0cdbed9.
- Infraestructura inicial: scripts/run_tests.py separa logs y evidencia JUnit, asigna IDs unicos, usa directorios temporales y propaga el codigo de salida. No reemplaza aun el logger de la aplicacion.
- Verificacion: py_compile aprobado; tres pruebas unitarias con unittest y subprocess simulado aprobadas (XML ausente o invalido, ejecuciones distintas y codigo de salida, fallo de lanzamiento). Falta ejecucion pytest real y pruebas persistentes de integracion.
- Baseline de pytest: instalacion iniciada con el codigo original; revisar results/tests/baseline/install.log, pytest.log, junit.xml y exit-code.txt. No declarar baseline aprobado sin resultados.
- Pendiente: lectura completa de los diagnosticos prioritarios, completar trazabilidad y contrastar con implementacion real; no mover paquetes ni alterar protocolos cientificos antes de ello.
- No se modificaron modelos, releases ni el clon de defensa. Sin push ni merge.

## Baseline y primer bloque CLI
- Baseline original: 442 passed, 11 skipped, 8 warnings, 56.70 s, exit 0. results/tests/baseline/junit.xml.
- Consola: menu principal 1/2/5, suborden de prediccion, pausa de preboot, cancelacion de entrenamiento y confirmacion de instalacion. Entry point instalado unificado con ciclo de vida.
- Test de contrato: 8 fallos reproducidos antes; regresion posterior 18 passed, 5 skipped. Run 20261124T200155Z_113122b142a14e7e8992d4ffdb17208b.
- Pendientes: consulta de resultados, laboratorio, presentacion detallada, progresos, EDA, reorganizacion y correcciones cientificas.
- Se revisaron los 57 parrafos de diagnostico y la especificacion de consola; lectura de antecedentes y complementos pendiente.

## Consulta de evidencias
- Resultados de solo lectura: PDF y XLSX existentes, modelos activos y candidatos, manifiestos EDA, tests y simulaciones. PDF: comprobacion estructural preliminar; no certifica toda su integridad. Pendiente validar hashes declarados y compatibilidad EDA efectiva.
- Acceso separado conservado: python -m lithiumscope.cli.candidate_admin bundle | evaluate | promote --confirm. La promocion exige opcion explicita; no se ejecuto ninguno de estos comandos. Esta via CLI es una decision de implementacion para no agregar opciones permanentes.
- Regresion: 13 passed, 0 skipped, 0 failed. Run 20261124T200348Z_9049c7f6d1f44a45bf820685b5f77848.

## Contratos científicos iniciales
- M1: concentraciones físicas, política cero explícita, edad como esquema fijo antes de CV. M2: filtro físico independiente.
- Tests antes: 4 fallos. Después: 36 aprobadas sin fallos (20261124T200524Z_10aa90efcf90439b91fe4522f4c40b61).
- Commit 73ddaee. No se cambian modelos históricos; falta validar datos reales y semántica por fuente.

## EDA descriptivo M1/M2
- Ambos reutilizan sus preparadores/lectores de entrenamiento y no entrenan estimadores.
- M1: perfiles antes/después por fuente, etapas, filas excluidas, categorías, estadísticas, correlaciones y gráficos.
- M2: características configuradas, procedencia de manifiesto, perfiles y calidad de escena (no calidad local del parche).
- Firmas de datos/código/configuración y hashes por artefacto. Reutilización de evidencia descriptiva intacta; reconstrucción si se altera.
- Pruebas sintéticas: 10 aprobadas (20261124T200728Z_38175446fb484585af1d10bf87446c7c).
- Aún no es EDA de desarrollo independiente; no se utiliza automáticamente para selección o ajuste.
- No se adquirieron datasets ni se ejecutó EDA real. M2 v4 está condicionado a los manifiestos y matrices congelados que no están versionados.

## Laboratorio de diagnostico
- Submenus EDA M1/M2/ambos, tests/simulaciones/ambos y salud solo lectura. Catalogo de simulaciones inferenciales con bloqueo por recursos faltantes.
- Se conserva acceso manual historico via python -m lithiumscope.cli.legacy_simulations; acceso CLI separado, no opcion permanente.
- Regresion: 17 passed, 0 failed (20261124T200849Z_90df3b61a5a6410a845789f1487868ff).
- Pendiente: catalogo completo de protocolos, progreso real por test/operacion, auditoria mas profunda de calidad y adquisicion M2.

## Recursos y logging por sesion
- resources/: deteccion de CPU/afinidad/RAM/disco/GPU, comprobacion de compatibilidad por backend, presupuesto y monitorizacion por operacion. Politica en config/resources.yaml; concurrencia exterior de 1 para evitar paralelismo anidado no medido.
- core/logging/: sesiones y operaciones identificadas, JSONL estructurado, errores separados, retencion protegida, logs de entrenamiento y pruebas integrados. Fachadas historicas conservadas por compatibilidad.
- Pruebas focalizadas: 14 de recursos y 31 del bloque combinado, sin fallos. Pendiente medir rendimiento real, verificar multiproceso y completar trazabilidad.
- Inventario historico en modo lectura: M1 training_combined.csv y M2 manifiesto de 706 filas con SHA256 igual a config, matriz v4 y auditorias existentes. El manifiesto M2 contiene rutas absolutas al clon anterior; falta importacion independiente con comprobacion de integridad y resolucion de rutas. No se copio ni modifico ningun archivo historico.

## Regresion integral y CI
- 469 passed, 11 skipped, 8 warnings, 82.68 s, cobertura 51%, sin fallos. Run 20261124T201739Z_f54230fee57c488eb931b0756c45a250; summary.json, junit.xml y coverage.xml locales.
- +27 pruebas aprobadas frente al baseline de 442. Una regresion anterior se interrumpio por selector historico (opcion 4) incompatible con la numeracion aprobada (opcion 2); actualizado el test y comprobada ausencia de selector de archivos.
- CI ahora ejecuta runner con cobertura y sube evidencia en matriz existente. No se ha ejecutado CI remoto.
- Lectura: secciones de diagnostico de M1 y M2 (57 criterios) y especificacion del repositorio revisadas; antecedentes, tablas, codigo comparativo y documentos complementarios aun no revisados de principio a fin. Checklist y diagnostico general leidos; E1/E2 diagnostico vacio.
- Cierre de bloque, no cierre de refactorizacion: EDA real y holdout independiente pendientes. No se han entrenado modelos reales ni promovido candidatos.

## Importacion independiente y cache espectral
- 1425 archivos copiados y verificados; 706 parches originales, manifiesto congelado intacto, matriz v4 y QA, fuentes M1. Evidencia: results/diagnostics/historical_import.json. No se copiaron modelos ni se escribio en el clon anterior.
- Cache por contenido y matriz SHA256; IDs, orden de etiquetas y grupos, esquema, imagenes, configuracion y codigo. Regresion 29 passed, 1 skipped (20261124T203905Z_bb6c4422eea24df5be215a1a5955cc73). Verificacion real 706 x 172 (170 predictores), 690 elegibles QA seis bandas. Pendiente identidad red-edge original y pruebas de concurrencia/interrupcion.
- Restriccion expresa: commits solo locales, no push ni workflows remotos.

## Preparacion M1 versionada
- Censura analitica v2 y categorias semanticas v2, conservando version 1 para esquemas historicos. EDA incluye censura por observacion y categorias antes/despues. 27 pruebas aprobadas (20261124T203956Z_cee575ed0ecf4797a2305d83d479379c).
- Pendiente validar convenciones decimales y politica de ceros por fuente; no se afirma superioridad ni equivalencia metrica.

## Separacion del renderer PDF
- reporting/ trasladado a la raiz. Excel, graficos y mapas en prediction/exports/; decision integrada y evidencia de seleccion en prediction/analysis/. Estadisticas de aplicabilidad, clasificacion y concordancia calculadas antes del renderer. Conservada la interfaz write_pdf_report de prediction/report.py y la numeracion del informe.
- Regresion 50 passed, 4 skipped, sin fallos (20261124T204255Z_33955736185646f69b496fd90f75a2fe). Falta prueba visual con modelos reales, reorganizacion interna restante y separar decisiones de presentacion historicas aun existentes.
