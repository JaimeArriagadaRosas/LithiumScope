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

## EDA real preliminar (sin entrenamiento)
- M1: 4765 filas -> 3692 con Li valido -> 3420 tras QC; 2744 GEOROC + 676 Mamani09 bootstrap, 33 predictores y 244 grupos. 1073 excluidas por validez de Li (Mamani09), 272 por QC (242 GEOROC, 30 Mamani09). 6 registros censurados en la instantanea combinada; no representa toda la fuente bruta GEOROC. Li maximo GEOROC 4160 ppm, Mamani09 100.6. Cinco folds: perfiles solo de TRAIN, sin aprendizaje estadistico; no holdout independiente.
- M2: 706 filas, 170 predictores, 66 grupos, QA conjunto 690 elegibles, 16 excluidas solo de TRAIN. Umbrales geoquimicos por fold (21.45, 21.00, 22.40, 21.40, 21.90 ppm). No se certifica calidad de roca en campo ni se reextraen las cuatro bandas red-edge originales.
- Evidencias: results/model_1/eda/eda_20261124T204130_035846Z_0444675e y results/model_2/eda/eda_20261124T204141_518008Z_5998c733; 15 y 13 artefactos respectivamente, integridad SHA256 verificada. Corresponden a la version del codigo en esa ejecucion, no a cambios posteriores.
- Tests focalizados 6 aprobados, ninguno fallido (20261124T204115Z_457590a79706407993a3184beafd890c). Falta completar EDA de desarrollo/holdout, protocolos de ablation, auditoria fuente bruta, redundancia, distancias y diagnosticos.

## Inferencia espectral unificada
- M2 individual e integrado comparten un unico evaluador. No se rellenan features faltantes con ceros; abstencion y diagnostico por caso. Perfil incompleto no se declara OK. V4 requiere metadatos de escena, coordenadas, bandas originales y red-edge de la misma escena con QA; adquisicion real no ejecutada.
- Regresion: 66 passed, 4 skipped (20261124T204500Z_e3d2fcca25e9402ebf65f1079e236b6f). Pendiente comprobar caso real v4, versionado de calidad y aplicabilidad M1.

## Regresion integral tras importacion y EDA real
- 485 passed, 7 skipped, 8 warnings, 130.99 s, cobertura 52%, sin fallos ni errores; 492 tests totales. Run 20261124T204551Z_e0de075182b943658f3f18853edc6236; summary.json, junit.xml y coverage.xml locales.
- Ruff sin incidencias; 563 archivos de codigo, ninguno por encima de 300 lineas.
- Pendientes: reorganizacion de algoritmos y validacion, adquisicion independiente completa, auditoria de fuentes brutas y protocolos, EDA de desarrollo y holdout, lectura integral de documentos complementarios. No se declara cierre cientifico.
- Restriccion vigente: commits locales exclusivamente, sin push ni ejecucion remota de workflows.

## Arquitectura de preparacion, validacion y algoritmos
- 52 modulos redistribuidos por responsabilidad: lector tabular compartido, preparacion M1 y M2, validacion espacial, algoritmos separados, preprocesamiento, optimizacion, ejecucion y calibracion. Sentinel-2 y red-edge en tools/acquisition/sentinel2. Se actualizaron imports internos, pruebas y rutas de firma.
- Compatibilidad joblib explicita en los dos modulos historicos de FeatureSchema y TabNetRegressorAdapter; pendiente comprobar bundles reales. Se conservan las entradas publicas de competition/trainer y comandos de diagnostico historico.
- Regresion local 485 aprobadas, 7 omitidas, 8 warnings, cero fallos (20261124T205934Z_5ed962a6a6b74676be9dfc16f42bc7e9). Ruff y limite 300 lineas OK. No push.

## Contrato tabular y geoquimico
- Lector compartido valida encabezados, filas y delimitador; registra formato, codificacion y hoja. Excepcion de cuatro encabezados vacios restringida al SHA256 de Mamani09 (1788 x 101, CP1252).
- EDA real M1: 1073 descartes por Li ausente/no interpretable; QC 160 bajo minimo, 3 sobre maximo, 109 faltantes. 3420 filas, 33 predictores y 244 grupos conservados. Derivados v2: 7 ausencias adicionales, sin divisiones invalidas ni negativos; v1 mantiene formulas anteriores. Evidencia results/model_1/eda/eda_20261124T210147_943982Z_60f8d415.
- Pruebas focalizadas 79 aprobadas, 0 fallos (20261124T210128Z_5a61039109644304808379ed6bb9cb1c). Pendiente confirmar unidades, censura y equivalencia quimica de sumas por fuente. No se ha entrenado un candidato.
