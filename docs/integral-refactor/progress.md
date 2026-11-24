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

## Contrato de inferencia y aplicabilidad M1/M2
- M1 distingue generacion real de columnas y padding de ausencias, verifica perfil completo, marca salidas fisicamente invalidas sin alterar estimaciones historicas. Normalizacion numerica v2 comun a inferencia y entrenamiento. M1/M2 comparten comprobacion de perfil marginal.
- 91 aprobadas, 4 omitidas (20261124T210254Z_d10ad5522e724aac819c0c302c58aac0); tests GLOBAL de pickle resuelven ambos nombres historicos. No se probaron bundles reales ni se certifica incertidumbre independiente.

## Auditoria de separacion espacial
- Calculo de distancias geodesicas con BallTree por observacion y fold, usando TRAIN completo y elegible. Controles de cobertura, grupos, distancia minima configurable, escenas y procedencia compartidas; cribado conservador de solapamiento de parches por distancia entre centros, no certificacion de geometrias.
- EDA real M1: minimo 153.57 m, 24 muestras TEST con TRAIN a <=1 km, ninguna <=100 m. M2: minimo 1023.33 m, 0 posibles solapamientos de parches de 640 m, 547 casos TEST con escena compartida; 0 hashes de imagen compartidos. No implica independencia externa. No se cambiaron folds ni se ajustaron modelos.
- Evidencia M1 results/model_1/eda/eda_20261124T210434_548568Z_cedde0b5; M2 results/model_2/eda/eda_20261124T210444_094402Z_864e808b; hashes verificados. 35 aprobadas, 1 omitida (20261124T210403Z_801d49ab07ff4d1aa57c35e52bc89495). Pendiente protocolo prospectivo de holdout, auditoria de calibracion e inner folds, independencia documental.

## Preboot y dependencias por operacion
- Arranque con dependencias core, sin exigir ni instalar extras opcionales; no se eliminan .gitkeep. M2 no requiere torch/TabNet. Arranque interactivo real main.py, ENTER y 0, exit 0, sin advertencias ni errores, con GTX1650 detectada y XGBoost CUDA comprobado. M1 aun requiere instalar torch/TabNet para competencia completa; adquisicion nueva requiere pystac-client. Log results/diagnostics/interactive_startup_smoke.log.
- Regresion 66 aprobadas, 1 omitida (20261124T210530Z_6d2492b413044a1380ac0fc45ea30e95). No se realizaron instalaciones de extras ni push.

## Separacion de demostracion y promocion
- La evaluacion de los diez casos conocidos es exclusivamente comparativa; no cambia candidato pendiente ni modelo activo. API y CLI bloquean promocion sin holdout independiente. Comparacion requiere mismos casos, etiquetas y predicciones finitas; los gates rechazan metricas invalidas. No se activo ni entreno ningun modelo.

## Entradas del EDA local
- Laboratorio resuelve instantanea local M1 segun configuracion y manifiesto M2 congelado o expandido. No reconstruye ni descarga datasets desde diagnostico; entrada ausente bloquea, sin fallback de cohorte. No acepta archivos de otro workspace.

## Regresion integral tras contratos, arquitectura y EDA local
- 515 passed, 7 skipped, 8 warnings, 0 fallos/errores; 53% cobertura; 522 casos totales. Evidencia results/tests/20261124T210954Z_8227556d2f024c6c9a439e49f7688abb (summary, JUnit, coverage XML); 102.922 s runner. Ruff sin incidencias; 590 archivos, ninguno >300 lineas.
- Trazabilidad: 18 criterios parcialmente verificados, 1 parcialmente implementado, 38 pendientes. No se considera concluida la implementacion integral. Sin entrenamiento real, promocion, push ni workflows remotos. Manifiesto congelado M2 conserva SHA256 original.

## Auditoria de procedencia GEOROC
- Original CP1252: 25316 filas (23910 WR, 491 GL, 915 desconocidas). Filtrado UTF8: 23910 WR, identidad reconciliada por ID integral con sufijo .0 historico, 0 IDs perdidos/extra, 109 columnas quimicas comparadas sin diferencia numerica. 3221 Li informados, 8 ceros de significado no certificado, 0 tokens censurados. Metodo de Li y CRS no declarados.
- Armonizacion candidata aislada: 2999 de 23910 filas, mismo recuento historico; 2429 FeOT convertidos, 496 Fe2O3 originales, 74 sin hierro disponible. Conserva ID, cita, posicion de fila, Li original, unidad, metodo y CRS desconocidos, Fe original, auditoria de censura y hashes. Se corrigio orden de columnas de archivos heterogeneos, filtros fisicos y conteo de descartes. Evidencia results/diagnostics/georoc_source_contract. Instantanea historica sin cambios. Pruebas focalizadas 9 aprobadas (incluye reconciliacion .0) sin fallos; no entrenamiento ni push.

## Reconstruccion espectral v4 local
- Importados y comprobados 1476 archivos de la cohorte congelada: 706 cubos red-edge, 706 sidecars y 64 escenas STAC. No se escribio en el clon historico. Verificacion de todos los hashes, coordenadas, escenas y QA; 31064 descriptores red-edge reconstruidos (706 x 44), diferencia maxima absoluta 0.0. Evidencia results/diagnostics/rededge_import.json y rededge_verified.json.
- La carga v4 y la firma de reanudacion ahora exigen verificacion local de cubos, matriz, manifiesto, QA y metadatos; no hacen adquisicion automatica. Correccion de denominacion QA 10 bandas (6+4). Fuente desconocida se distingue del nombre de cohorte historica. No se alteraron las 126 caracteristicas anteriores ni se ajusto un estimador.

## Metricas binarias y umbrales
- Contrato explicito para etiquetas 0/1, scores finitos [0,1], forma y clases. Monoclase bloqueada en CV; en demostracion permite precision descriptiva pero BA, ROC-AUC, AP, MCC y Kappa no definibles. Matriz TN/FP/FN/TP, accuracy, sensibilidad, especificidad, ECE10, Brier y log loss; tabla de fiabilidad con bins y soporte. Umbral no descarta scores invalidos; fallback monoclase se marca no disponible, sin objetivo favorable. 100 pruebas aprobadas y 3 omitidas, run 20261124T212023Z_b972b519ea3149d4b5d75fd77fe308aa. Pendiente umbral estrictamente interno por fold y comparacion de calibracion real.

## Integridad de checkpoints y evidencias OOF
- SHA256 del contenido de checkpoints versionados, escritura atomica con temporales unicos; formatos anteriores sin verificacion se ignoran y se recalculan si se autoriza ejecutar. Cobertura TEST exacta, indices unicos y sin solapamientos, validacion de forma y finitud. Metricas principales recalculadas al reanudar. M2 valida identidad, fold y etiqueta; tabla OOF por muestra y umbral; mapa por ID, escala 0-1 y bins fijos.
- Estado numerico separado de exportaciones: fallo de tablas esenciales excluye del ranking sin duplicar candidato; fallo de figuras se registra como evidencia parcial. 57 pruebas focalizadas aprobadas; commit ab50a93. No se entrenaron modelos definitivos. Falta enlazar individualmente checkpoints al manifiesto de datos/config y probar reanudacion de ejecucion real, baseline OOF identificado, umbral interno y holdout independiente.

## Concentraciones extremas M1: evidencia retrospectiva
- scripts/scientific/audit_m1_historical_extremes.py reproduce desde los CSV historicos 27240 registros OOF (6 algoritmos x (1120+3420)), valida cobertura, identidad, finitud, errores absolutos y RMSE/MAE/sesgo contra tabla publicada. Hashes en results/diagnostics/m1_historical_extremes/audit.json. Top10 por algoritmo y cohorte, SSE y sesgo en Li alto (Q75 de TRAIN por fold). Candidato 3420: top10 explican 87.28%-97.35% de SSE; todos los algoritmos muestran sesgo negativo en Li alto. No se atribuye causalidad; falta unir muestra a expediente quimico/analitico y validar un protocolo prospectivo. No se entreno ni cambio modelo.

## Regresion integral tras procedencia, espectro y contratos de entrenamiento
- 537 passed, 7 skipped, 8 warnings, 0 fallos/errores; 53% cobertura, 544 casos. Runner 226.453 s; pytest 224.46 s. Evidencia results/tests/20261124T212342Z_4de3e84e7e9c46509b3219b8d359efd8. 601 archivos de codigo, ninguno >300 lineas. El script de extremos se verifico sobre evidencia real por separado. Sin push ni workflows remotos, sin entrenamiento definitivo o promocion.

## Umbral operativo y baseline M2 sin TEST
- Protocolo canonico grouped_development_oof. Cada fold obtiene scores internos con grupos aislados, politica de calibracion compatible y mismo endpoint Li del TRAIN exterior. Optuna aprende cuantil en cada TRAIN interno; no omite monoclases. Se registra corte, scores internos y metricas TEST a 0.5 y corte seleccionado; reanudacion reconstruye la seleccion desde evidencia sellada. Seleccion final solo con OOF del desarrollo al endpoint final, sin afirmar independencia externa. Comparacion de umbrales acelerada con conteos acumulados. Metricas TRAIN descriptivas separadas. Baseline prior usa QA TRAIN, exporta OOF identificado. Firma incluye checkpoint, decision y adquisicion Sentinel2.
- Preflight real sin ajuste: 706 muestras, 690 elegibles, 15 splits internos exteriores y 3 de desarrollo final; minimo 30 positivos de validacion operativa, 23 en validacion Optuna; calibracion anidada al menos 17 positivos, 62 negativos. results/diagnostics/m2_operating_preflight.csv. 58 pruebas focalizadas aprobadas tras admitir indices de fold desde cero; Ruff limpio. No se cambia la politica legacy de calibracion por defecto. El endpoint Li se conserva definido desde el TRAIN exterior completo (QA de imagen afecta ajuste, no verdad geoquimica); no se afirma que sea la mejor eleccion.

## Expediente de procedencia de los errores extremos de M1
- Verificado SHA del combinado historico 3420 y CSV OOF; reconciliacion de los seis algoritmos con Li, grupo, fuente y cinco checkpoints originales. 2740/2744 GEOROC con identidad unica por nombre de muestra, Li y coordenadas a ocho decimales; cuatro ambiguas, no se asignaron arbitrariamente. Los 16 casos distintos del top10 de los seis candidatos tienen ID unico; nueve Macusani [30009], cinco San Rafael [25383], dos otras publicaciones. Todos los nueve Macusani caen en fold5: maximo Li TRAIN 696 ppm, cero >=1000 ppm y cero muestras confirmadas de esa publicacion; Li TEST 930-4160 ppm. Los 16 no tienen Rock_type en la entrada historica pese a Rock_class de origen. Evidencia de extrapolacion y perdida de metadato; no prueba de causalidad. Cita es codigo, no referencia bibliografica completa; metodo y CRS siguen desconocidos. No se eliminaron extremos ni se reentreno. Script reproducible scripts/scientific/audit_m1_extreme_provenance.py, resultados results/diagnostics/m1_extreme_provenance.

## Registro prospectivo y evaluacion externa no promotora
- CLI separada register/score/evaluate. Registro ciego a etiquetas, compromiso SHA del custodio, hashes de modelos/sitios/exposiciones/revision e imagenes; rechazo por ID, fuente, sitio, grupo, distancia y escena/hash M2. Score local de bundles confiables sin adquisicion; M2 exige endpoint Li comun. Sellos de predicciones y revelacion; cohortes completas, abstenciones conservadas, metricas emparejadas y bootstrap por grupos. No re-evaluacion tras revelar ni promocion. Gate valida soporte y dominios numericos. 15 pruebas focalizadas aprobadas, incluido flujo completo M1/M2 v3 sintetico; Ruff aprobado. Protocolo y limitaciones docs/integral-refactor/prospective-holdout.md. Falta inventario completo de exposicion con identidades originales, cohorte nueva y custodia/revision independientes, scoring v4 real y aprobacion metodologica del gate. Los hashes no certifican independencia humana. No se registraron datos externos reales.

## Regresion integral y contraste documental
- 548 passed, 7 skipped, 8 warnings, 0 fallos/errores, 54% de cobertura y 555 casos. Runner 191.125 s; pytest 189.35 s. Evidencia results/tests/20261124T214435Z_7cc95ebaf6c341e3977736e664bcf83a (summary, JUnit, coverage XML). Ruff sin incidencias, 610 archivos de codigo, ninguno supera 300 lineas.
- Relectura completa de Procedimientos y Codigo en los diagnosticos M1 y M2, y documento 06 de presentacion. Persisten secciones de tablas y documentos complementarios por revisar. No se confunden metricas medias por fold, OOF globales y evaluacion independiente.
- Trazabilidad actualizada: 38 parcialmente verificados, 1 parcialmente implementado, 18 pendientes; cero criterios cerrados integralmente. Cuatro criterios M2 antes pendientes incorporan evidencia de commits anteriores sobre separacion fisica, matriz, estados y persistencia; no se presentan como nuevas implementaciones.
- Incidencias resueltas: cuatro tests focalizados de indexacion de folds desde cero y limite de longitud del comando Windows al escribir el nuevo modulo (escritura dividida); regresion final exit 0. No se entrenaron modelos definitivos ni se promovio un candidato. Sin push ni ejecucion remota de CI.

## Variante espectral fisica V5 experimental, no entrenable por menu
- 164 descriptores (120 base sin seis duplicados NBR/nir_swir22_nd y 44 red-edge), radiometria STAC por cada una de las diez bandas, nodata declarado, calculo en float64 antes de volver a float32, mascara conjunta, sin relleno, textura solo vecinos validos, divisiones indefinidas NaN con soporte explicito. SCL opcional y verificacion de codigos categoricos; no se infieren mascaras desde resumenes agregados. Politica provisional 95% pixeles y soporte de cada descriptor; epsilon 1e-6, SCL no exigido para auditoria offline. Sin cambios v3/v4, config ni modelos.
- Auditoria real 706/706 fuentes verificadas por SHA, sidecars, escena y coordenadas; 706 matrices finitas, 690 elegibles, 16 conservadas con advertencia. 36 DN finitos 0 enmascarados segun nodata STAC; 2538 reflectancias cero reales y 982018 valores negativos conservados, no se recortan. 164 descriptores cambian respecto v4 en alguna muestra. Resultados results/diagnostics/m2_physical_v5_706_precision; la carpeta m2_physical_v5_706 es la extraccion preliminar anterior a correccion de precision. SHA matriz v5 240f8ba64e52f9d3a6b0d13abdadd29220b025a61a023d34b454bc18462e990c. NPY historico permite verificar forma y contrato de adquisicion pero no certifica georreferencia independiente. Sin SCL por pixel ni afirmacion de mejora. 9 pruebas focalizadas aprobadas; Ruff limpio.

## Reconciliacion quimica y recuperacion litologica M1
- Comparacion por nombre, Li, coordenadas a 8 decimales, 23 campos quimicos (10 oxidos, 12 trazas y suma), coincidencia de ausencias y convencion Fe; tolerancias rtol 1e-12, atol 1e-10. Resuelve los 4 ambiguos del analisis anterior: 2744/2744 muestras GEOROC preparadas con ID unico; bruto 2913/2986 unicas y 73 con discrepancia quimica (no ingresan en las 3420 preparadas). No se asigna ID por mera similitud. 2327/2744 recuperan litologia reportada no ambigua (2488 filas brutas); 417 no se completan arbitrariamente. Candidato aislado conserva 4765 filas brutas, 3420 preparadas, 244 grupos, 33 predictores; solo cambia Rock_type, y pipeline comprueba mismas etiquetas, filas, grupos, esquema y demas predictores. No se ajusta modelo. SHA candidato a43d5cc288cf5225576247820fedf792df96f90216efa9360603d445309144a4.
- Balance de oxidos, tolerancia descriptiva 0.05: GEOROC 2744/2744 con suma 10; Mamani09 602/676 con suma 10 y 676/676 al incluir FeO reportado (81 con FeO). No prueba equivalencia de metodos, ni cambia QC 94-102. Cuatro filas en dos pares exactamente iguales predictor+target, permanecen sin eliminar mientras se verifica identidad fisica. Evidencia results/diagnostics/m1_source_reconciliation. 17 pruebas focalizadas aprobadas (11 M1 + 6 V5); Ruff aprobado. Incidencia de vista pandas de solo lectura en primer intento corregida con copia explicita; repeticion real completa.

## Contrato de entrada satelital para candidatos nuevos
- make_patch_contract declara seis bandas en orden, 640 m/32 px, 20 m, Sentinel-2 L2A y DN. load_verified_patch comprueba sidecar unico, SHA, escena, coordenadas, forma y metadatos; TIFF comprueba CRS proyectado en metros, transformacion norte-arriba, resolucion, centro y descripciones semanticas, lee con mascara. NPY valida declaracion, no georreferencia independiente. Inferencia usa una sola lectura y extrae con preparacion compartida; candidatos nuevos guardan contrato en bundle y metadata, viejos sin contrato conservan comportamiento historico y reportan geometria no verificada. Adquisicion nueva comprueba CRS metrico, forma y bandas con datos; guarda hash, geometria y convencion, y rechaza drift de cache con contrato. No se modifican sidecars historicos. 28 tests focalizados (incluida equivalencia NPY/TIFF con mascara, permutacion, hash, corrupcion y resolucion). No se hizo adquisicion real, adaptacion de resolucion ni inferencia de bundle real nuevo. Advertencia de operador affine corregida con calculo directo de centro.
