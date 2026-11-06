# LithiumScope: plan integral de entrenamiento cientifico (vNext)

Actualizado: 2026-11-06. Rama: feat/scientific-simulation-lab.
Estado: **investigacion y desarrollo**, sin modelos aprobados para release.

## Acuerdos y limites
- M1 y M2 ya existentes; NO crear un tercer modelo.
- Los entrenamientos del 2026-11-05 se preservan como experimentales y referencia cientifica; NO son el release nuevo.
- Mantener las rutas de descarga GEOROC/Mamani09 y Sentinel-2, filtros originales y menu intactos salvo cambios necesarios autorizados.
- Commits locales por etapa para volver atras; **sin push a GitHub**.
- No se modifica ni elimina el release historico ni modelos guardados.
- Desarrollar y evaluar etapas 1, 2 y 3, con tests y reportes. En **etapa 4** presentar evaluacion de posibles fuentes adicionales y **DETENERSE** antes de descargas/adaptadores nuevos, de cambiar definitivamente datasets o de entrenar modelos finales (etapa 5).

## Evidencia experimental previa
- M1: 4.765 filas combinadas; 3.692 Li valido; 1.120 despues de QC. 444 GEOROC + 676 Mamani09.
- M1: SVM nuevo vs antiguo: inferencia y perturbaciones identicas en las pruebas pareadas. RMSE por fuente (OOF): GEOROC 39.44 ppm frente a Mamani09 8.45 ppm; sesgo para Li alto: -15.30 ppm.
- M2: 706 muestras, 66 grupos, uno con 156 muestras (22.1% del conjunto); Random Forest nuevo vs CatBoost historico: robustez mixta, sin demostracion de mejor generalizacion externa.
- Los datos del release previo ya habian sido filtrados: que se reutilicen las mismas muestras es normal y esperado.
- Informes previos: results/scientific_simulations/post_training_20261106/ (datos locales, sin versionar los artefactos grandes).

## Etapa 1: protocolo, auditoria y reproducibilidad
1. Identificar SHA-256 de datasets, dos bundles recientes, release historico, semillas, versiones y separacion de grupos.
2. Congelar criterio de evaluacion; comparar variantes con los mismos grupos de evaluacion. Usar CV de desarrollo, no llamar 'independiente' a un holdout reutilizado para elegir configuraciones.
3. Auditar GEOROC por fuente, cobertura, Fe, sumas de oxidos, distribucion de Li y material. En particular:
   - Protocolo A (control): SUM (no water) entre 94 y 102.
   - Protocolo B (suma total reconstruible): solo si **los 10 oxidos** principales son validos, sin doble conteo del hierro; verificar 94..102. No rellenar suma con un numero ficticio.
   - Protocolo C (exploratorio): registros GEOROC WHOLE ROCK con Li+coordenadas validas, 8 de 10 oxidos, SiO2/Al2O3/CaO/Na2O/K2O presentes, al menos 2 trazas y suma parcial coherente; siempre marcar C_UNVERIFIED_EXPERIMENT_ONLY. No son analisis certificados.
4. Considerar FeO/FeOT del CSV original como posibles fuentes adicionales de hierro **solo tras revisar equivalencias, unidades y posibles duplicidades**. Actualmente los 10 oxidos requeridos incluyen Fe2O3 y el calculo estricto de B ya existe en la armonizacion.
5. La recuperacion del protocolo C nunca es automatica en la configuracion de produccion; exigir no deteriorar error espacial, caracterizar sesgo entre fuentes y documentar calidad quimica.

## Etapa 2: mejora real de M1
- Experimento de ablacion A/B/C sobre filas georreferenciadas del mismo dataset con un test comun de los grupos evaluados bajo A, entrenamiento espaciamente disjunto.
- Comparar SVM identity vs SVM log1p y modelos existentes regulares (RF o HGB) con pipelines de preprocesamiento entrenados solo en cada fold.
- Evaluar RMSE, MAE, sesgo, p90 de error, por fuente, por cuartil de Li y subestimacion de Li alto.
- Curvas de aprendizaje con fracciones de train y las mismas regiones de test; analizar si mas muestras realmente mejoran la generalizacion.
- Datos de 'C' hipoteticos de calidad no certificada; si mejoran un test de desarrollo, esto solo autoriza mas investigacion, no release automatico.

## Etapa 3: mejora real de M2
- Evaluacion espacial con GroupKFold y controles de bloqueo; no usar y/test para elegir folds (no fuga).
- Variantes de RF/CatBoost existentes: profundidad, hojas minimas, regularizacion y calibracion espacial si hay grupos y clases suficientes.
- Aumentacion experimental en training folds; separar la variante sobre caracteristicas de una verdadera simulacion fisica basada en pixels/SCL. Nunca alterar test/holdout.
- Etiqueta alto Li derivada del q75 **solo en train de cada fold**; comparar sobre los mismos test indices en todos los candidatos.
- Metricas ROC-AUC, average precision, balanced accuracy, F1, Brier, calibracion, falsos negativos y estabilidad espacial.
- Reportar resultados y limitaciones sin promover automaticamente un ganador.

## Etapa 4: EVALUAR FUENTES Y PAUSAR PARA DECISION DEL USUARIO
- Analizar necesidad mediante curvas de aprendizaje, fuente/region, concentracion extrema y cobertura.
- Fuente actual GEOROC: auditar si faltan registros reutilizables, SIN descargar un lote nuevo ahora.
- Fuentes candidatas externas: EarthChem/PetDB, actualizaciones GEOROC y SERNAGEOMIN (ultimo prioritariamente contraste); evaluar solapamientos, licencias, litologia, Li y georreferenciacion.
- Si una fuente es valida y realmente aporta informacion nueva, presentar propuesta para una herramienta del arbol tools reutilizando los patrones de navegacion, contrato, descarga, checkpoint, normalizacion y auditoria existentes. **No ejecutar esta integracion ni bajar datos hasta nueva decision**.

## Etapa 5: PENDIENTE (sin autorizacion de ejecucion)
- Con datos y protocolos finalmente aprobados, ejecutar Entrenar ambos y comparar con release en evaluacion independiente congelada.
- Criterios: rendimiento, problema especifico (GEOROC/Li alto, espacial/M2), robustez, evidencia y decision de candidato por separado.
- Ninguno de los modelos experimentales actuales se promueve automaticamente.

## Fuentes metodologicas verificadas
- GEOROC: https://georoc.mpch-mainz.gwdg.de/georoc/new-queryBlank.asp
- Geoquimica y calidad de sumas: https://essd.copernicus.org/articles/11/1553/2019/
- Datos composicionales incompletos: https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2026.1769579/full
- Transformacion de objetivo: https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
- CV grupos: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- Sentinel-2 SCL y radiometria: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html

Las alternativas son hipotesis a contrastar, no resultados ya demostrados.

## Hallazgo de factibilidad GEOROC (2026-11-06, sin cambiar datos originales)
- GEOROC original: 23.910 filas, 3.221 con Li numerico; la ingestion estandar retuvo 2.999 (2.986 tras union/deduplicacion).
- Solo 527 filas originales tenian diez oxidos completos cuando el hierro dependia de FE2O3; protocolo B ingenuo no recuperaba filas.
- FEOT (hierro TOTAL expresado como FeO) tiene 2.429 filas con Li. Una conversion documentada Fe2O3 total equivalente = FEOT * 1.111348 permite estudiar una reconstruccion completa sin inventar medidas.
- Auditoria preliminar de las filas crudas: 2.305 muestras adicionales con Li, material WR, nueve oxidos no ferrosos mas FEOT completo y suma reconstruida entre 94 y 102. Esta cifra esta **antes** de deduplicacion/armonizacion y no es recuento final aprobado.
- Prioridad de experimentacion: B con FeOT total, guardando provenance de hierro convertido, manteniendo test de cierre quimico, y luego C como candidato no verificado.
- Fuentes: https://geochem.nrcan.gc.ca/cdogs/content/sui/sui00557_e.htm y https://www.nature.com/articles/s41597-023-02324-7 .
- Ninguna conversion cambia de inmediato el dataset canonico ni el modo actual del menu.


## Etapa 2 M1: resultados de desarrollo (realizados)
- GEOROC FEOT se reexpreso como Fe2O3T equivalente en una COPIA EXPERIMENTAL de la fuente cruda, sin alterar el dataset original.
- 1.120 muestras A -> 3.420 B tras ingestion GEOROC existente, fusion y deduplicacion; C provisional 3.428.
- Comparacion sobre **exactamente los mismos registros A de test**, con grupos espaciales excluidos del train A/B/C.
- SVM A identity: RMSE por fold medio 20.7817, MAE 7.7399 ppm.
- SVM B FEOT identity: RMSE 19.3999, MAE 7.1796 ppm; mejora relativa 6.65% RMSE, 7.24% MAE.
- SVM B log1p: RMSE 19.9535, MAE 8.2445: no supera identity.
- SVM B mas regularizado: RMSE 20.2295, MAE 7.4172: no supera B identity.
- C provisional: RMSE 19.3910, MAE 7.1693: ganancia demasiado pequena para justificar las 8 muestras no verificadas.
- M1 GEOROC RMSE sobre mismo test: 39.57 -> 36.34 ppm; Mamani09 8.31 -> 8.19 ppm.
- Bootstrap por grupos espaciales para la GANANCIA en RMSE A-B: IC exploratorio 95% [-0.204, 3.202] ppm; NO excluye cero. Para la ganancia MAE: [0.047, 0.908] ppm. Ambos se basan en CV de desarrollo que eligio los candidatos, no en holdout independiente.
- Curva de aprendizaje de un fold: al agregar grupos desde 30% hasta 100% el MAE descendio, mientras el RMSE no fue monotonico. No justificar nuevas fuentes sin ampliar curvas multi-fold.
- El bundle candidato B se guarda **exclusivamente** en results/scientific_simulations/vnext_20261106/stage2_m1/experimental_models/. No instalar, activar ni publicar.
- Tablas auditables: m1_development_fold_metrics.csv, m1_paired_oof.csv, m1_paired_by_source.csv, m1_paired_by_lithium.csv, m1_development_diagnostics.json y m1_group_learning_curve.csv.
- Estado: etapa 2 desarrollo M1 realizado. Entrenamiento final pendiente de la parada y revision de etapa 4.
