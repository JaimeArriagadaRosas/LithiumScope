# M2 — Ampliación espectral y regularización sin descartar algoritmos

**Fecha:** 13/11/2026. **Estado:** diseño técnico + inventario de fuentes completado; **no** se ha descargado ni validado una nueva imagen, creado etiquetas nuevas, entrenado, alterado el menú ni activado un release.

## Restricciones permanentes

- **Se mantienen las seis familias**, sin recortar competencia: `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `catboost`, `svm_rbf`. Un algoritmo puede sorprender al cambiar distribución o características. Las familias no se eliminan por un ranking exploratorio.
- **Modelo 1**: predicción geoquímica de concentración y selección de candidatos. **Modelo 2**: reevaluación **principalmente espectral** de las **tres zonas candidatas emitidas por M1** y selección de una. Las concentraciones observadas **solo para supervisión**, no como entrada de M2; no recrear 33 variables químicas de M1. Comparar M1 solo vs M1→Top3→M2→Top1 usando candidatos generados realmente por M1, no únicamente ternas aleatorias de muestras históricas.
- Los 126 rasgos históricos permanecen como **bloque congelado**. La ampliación de características tiene nueva versión y caché; nunca cambiar en lugar la semántica de modelos activos. El protocolo de 706 y el de 783 quedan intactos para comparaciones pareadas.
- Descargas, caches, scripts y temporales en disco **D:**. Commits locales, **sin push**. Menú interactivo sin cambios.

## 1. Estado real comprobado y antecedentes que no se deben repetir

La imagen actual utiliza parches 640 m/32 px de Sentinel-2 L2A con **seis bandas**: B02 azul, B03 verde, B04 rojo, B08 NIR, B11 SWIR1, B12 SWIR2. El extractor v3 produce 126 números por parche: seis bandas × 8 estadísticas/textura + 13 índices × 6 estadísticas. Esas 126 características se conservan. Las bandas y los índices no son 126 sensores independientes.

Se documentó **overfitting**: RF candidato 12/11 entrenamiento aparente AUC 0,990865, OOF espacial AUC 0,644973 (comparaciones con ajuste y cortes de etiqueta distintos, pero señal inequívoca de brecha). En el estudio pareado con árboles regularizados, las ganancias ±3 % en DN sobre TRAIN redujeron AUC de 0,65747 a 0,64546, aunque bajaron Brier 0,21646→0,20980. KBest 48 redujo parcialmente brecha, **no demostró mejorar AUC**; embargo 5 km no aportó ganancia concluyente. No habilitar automáticamente «aumentación» ni cambiar todos los algoritmos por RF. Ya existen módulos opt-in de selección de escenas SCL, radiometría de escala/offset y ablation; **reutilizarlos**.

## 2. Fuente de muestras geoquímicas nuevas: inventario real

Entrada GEOROC: `results/scientific_simulations/vnext_20261106/georoc_source_validation/georoc_B_screened_sites.csv` (2.744 registros).
- 1.094 tienen estado **PROVISIONAL_SITE_REVIEW**. 1.040 satisfacen las reglas preliminares registradas (DOI/referencia completa, Li ≤100,6 ppm, coordenadas válidas).
- De 100 seleccionadas previamente en grupos distintos, **77** se incorporaron a la ampliación M2 y **23** quedaron pendientes/fallaron descarga.
- Hay **940 filas todavía no en la cola**, pero **solo 22 grupos espaciales de 0,5° que no están ya representados en el conjunto de 783** y que tampoco pertenecen a las 100 seleccionadas. Son candidatos, **no muestras aceptadas**; deben someterse a revisión de procedencia roca–ubicación–espectro y QA de parches.
- Se requieren fuentes NUEVAS y grupos nuevos para crecer substancialmente sin pseudorreplicación; consultar GEOROC por publicaciones/lugares no presentes y, como contraste de procedencia, EarthChem (deduplicando contra GEOROC). Priorizar **20°S–30°S** donde RF actual tiene AUC 0,4669 sobre 85 históricos, sin excluir las regiones restantes.
- Selección de cada nueva etiqueta: `sample_id`, coordenadas y precisión reportada, `source_doi`, material superficial reconocido (no salmuera/subsuelo), unidades Li ppm, publicación, sitio/grupo espacial, escena y fecha, calidad por banda y SCL, huella SHA y comprobación de repetidos. Etiquetas de rocas medidas en laboratorio, **no inferidas** desde la imagen, M1 u otros modelos. Provisionales se reportan como provisionales.
- Primer objetivo **no vinculante**: recuperar los 23 pendientes, examinar los 22 grupos geográficos nuevos y luego abrir una consulta independiente para cubrir regiones poco representadas. El número de nuevas filas aceptadas se determina por QA real, no se inventa un objetivo cerrado de +700.

Inventario ya generado sin descarga:
`results/scientific_simulations/m2_spectral_expansion_20261113/georoc_candidate_inventory.csv`
y `candidate_inventory_summary.json`. Se reproduce con:
`PYTHONPATH=src D:\\Librerias\\Python312\\python.exe scripts/scientific/m2_spectral_expansion_preflight.py` (usar sintaxis CMD/PowerShell apropiada).

## 3. Nuevas observaciones Sentinel-2: exactamente qué añadir

**Banda única por sitio:** de la misma escena STAC L2A, incorporar **B05 rededge1 (705 nm), B06 rededge2 (740 nm), B07 rededge3 (783 nm), B8A nir08 (865 nm)**. Son cuatro bandas de 20 m en el catálogo Earth Search y no están en nuestro extractor v3. Mantener B02/B03/B04/B08/B11/B12 actuales.

**Características v4 (opción versionada):** 126 originales como referencias intocables + 4 bandas × sus ocho estadísticas existentes = **158** + dos índices red-edge predeclarados × seis estadísticas = **170** en total. Probar ambos brazos `base_126` y `base_plus_rededge_170`, sin seleccionar índices después de mirar TEST. Los índices red-edge describen superficie/vegetación, **no son sensores directos de litio**. Si se aplica reflectancia, recalcular índices dentro del brazo coherente de reflectancia; nunca mezclar índices calculados de DN con reflectancia.

**Segundo momento por sitio (opcional, separado del cambio de bandas):** otra escena real del MISMO lugar en ventana estacional distinta con cobertura SCL útil. Añadir máximo 12 estadísticas robustas de variación temporal en bandas/índices ya definidos. No duplicar el label como 2 observaciones «independientes» y no incluir ninguna escena proveniente de TEST en el aprendizaje de TRAIN para un fold. Registrar fechas, radiometría y fracciones válidas. Brazo `base_plus_rededge_temporal` solo después de probar `rededge_170`.

**Radiometría:** inspeccionar `raster:bands.scale/offset` por asset/escena; aplicar exactamente esa transformación a DN, preservar `nodata`, nunca usar fill ficticio como reflectancia ni cambiar el extractor v3 en lugar. El inventario recuperó 111 metadatos STAC, 108 offsets -0,1 y 3 offsets 0. El contraste previo sobre 706 mostró **resultados mixtos** al corregir BOA: no se considerará mejora automática.

**Máscara QA, no etiqueta:** SCL para excluir nubes, sombras, agua/nieve cuando afecten roca superficial, cubrimiento conjunto ≥95 % y clase 5 no vegetado como **proxy**, no como prueba de litología. Registrar `QA abstention` en lugar de inventar imagen o textura válida. Guardar originales y variantes con SHA y path diferenciados.

**Geología auxiliar limitada:** un mapa general de unidades/litología **superficial** por región desde SERNAGEOMIN donde exista cobertura y calidad de georreferenciación; opcionalmente solo **una categoría litológica gruesa** y un campo de disponibilidad. No incorporar las 33 características químicas de M1, ni exigir datos geológicos ausentes, ni usar información mineral o etiquetas de Li derivadas retrospectivamente.

Fuentes comprobables: ESA Sentinel-2 MSI banda/resolución, Element84 Earth Search v1 STAC, metadatos BOA de Copernicus, GEOROC y EarthChem; SERNAGEOMIN para unidades cartográficas chilenas.

## 4. Regularizar sin quitar ninguna de las seis familias

| Algoritmo obligatorio | Búsqueda de estabilización dentro de TRAIN | Qué NO hacer |
|---|---|---|
| Random Forest | límites a `max_depth`, `min_samples_leaf`, `max_features`, peso por sitio | no usar profundidad ilimitada + memorizar parches |
| Extra Trees | hoja y profundidad mínima/máxima, promediar aleatorización, pesos por sitio | no reemplazarlo por Random Forest |
| HistGradientBoosting | `min_samples_leaf`, `l2_regularization`, `max_leaf_nodes`, `early_stopping` interno | no usar etiquetas de outer TEST para parar |
| XGBoost | `max_depth`, `min_child_weight`, `reg_alpha/reg_lambda`, `subsample/colsample_bytree`, early-stop interno | no seleccionar hiperparámetros por un TEST ya visto |
| CatBoost | `depth`, `l2_leaf_reg`, `random_strength`, early-stop interno, contracción de aprendizaje | no desactivar CatBoost por su ranking histórico |
| SVM-RBF | `C`, `gamma`, escalado dentro del fold y ponderación por grupos | no asumir que más variables siempre favorecen al SVM |

**Aumento espectral:** no repetir ganancia DN que empeoró AUC; solo probar perturbación físicamente justificada **en reflectancia calibrada**, pequeña y dependiente de incertidumbre/escena, sin modificar máscaras ni etiquetas, dentro de TRAIN. Mantener peso TOTAL de cada sitio constante (original+copia) para evitar pseudorreplicación. **No producir imágenes espectrales inventadas con IA** ni suponer que se comportan como un sensor. Una imagen adicional real de otra fecha no es otra medición independiente de litio.

**Dependencia espacial:** plegar por sitio/publicación y grupo; al evaluar, buffers 5/10/20 km que purgan vecinos TRAIN cercanos a TEST; evaluar deltas con muestras test fijas cuando sea posible. Reducir peso de repetidos/celdas muy densas según protocolo train-only.

## 5. Experimentos controlados y condición de éxito

Sin retirar algoritmos, mismo protocolo de cinco outer folds + tres inner y comparable TEST:
- E0: 126 actuales + seis algoritmos con regularización vigente; baseline congelado.
- E1: las mismas 126, parches recalculados con radiometría física y máscara válida, seis algoritmos; evaluar primero si aporta señal real.
- E2: 170 con red-edge en las mismas observaciones, seis algoritmos. Aislar efecto de **nuevas bandas**.
- E3: 170 + hasta 12 de estabilidad estacional, mismas etiquetas y grupos, seis algoritmos. Aislar efecto **multitemporal**.
- E4: comparar 706, 783 y nuevas muestras **SOLO de TRAIN**, con TEST de sitios geológicamente independientes congelado. Aislar efecto **nuevas muestras**.
- E5: comprobar en ejecución **M1→Top3→M2→Top1** si cada uno de los seis modelos mejora elegir el sitio con Li observado mayor frente a **M1 solo**, con candidatos generados por M1 sin usar los datos de TEST para seleccionarlos.

Métricas: ROC-AUC/AP espaciales, Brier, VP/VN/FP/FN a umbral preseleccionado **solo en TRAIN**, incertidumbre por grupo, brecha TRAIN-TEST, estabilidad 20°S–30°S, precisión Top1 de **tres candidatos M1 reales** vs M1 solo, cobertura y abstención. Todos los cambios de imagen/versiones deben mantener hash y registros de procedencia. No entrenar ni promover durante el mero inventario.

**Criterio científico:** no prometer un aumento de AUC de 0,05 sin datos. La meta es una mejora **consistente y útil** con control de falsos negativos y dispersión espacial, no ganar una milésima seleccionando el mejor de múltiples experimentos ya vistos. Un holdout externo nuevo es imprescindible para una afirmación confirmatoria de superioridad.

**Estado hoy:** únicamente el inventario de fuentes está implementado y ejecutado; la descarga red-edge, variantes v4, augmentación física y competición E1–E5 son etapas de implementación **pendientes**.
