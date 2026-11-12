# LithiumScope — Implementación del plan metodológico satelital

**Fecha:** 12/11/2026. **Estado:** código, auditorías y contrastes internos ejecutados. Ningún cambio en modelos activos, menú, etiquetas, imágenes ni folds originales.

## Diagnóstico y criterio científico

Los 16 registros de Sentinel-2 con datos incompletos son 10 imágenes originales distintas; las pruebas de robustez que eliminan píxeles artificialmente se ejecutan por separado. El código anterior seleccionaba el menor porcentaje de nubosidad global, aun cuando el parche de 640 m tuviera cobertura insuficiente o coincidiera con bordes de escena.

La calidad local se debe juzgar conjuntamente por seis bandas, valores sin datos y la clasificación SCL. Una escena de datos completos puede tener nieve, agua o sombras y carecer de señal útil para la prospección de roca superficial. La clase 5 significa no vegetado y no garantiza litología ni validación de terreno.

## 1. Inventario del entrenamiento original

| Indicador | Valor |
|---|---:|
| Muestras / parches de entrada | 783 |
| Imágenes únicas por SHA-256 | 593 |
| Escenas Sentinel-2 recuperadas | 111 / 111 |
| Escenas con metadatos radiométricos completos | 111 / 111 |
| Escenas STAC con offset -0,1 | 108 |
| Escenas STAC con offset 0 | 3 |
| Muestras con cobertura conjunta inferior al 95% | 16 |
| Imágenes únicas incompletas | 10 |
| SCL local de imágenes únicas comprobado | 593 / 593 |
| SCL local pendiente | 0 |
| Imágenes únicas con proxy óptico aceptado | 458 / 593 |
| Muestras (incluidos sitios repetidos) con proxy óptico aceptado | 613 / 783 |
| Muestras con abstención de proxy óptico | 170 / 783 |
| SCL no disponible | 0 |

Las máscaras espectrales locales se calcularon sobre los NPY originales, sin reescribirlos. Los metadatos STAC se almacenaron en caché independiente, con identificador de escena, nubosidad, no-datos globales, fecha, baseline y parámetros radiométricos por banda. La lectura SCL está separada del manifiesto geoquímico y tiene checkpoints recuperables.

## 2. Selector de escenas: calidad local, fecha predefinida y abstención

La API nueva search_validated_scene permite elegir una escena de manera explícita con ventana de fechas obligatoria y limitación de candidatos, dejando search_best_scene histórico sin cambios. Examina seis bandas, SCL y procedencia radiométrica; acepta parches únicamente con cobertura espectral >=95%, bloqueos SCL <=25% y clase 5 no vegetado >=50% como proxy conservador. El selector nunca certifica roca ni valida litio.

El parámetro selected_scene de download_patch permite adquirir una nueva muestra desde la escena validada sin volver a seleccionar por nubosidad; si el archivo ya existe, se respeta la caché original y no se sobrescribe.

| Imagen única | Cobertura antigua | Candidata aceptada de la misma fecha | Decisión |
|---|---:|---|---|
| Ubi_48b | 68.8% | S2B_19KBB_20261030_0_L2A | SAME_DATE_PROXY_ACCEPTED |
| UBI-99-01 | 50.0% | S2B_19KBB_20261030_0_L2A | SAME_DATE_PROXY_ACCEPTED |
| PAR_039 | 84.4% | ninguna | NO_ACCEPTED_SAME_DAY_SCENE |
| PAR_03-38 | 84.4% | ninguna | NO_ACCEPTED_SAME_DAY_SCENE |
| PAR_028 | 84.4% | ninguna | NO_ACCEPTED_SAME_DAY_SCENE |
| CRD_94_234 | 71.9% | ninguna | NO_ACCEPTED_SAME_DAY_SCENE |
| PER_95-276 | 92.1% | ninguna | NO_ACCEPTED_SAME_DAY_SCENE |
| YAH-00-14 | 59.4% | S2B_18LXH_20261013_0_L2A | SAME_DATE_PROXY_ACCEPTED |
| PIG-03-132 | 62.5% | S2C_18KZG_20261015_0_L2A | SAME_DATE_PROXY_ACCEPTED |
| Ilo_I-04-03 | 65.2% | ninguna | NO_ACCEPTED_SAME_DAY_SCENE |

Resultado: **4 de 10** imágenes únicas tienen una candidata satelital de la misma fecha que supera los filtros técnicos; **6 de 10** quedan en abstención. Todas las decisiones quedan auditadas con motivos y fracciones SCL. No se reemplazó ninguna imagen.

## 3. Radiometría y tratamiento de datos incompletos

Se implementó conversión por escena y banda reflectancia = DN × scale + offset, leyendo obligatoriamente raster:bands de Earth Search. No se aplica offset -0,1 a escenas antiguas con offset 0. Los píxeles no-datos siguen siendo NaN mientras se calculan índices espectrales, para no confundir un denominador inválido con un índice físico cero. Se han generado tres cachés de 783 × 126 características sin modificar la caché histórica.

## 4. Experimento pareado interno sobre los 706 test históricos

Se reutilizó la predicción CatBoost de referencia y se ajustaron 15 modelos experimentales (tres variantes × cinco folds) con los parámetros congelados anteriores. Las 77 etiquetas GEOROC provisionales no se usaron en este entrenamiento, y no se utilizaron los diez casos conocidos para elegir configuraciones.

| Representación | ROC-AUC | AP | Brier (menor mejor) | Recall 0,5 |
|---|---:|---:|---:|---:|
| Referencia histórica DN con mediana | 0,6207 | 0,3827 | 0,1841 | 0,1695 |
| DN sin imputación | 0,6321 | 0,3685 | 0,1857 | 0,1751 |
| BOA con escala y offset, solo píxeles válidos | 0,6022 | 0,3274 | 0,1968 | 0,1695 |
| BOA con escala y offset, mediana histórica | 0,6406 | 0,3678 | 0,1856 | 0,1525 |

Los intervalos exploratorios se calcularon mediante 320 remuestreos por los 66 grupos históricos. Las diferencias ROC-AUC respecto de referencia incluyen cero en todos los casos.

| Variante respecto a referencia | ΔAUC | IC95% exploratorio | ΔAP | IC95% exploratorio AP | ΔBrier | IC95% exploratorio Brier |
|---|---:|---|---:|---|---:|---|
| DN sin imputación | 0,0114 | [-0,0187, 0,0586] | -0,0142 | [-0,0492, 0,0403] | 0,0016 | [-0,0088, 0,0085] |
| BOA con escala y offset, solo píxeles válidos | -0,0185 | [-0,0501, 0,0284] | -0,0553 | [-0,0911, -0,0092] | 0,0127 | [0,0031, 0,0226] |
| BOA con escala y offset, mediana histórica | 0,0199 | [-0,0136, 0,0624] | -0,0149 | [-0,0481, 0,0400] | 0,0015 | [-0,0073, 0,0078] |

**Interpretación:** la variante de reflectancia físicamente corregida sin imputación tuvo menor AP y peor Brier. La corrección combinada con mediana elevó AUC, pero no mejoró simultáneamente AP o calibración. **No está demostrado que ninguna configuración generalice mejor**; la corrección radiométrica es científicamente pertinente pero exige reentrenamiento y validación específica antes de desplegarse.

## 5. Elegibilidad de muestras y validación independiente

Persisten 25 sitios históricos con etiquetas Li alto/bajo incompatibles sobre idénticos parches y 77 muestras GEOROC provisionales. La documentación original Mamani09 permitió recuperar las fuentes, pero ni el DOI ni la clase SCL de suelo no vegetado certifican la correspondencia muestra–afloramiento–imagen.

**Gate de campo: NO APROBADO.** Las revisiones manuales documentadas anteriormente no fueron sustituidas por reglas automáticas. No hubo cambio de etiquetas, exclusiones retroactivas del test ni selección de políticas según los diez casos.

Se mantiene como pendiente científico un conjunto externo nuevo, con muestras geoquímicas verificadas y bloqueo espacial/temporal desde el inicio. Los 706 históricos ya se han utilizado para desarrollo y sus intervalos no son confirmación independiente.

## Ejecución y entregables técnicos

El paquete científico opt-in contiene satellite_quality.py, satellite_inventory.py, scl_inventory.py, radiometric_features.py y radiometric_comparison.py. Su integración compatible está en data/sentinel2.py. Las reglas de selección de escenas no se activaron en el entrenamiento actual.

Comandos reproducibles (ejecutar desde la raíz del repositorio):

    python -m lithiumscope.model_2.methodology.satellite_inventory --output results/scientific_simulations/sentinel_methodology_20261112/inventory --online-metadata --alternatives
    python -m lithiumscope.model_2.methodology.scl_inventory --output results/scientific_simulations/sentinel_methodology_20261112/local_scl
    python -m lithiumscope.model_2.methodology.radiometric_comparison --output results/scientific_simulations/sentinel_methodology_20261112/experiments

Carpeta de evidencias: results/scientific_simulations/sentinel_methodology_20261112. Incluye manifest original auditado, 111 respuestas STAC cacheadas, decisiones mismas fechas, tres tablas espectrales, estimaciones OOF pareadas, incertidumbre y logs.

## Bibliografía técnica

- ESA, Sentinel-2 processing, códigos SCL 0–11: https://sentiwiki.copernicus.eu/web/s2-processing
- Documento de especificación Sentinel-2: https://sentinels.copernicus.eu/documents/d/sentinel/s2-pdgs-cs-di-psd-v15-0
- Copernicus Sentinel-2 L2A, harmonización DN y PB 04.00: https://documentation.v1.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
- Sentinel-2 offset y escala física: https://ycgs-lab.github.io/data-guide/data-products/Sentinel-2.html

No se efectuó reemplazo de escenas, ni promoción de modelos, ni modificaciones del menú, ni push.
