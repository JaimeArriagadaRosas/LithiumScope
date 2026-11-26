# Fuentes científicas de M1: catálogo, uso, confiabilidad y decisión de cierre

Fecha de revisión: 26 de noviembre de 2026. LithiumScope, rama feat/m1-targeted-data-refinement.
Alcance: fuentes históricas, nuevas adquisiciones y alternativas expresamente registradas
en el laboratorio de datos M1; referencias metodológicas de esta línea. No es una
enumeración de los miles de artículos de las bases federadas ni de la bibliografía M2.
Los inventarios de citas por fila se conservan en los artefactos originales de auditoría.
No publicar archivos originales de terceros sin revisar licencias. M2 congelado.

## 1. Criterio de confiabilidad

Distinguir autoridad del proveedor, identidad del archivo (SHA-256), calidad analítica
(método, unidades, censura, control instrumental), identidad de muestra (ID,
publicación, localización, material), compatibilidad con predictores y población,
independencia y licencia. Un hash prueba integridad, no veracidad geológica.
Una publicación revisada no certifica todas sus filas para la inferencia de M1.
Una fuente derivada de GEOROC no es una réplica independiente por cambiar de editor.
Ningún nuevo registro dirigido está aprobado para producción. Las cifras de esta
sección provienen de ejecuciones locales y evidencias verificadas; se distingue el
caso whole-rock, cuyas salidas fueron verificadas localmente por preflight, no
recalculadas en chat directamente. El catálogo permite recuperar fuentes sin repetir
la búsqueda; las fechas de descarga/versiones se conservan en manifests.

## 2. Fuentes históricas y compilaciones registradas

### Mamani, Wörner y Sempere (2010; publicación en línea 2009)
- [Artículo original](https://doi.org/10.1130/B26538.1), GSA Bulletin 122, 162–182.
- [Compilación GFZ/DIGIS](https://doi.org/10.5880/digis.e.2024.005),
  [registro institucional](https://dataservices.gfz-potsdam.de/digis/showshort.php?id=6b16c88c-c124-11ee-967a-4ffbfe06208e).
  La compilación integra datos de literatura hasta 2009 y nuevas mediciones del estudio.
- Uso: etiqueta histórica Mamani09; en cohorte académica 674/3263 filas; mediciones
  adicionales ICP-MS/XRF separadas por método. Las 38 columnas adicionales del
  inventario inicial solo tenían cobertura Mamani09.
- Confiabilidad: fuente regional directamente pertinente; no asumir que el archivo
  GFZ sea idéntico a la copia histórica ni independiente de ella. Se investigó la
  compilación como alternativa, no se adquirió un nuevo lote de esa publicación.
- Decisión: preservar base congelada y procedencia; evitar doble conteo con GEOROC.

### GEOROC (Göttingen/DIGIS; compilaciones versionadas)
- [Portal](https://georoc.eu/), [reglas de compilación](https://georoc.eu/georoc/CompRules.html),
  [aviso de calidad](https://georoc.eu/georoc/disclaimer.asp),
  [citación y CC BY-SA 4.0](https://georoc.eu/georoc/cite.asp).
  Citar compilación/versiones y autores originales. La prioridad de método de los
  precompilados no garantiza que todo Li(PPM) sea ICP-MS. Heterogeneidad en método,
  precisión, localización y antigüedad, explícita en aviso oficial.
- Base histórica GEOROC: 2589/3263 académicas; junto a Mamani09, 3263 académicas
  y 3420 preparadas espaciales (244 grupos), de 4765 brutas. Fuente combinada
  SHA256 94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94.
  No equivale a una descarga única de toda la base.
- [Convergent Margins DOI](https://doi.org/10.25625/PVFZCE):
  registro georoc_andean_202609, tres CSV 2026-09-PVFZCE_ANDEAN_ARC_part1/2/3.csv,
  ids 219128/219105/219094. 2998 filas armonizadas candidatas, 2990 claves
  coincidentes con histórico bruto, 2750 con preparado. Recuperación de trazas
  omitidas: 2686 vínculos aceptados, 2532 académicos, 2531 enriquecidos,
  58708 valores adicionales no faltantes académicos. Vínculos retrospectivos
  asistidos por Li: útiles para desarrollo, no identidad certificada al inferir.
  Usar georoc_ELEMENTO_ppm y mantener métodos originales separados.
- [Rock Types DOI](https://doi.org/10.25625/2JETOA):
  georoc_felsic_rocks, GRANITE y RHYOLITE, seis particiones dentro del total de
  nueve verificadas, 8234 candidatas. 221 claves históricas exactas, 205 preparadas.
  Candidatas intraplaca altas requieren transferencia experimental; 69 en estado
  de revisión sin soporte vecinal inicial. No asumir equivalente al arco andino.
- Inventario bibliográfico/metadata: DOI https://doi.org/10.25625/4EZ7ID registrado
  previamente; inventarios locales resuelven 5334 IDs, 6474 entradas de pie.
  La expansión total 11232 candidatas, 6781 elegibles exploratorias, 4611 pool,
  1483 reserva de desarrollo ya explorada, 687 cuarentena, 221 copias exactas
  (conteos no sumar si solapan). Métodos Li y CRS no explícitos por fila;
  referencias históricas resolubles en 676 filas, no independencia demostrada.
- Uso efectivo: experimentos espaciales y académicos de transferencia, no
  incorporación productiva. Transferencia espacial CatBoost atenuada mejoró
  algunas métricas pero mantuvo degradación en folds/intervalos. En protocolo
  académico la expansión sin ponderar empeoró RF, CatBoost y HGB.
- Evidencia: results/model_1/expansion/20261126T145129_ee7f5431, manifest
  0817aac5e5d869e9be0ec9356c8eea189278cf8ec1f896d19e025384affac6b2;
  provenance_followup manifest 0bb72d0273c5c2d5c25c7b8b2fdeb406b6e08d4bd2bb724b0ae453aa762abd7b.
  Recuperación results/generalization/model_1_expansion_linked_recovery/20261126T163242_d5f5c5fe,
  manifest f8469364dc93133a4fe23edcd385844a426f4d19e94ccf950d6e58365ccd7ce8.
  Inventarios de archivos, citas y hashes completos en esas carpetas;
  registro de descarga en datasets/management/research_sources.py.
- Decisión: conservar histórico enriquecido; no repetir ingestión, búsqueda ni
  expansión masiva ya evaluada.

## 3. Adquisiciones dirigidas con originales examinados

Carpeta relativa: data/raw/model_1/research/directed/<clave>.

### Meixner et al. (2020; en línea 2019)
[Lithium concentrations and isotope signatures...](https://doi.org/10.1007/s00126-019-00915-2),
Mineralium Deposita; [contexto CONICET](https://www.conicet.gov.ar/el-origen-del-litio/).
Fuente pertinente para origen del Li en Andes centrales.
- Clave meixner_2020. ESM1 DOCX: metodología/referencias; ESM2 XLS A1: muestras
  naturales, isótopos y concentraciones; ESM3 XLS A2: diez estándares y un blanco;
  ESM4 XLS A3: 14 ensayos de lixiviación (ng, no ppm). Cuatro originales
  inspeccionados y verificados por hash; no confundir masa/isótopo/estándar con
  objetivo de regresión. A1: 89 análisis, 7 repeticiones, 82 principales, 80
  Li positivo 5–159 ppm (mediana 24); 19 >40, 18 dentro de la cola académica.
  52 volcánicas, 16 sedimentarias, 12 metamórficas con Li. Solo SiO2 de los
  22 predictores químicos canónicos, 60 con coordenadas y SiO2.
- Uso: búsqueda de identidad no asistida por Li en 4765 filas y nueve
  particiones, cinco coincidencias de nombre, cero sitio+sílice. Tres
  históricas sin Li (32,19,16 ppm) no tienen identidad demostrada.
- Referencias auxiliares mencionadas en A1: Risse 2013; Lindsay 2001; Grocke
  2017; Rosner 2003; Egenhoff/Lucassen 2003; Kasemann 2000. No adquiridas como
  datasets separados ni afirmar que cada valor individual proceda de ellas.
- Calidad científica pertinente; baja suficiencia de predictores para M1.
  No aprobada para expansión directa. Licencia de suplementos por revisar.
- Evidencia model_1_meixner_identity_audit/20261126T202913_8686ac5d, manifiesto
  39b6218e257284b19061eb1ac9c2b65a65a29e555614f932a57d79c1d6fbd193, 4/4 salidas.
  Originales SHA256:
  - 126_2019_915_MOESM1_ESM.docx: 2070c8e531c40ef2816621e3d786e626c43a4c622d69344df50d12ac4a00272e
  - 126_2019_915_MOESM2_ESM.xls: ffb7171170cfb383bc47ccc4f86f6aa7f1197f08c9a4bcdfed9e65a14efffbe1
  - 126_2019_915_MOESM3_ESM.xls: 7e172ef76fb1387aa6a264ed9bce62b9a4f5e8c25a5f725c192e3b6bb195767d
  - 126_2019_915_MOESM4_ESM.xls: 155c7c5a44c090a0a99726c49cc247e60df551ad907296507a861e31ae557286

### Fernández-Turiel et al. (2020), shoshonitas Puna
[DOI DIGITAL.CSIC](https://doi.org/10.20350/DIGITALCSIC/10527),
[registro](https://digital.csic.es/handle/10261/197644),
[EPOS-MSL](https://epos-msl.uu.nl/data-publication/0aa9848d06477eaee50b61f3aa829b64).
Roca total, vidrio y minerales, San Jerónimo y Negro de Chorrillos; CC BY 4.0
según catálogo previamente revisado.
- Clave fernandez_turiel_2020. Original recibido whole_rock.xlsx (no presumir
  incluye el archivo combinado anunciado por EPOS), y 0_metadata.xlsx.
  Seis IDs/IGSN concordantes, 22/22 química, seis coordenadas/QC, 19.96–29.64
  mg/kg = ppm, método HR-ICP-MS (Element XR). Suma de óxidos 99.17–100.48%.
  No se usaron los siete ficheros minerales/vidrios como roca total.
- Uso: inventario, unión de muestra/método, deduplicación y soporte TRAIN.
  6 candidatas iniciales, 0 compatibles en los cinco folds por distancia
  multivariada, ninguna >40. Alta trazabilidad analítica de este lote, pero
  no representativo para ampliar la distribución objetivo. No usar en cierre.
- SHA256 metadatos 4aa129fa5930313d892ceee79f98f59773a3ede672abcd8d45bf2c2d9f5afa56;
  roca total 4338150d9b3822a4e1cf7ef1f91283e98b4ec44fac3caf5ade9da8df0b831e45.

### Abat et al. (2025), Macusani, SE Perú
[Dataset IEDA](https://doi.org/10.60520/IEDA/113778),
[EarthChem 3778](https://ecl.earthchem.org/view.php?id=3778), CC BY 4.0,
[Ramirez-Briones et al.](https://doi.org/10.1093/petrology/egaf041).
La ficha confirma XRF mayores, ICP-MS trazas y distintos métodos isotópicos.
- Clave macusani_2025; 3778-1_EC_BulkSample_Analyses_TorroetalMACUSANI.xlsx,
  SHA256 ee8a26efb260015035930bc5199e4f94fd078b8dabccfd839aaeec9305ba1666.
  29 registros, 22 Li numérico positivo 310–4160 µg/g = ppm, ocho con Li y
  suma completa de óxidos 94–102, ninguna dentro del rango académico.
  20 posibles duplicados; 29 bloqueados por publicación [30009] ya presente
  en GEOROC. Método código 3 de Li, no atribuir certeza fila por fila solo
  por ficha. Incluye mena sedimentaria/roca volcánica e intrusiva.
- Uso: verificación, armonización diagnóstica y auditoría. Alta autoridad
  y datos primarios identificables; no independientes ni adecuados al rango
  académico. Cuarentena, no usar en cierre.
- Evidencia común Puna/Macusani:
  results/generalization/model_1_directed_whole_rock_audit/20261126T205508_dccd2ea0,
  manifiesto 61c1e6c083019ff21ab16cbb54961c864c025d4e571661dd52147f9f933d97ce,
  seis tablas validadas localmente en preflight.

### Chen, Lee y Tang (2020), arcos volcánicos globales
[Artículo Nature Communications](https://doi.org/10.1038/s41467-020-19106-z),
[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7575555/).
Los autores identifican el Supplementary Dataset 3 como extracción GEOROC
de roca total de arco. No es fuente independiente de GEOROC.
- Clave chen_global_arcs_2020. Archivo 41467_2020_19106_MOESM6_ESM.xlsx,
  10950278 bytes, SHA256 a5cfd0c5453ea26ed9f2e436e2de87678fb0a34e91b148d841dfda5f5db0e69d;
  [URL editorial que funcionó](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-19106-z/MediaObjects/41467_2020_19106_MOESM6_ESM.xlsx).
  Primer intento PMC rechazado como HTML. 50145 filas de hoja, 66 columnas,
  5201 referencias aparentes; 3680 Li positivo 0.3–597.1 ppm, mediana 9.795.
  No son 50145 muestras etiquetadas ni 3680 identidades únicas certificadas.
- Tratamiento: Li(PPM), FeOT a Fe2O3T equivalente por 1.111348 conservando
  base original, diagnóstico 94–102 ambas bases; 3676 QC, 3461 rango
  académico, 95 cola inicial. 22 químicos canónicos mapeables, trazas
  adicionales ppm. Métodos desconocidos, no inventar litologías, edad, Arc,
  Domain o CRS; no usar espesor cortical/Moho/Li como predictores nuevos.
- Deduplicación: 826 posibles duplicados, 1122 solapamientos de referencias
  con expansión, 2303 candidatos iniciales, 45 cola. Sin nombres de muestra,
  intervalos geográficos y bibliografía; 0 coincidencias históricas por
  tokens exactos no demuestra ausencia. Cita [17558] en cuarentena, 115
  límites invertidos, 252 amplios; filtros pueden solaparse.
- Preflight: 1974 estáticos, 1588 distintos en al menos un fold, 1469 en
  los cinco, 226 grupos y 237 etiquetas de cita; 4 únicos >40 ppm
  (41.4–46.19), sin >60. Conteos por fold 1505/1560/1529/1559/1539,
  cola 2/4/3/3/3. Química y bloqueo de evaluación estimados por TRAIN.
  Uso permitido únicamente como sensibilidad de desarrollo explícita,
  sin certificar material/método/CRS, identidad ni independencia.
- Evidencias: model_1_directed_source_intake/20261126T205949_acebdd9e,
  model_1_chen_profile/20261126T210320_3f80751b,
  model_1_chen_overlap_audit/20261126T210740_ed33aa31 (6/6 hashes,
  manifiesto b0bfff9a1ee3e7660a5e7388acd2b3ce1886ca6acc1fe6fc7895f057eca44925),
  model_1_directed_transfer_preflight/20261126T211301_3a7cd8be
  (4/4 hashes, manifiesto 21b112c5faaee0c410a5045ab295b57a41c875eb6dda78f268c8b4c0cfb0d99b).
  Todos bajo results/generalization. Licencia del suplemento por comprobar
  antes de redistribución; no subir binarios al repo.

## 4. Fuentes investigadas, no usadas como nuevas filas

- [Macdonald, Smith y Thomas 1992, USGS PP1523](https://doi.org/10.3133/pp1523),
  [página con workbook](https://pubs.usgs.gov/pp/1523/): obsidianas silícicas
  subalcalinas no hidratadas. La página indexa un Excel derivado del libro,
  pero no se descargó/inspeccionó; acceso directo devolvió 403 en esta revisión.
  Cita [17558] contiene 70 candidatos Chen, 32 de 45 cola preliminar y todos
  seis andinos; concentración de fuente y ausencia de identidad/método
  certificado justifican cuarentena, no rechazo universal de obsidiana.
  No usar para el cierre; no abrir otra ronda de descarga.
- [Álvarez-Amado et al. 2022](https://doi.org/10.1016/j.gexplo.2022.107062),
  [ficha Universidad Mayor](https://researchers.umayor.cl/es/publications/the-role-of-ignimbrites-and-fine-sediments-in-the-lithium-distrib/):
  ignimbritas, sedimentos y agua; resumen indica 20–50 ppm en roca
  volcánica, media 33, sedimentos media ~290. No mezclar matrices ni
  asumir cobertura de 22 predictores; original/suplementos no adquiridos.
  Interesante para investigación futura, no indispensable para cierre.
- [Del Bono et al. 2026](https://doi.org/10.1007/s00126-026-01428-5),
  [CONICET](https://ri.conicet.gov.ar/items/deee6775-ca99-430f-add1-1c01b1dfbdd6):
  interacción agua-roca, ceniza y génesis de salmueras Puna. Revisión previa
  detectó suplementos de descripción/localización y figuras de lixiviación;
  tabla multielemental Li de roca total no confirmada. No descargados ni usados.
- [EarthChem/PetDB](https://earthchem.org/petdb), [alcance federado](https://earthchem.org/about/about-overview):
  canal de consulta geocientífica con bases que pueden compartir datos con
  GEOROC. No se exportó un dataset PetDB nuevo; no contar como fuente
  independiente. EarthChem Library Macusani sí fue adquirida (sección 3).
- Compilación Wörner/Mamani GFZ descrita en sección 2: no adquirir otra copia
  para contabilizarla como nueva.
- Memoria OASIS del grupo de compañeros: documento aportado por usuario,
  tabla 15 p.112 según revisión previa registrada; 681 observaciones, RF
  RMSE medio 5.361 ppm y R² medio .499. Referencia de rendimiento descriptiva,
  no datos de entrenamiento, no fuente pública comprobada en esta revisión.
  Población y protocolo distintos. No prometer superación científica directa.

## 5. Literatura metodológica y evaluación crítica del enfoque

Referencias comprobadas en web en esta revisión:
- Branco, Torgo, Ribeiro (2017), [SMOGN](https://proceedings.mlr.press/v74/branco17a.html):
  relevancia del objetivo continuo, sobremuestreo de zonas raras e interpolación
  o ruido. Fundamenta evaluar rareza y soporte, no fabricar valores extremos.
  Nuestras variantes acotadas SMOTER/gaussiano no son SMOGN original.
- Branco, Torgo, Ribeiro (2019), [Pre-processing approaches for imbalanced
  distributions in regression](https://www.sciencedirect.com/science/article/pii/S0925231219301638):
  comparar preprocesamiento y relevancia según problema. Inspiración de
  ponderaciones/relevancia, no afirmación de implementación WERCS completa.
- Yang et al. (2021), [Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html):
  distribución continua, cercanía entre etiquetas y soporte de características.
  No prescribe igualar todos los ppm ni exige cambiar a una red neuronal.
  LDS/FDS no se implementaron; no iniciar búsqueda de familias por este cierre.
- Roberts et al. (2017), [Cross-validation strategies for structured data](https://doi.org/10.1111/ecog.02881):
  dependencia espacial/jerárquica, validación de acuerdo con el uso esperado.
  Mantener evaluación espacial separada; los folds aleatorios académicos
  reutilizados no son una prueba espacial ni externa independiente.
- [imbalanced-learn: common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls)
  y [Avoiding common machine learning pitfalls (2024)](https://doi.org/10.1016/j.patter.2024.101046):
  particionar antes de remuestrear, no usar TEST en selección, evitar sobreajuste
  secuencial a los mismos tests. Justifican cierre acotado, no nuevas rondas.
- [Arne y Garrett, análisis geoquímico multivariado](https://pubs.geoscienceworld.org/gsl/geea/article/26/3/geochem2025-077/729608/A-workflow-for-the-analysis-of-multivariate):
  composición, relaciones elementales y cobertura multivariada. Conservar
  ratios previamente evaluados, recalcularlos desde mediciones, no añadir
  columnas de proxies de objetivo. La revisión de literatura no certifica
  umbrales específicos de .1°, q95 o pesos usados en este proyecto.

Evaluación del trabajo: correcto conservar TEST, auditar unidades y procedencia,
reconstruir mediciones, filtrar soporte, registrar resultados negativos. Límites:
cohorte académica truncada distinta de OASIS, uso reiterado de TEST de desarrollo,
recuperación asistida retrospectivamente por Li, método/CRS no certificados,
falta de reserva independiente, pocos grupos/observaciones de cola, filtros
heurísticos (pueden excluir transferencia beneficiosa), falta de algunos
originales primarios. No atribuir todo el error al balanceo: ruido de medición,
heterogeneidad geológica, relaciones no observadas y extrapolación son factores
posibles. No se ha medido un error irreducible. No garantizar RMSE objetivo.

## 6. Representatividad y decisión de cierre

Académico: 3263, rango congelado 4.2439–126.12 ppm. Bandas:
<=5 81; 5–10 688; 10–20 1565; 20–40 794; 40–60 95; 60–80 21; 80–100 11;
>100 8. Todos los intervalos tienen representación, no todos los valores
continuos ni combinaciones químicas/geográficas. 135 >40 (4.14%) explican
~50.83% SSE del HGB combinado; RMSE medio 7.074911, R² .631231, MAE 3.892598.
Objetivo RMSE <5.361 (reducción ~24.23%) y R² >.499. En dominio completo,
3420 y 244 grupos; conservar extremos, no alterar población de evaluación.
No equiparar RMSE promedio de folds con RMSE global ni los dos protocolos.

Aumento sintético previo: 115 nuevos fits, control 7.074911, SMOTER 7.230849,
todos los otros brazos también empeoran; no repetirlos. Candidatos Chen
compatibles 1588, 1482 <=20, cuatro >40, ningún >60; Puna ninguno.
No sumar masivamente sin ponderación. No crear PPM intermedios como si fueran
nuevas observaciones medidas. Balance deseable = cobertura por concentración,
química, grupos espaciales y publicación, evaluada mediante error real.
La concentración en ppm es objetivo continuo, no 126 clases.

**Decisión de cierre de adquisición:** no seguir persiguiendo tablas públicas
durante esta iteración. Dejar fuentes pendientes documentadas. No relajar gates
ni autorizar filas sin metadatos; usar únicamente candidatos Chen del preflight
como sensibilidad experimental con autorización explícita en comando.
Completar una comparación fija de composición de datos y detener la búsqueda
si no alcanza criterio. No crear una reserva independiente ficticia ni promover.

Implementación augmentation.data_closure: 15 fits nuevos HGB (3 x 5),
reutilizar A histórico enriquecido y B SMOTER archivado; misma lista de
predictores por fold archivada, hiperparámetros, semilla 42, validación
3263 TEST congelados. Sin búsquedas ni cambios productivos; tampoco nuevas
selecciones internas porque las tres políticas se fijan antes de correr,
se comparan descriptivamente en desarrollo conocido y no se promueven.
C1 histórico + solo cola Chen compatible; C2 histórico + Chen compatible
por fold con dosis limitada; D C2 + SMOTER histórico de TRAIN únicamente
(reutiliza fórmula, semilla de outer y presupuesto anteriores, no sintetiza
desde Chen). Para ambos C y D, peso individual máximo .10 hasta 40 ppm
y .25 sobre 40; masa efectiva nueva por banda <=10% de histórica TRAIN;
si no hay histórico en banda, peso cero. Son hipótesis conservadoras
predeclaradas, no incertidumbre calibrada. No uniformizar PPM ni reducir
TEST. No cambiar variables del esquema enriquecido y no añadir métodos
desconocidos como mediciones ICP-MS certificadas; mantener categorías
desconocidas ausentes. No modificar dataset productivo.
Guardar OOF pareado, métricas de banda, MAE/RMSE/R² y sesgo cola,
membresías/pesos, esquema/particiones y valores sintéticos/linaje.
Criterio de cierre numérico: RMSE medio <5.361 y R² medio >.499,
RMSE menor al control, RMSE de cola no peor, MAE medio <= control+.1 ppm
(propuesta conservadora). Si no cumple, cerrar con mejor evidencia y brecha
reportada; no iniciar otro ajuste por defecto. Si cumple, solo candidato
exploratorio, no superioridad científica ni fit definitivo sin evaluación
espacial e independiente. No afirmar conjunto final aprobado. Los artefactos
de ensamblaje reproducen composición experimental por fold, no un único
CSV mezclado para entrenar y evaluar sobre sí mismo.

Ejecución única de cierre experimental (CMD raíz):
```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.augmentation.data_closure --allow-unverified-metadata
```
85 pruebas esperadas (82 previamente aprobadas, 3 nuevas sin validar).
Salida results/generalization/model_1_data_closure/<run>. Adjuntar manifest y
todas las tablas. No descargar, no repetir preflight, no repetir 115 fits.
No se ejecutó localmente en esta revisión. M2, producción y main intactos.
