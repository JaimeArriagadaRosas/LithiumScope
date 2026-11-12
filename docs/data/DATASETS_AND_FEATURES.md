# Datasets y features

Este documento describe qué datasets conoce LithiumScope, cuáles participan
realmente en cada modelo y cómo se transforman en features de entrenamiento.

---

## 1. Mapa general de datos

```text
Mamani09_Table_DR2 (mirror público)
        |
        +----------------------------+
        |                            |
        v                            v
     Modelo 1              muestras Li + coordenadas
                                     |
                                     v
                              Sentinel-2 L2A
                                     |
                                     v
                         training_manifest Modelo 2
                                     |
                                     v
                              features espectrales
                                     |
                                     v
                                  Modelo 2
```

El entrenamiento operativo de Modelo 2 no mezcla automáticamente todas las
fuentes registradas en `datasets/registry.py`.

---

## 2. Dataset principal de Modelo 1

### Registro

```text
key: mamani09_public_mirror
modelo: model_1
provider: direct
archivo: Mamani09_Table_DR2.csv
```

Fuente registrada:

```text
https://github.com/inshatazeen/Machine-learning-Rocks-Categorisation
```

LithiumScope utiliza actualmente un mirror público reproducible de
`Mamani09_Table_DR2`.

### Advertencia de procedencia

El propio registro del proyecto indica que este mirror **no ha sido confirmado
como la copia oficial OASIS utilizada por el proyecto académico original**.

Por eso debe describirse como dataset bootstrap/reproducible actual, no como
fuente oficial definitiva.

### Ruta local

```text
data/raw/model_1/Mamani09_Table_DR2.csv
```

---

## 2.1. Extensión opcional con GEOROC

LithiumScope incorpora una capa de ingesta para archivos GEOROC revisados localmente. La fuente registrada es la compilación precompilada de márgenes convergentes / Andean Arc de GEOROC. Por su tamaño y por la necesidad de revisar compatibilidad científica, **no se descarga automáticamente**.

Flujo:

```text
GEOROC CSV revisado
        ↓
adaptación de nombres/unidades
        ↓
filtro Li + coordenadas + densidad mínima de predictores
        ↓
cálculo de suma de óxidos cuando es posible
        ↓
armonización con el contrato LithiumScope
        ↓
deduplicación con dataset base
        ↓
dataset combinado
        ↓
steps comunes del Modelo 1
```

Los archivos aprobados se colocan en:

```text
data/raw/model_1/georoc/
```

La integración se activa explícitamente en `config/model_1.yaml` mediante `data_sources.georoc.enabled: true`. Con la opción desactivada, el comportamiento de entrenamiento permanece igual al dataset bootstrap actual.

La adaptación GEOROC está separada de los `steps` científicos de Modelo 1. El adaptador resuelve diferencias de esquema; los `steps` continúan aplicando las mismas reglas de limpieza, filtrado, control de calidad, categorías e ingeniería de características al dataset ya armonizado.

El proceso genera trazabilidad en:

```text
data/interim/model_1/georoc_harmonized.csv
data/processed/model_1/training_combined.csv
data/processed/model_1/source_merge_audit.json
```

El dataset combinado conserva `source_dataset`, `source_file` y `source_sample` cuando están disponibles. Esto permite medir cuántas muestras provienen de cada fuente y auditar duplicados.


## 3. Preparación del dataset de Modelo 1

El pipeline aplica, en orden:

1. carga CSV/XLS/XLSX;
2. limpieza de límites de detección;
3. normalización de faltantes;
4. selección y filtrado de `Li_icpms`;
5. control de calidad geoquímico;
6. limpieza de categorías;
7. feature engineering;
8. selección de esquema;
9. construcción de grupos espaciales para validación.

### Target

```text
Li_icpms
```

Antes de entrenar, el target se limita al intervalo configurado:

```text
percentil 2.5 <= Li_icpms <= percentil 97.5
```

Esto corresponde a la configuración actual:

```yaml
target_lower_quantile: 0.025
target_upper_quantile: 0.975
```

### Control de calidad geoquímico

Cuando existe una columna compatible de suma:

```text
94 <= SUM <= 102
```

### Validación espacial

Cuando las coordenadas lo permiten, las muestras se agrupan en celdas de:

```text
0.5 grados
```

y M1 utiliza esos grupos para evitar que muestras del mismo bloque espacial se
repartan entre train y validación.

---

## 4. Features de Modelo 1

### 4.1 Óxidos mayores

10 candidatos:

```text
SiO2
TiO2
Al2O3
Fe2O3
MnO
MgO
CaO
Na2O
K2O
P2O5
```

### 4.2 Elementos traza

12 candidatos:

```text
Th_icpms
U_icpms
Rb_icpms
Cs_icpms
Nb_icpms
Ta_icpms
Pb_icpms
Ba_icpms
Sr_icpms
Zr_icpms
V_icpms
Hf_icpms
```

### 4.3 Contexto espacial

LithiumScope detecta variantes de longitud y latitud presentes en la tabla. El
esquema final depende del dataset y queda guardado con el modelo.

### 4.4 Contexto geológico categórico

```text
Geologycal_age
Sample_type
Rock_type
Arc
Domain
```

### 4.5 Edad

Se acepta una variante disponible de edad en Ma cuando el nivel de datos
faltantes no supera el máximo configurado.

### 4.6 Features derivadas

```text
Alkali_Sum = Na2O + K2O

Mg_Number = MgO / (MgO + Fe2O3 + 1e-6)

A_CNK_proxy = Al2O3 / (CaO + Na2O + K2O + 1e-6)

K_Mg_ratio = K2O / (MgO + 1e-6)
```

Estas variables se generan tanto en entrenamiento como en predicción.

### 4.7 Preprocesamiento

Según algoritmo:

- imputación por mediana para la mayoría de modelos;
- `StandardScaler` adicional para SVM;
- imputación constante para TabNet;
- XGBoost recibe las numéricas como passthrough;
- categóricas con imputación por moda + OneHotEncoder;
- categorías desconocidas se ignoran en predicción.

---

## 5. Experimento de transformación del target en M1

La ronda actual de challengers mantiene seis algoritmos, pero añade tres
experimentos sobre el target:

```text
Random Forest + log1p
SVM-RBF       + log1p
CatBoost      + log1p
```

Los seis algoritmos siguen compitiendo con target original.

La variante logarítmica:

1. transforma `Li_icpms` con `log1p` dentro del entrenamiento;
2. ajusta el modelo sobre esa representación;
3. aplica `expm1` automáticamente al predecir;
4. calcula RMSE, MAE y R² en ppm originales.

Por tanto no altera la unidad científica de salida.

---

## 6. Dataset operativo de Modelo 2

Modelo 2 no requiere que el usuario prepare manualmente un dataset espectral.

LithiumScope construye su dataset desde:

```text
muestra geoquímica con Li + coordenadas
             +
        Sentinel-2 L2A
```

### Fuente de Sentinel-2

Configuración actual:

```text
provider: Earth Search STAC
collection: sentinel-2-l2a
cloud_cover_max: 35
patch_size_m: 640
patch_pixels: 32
```

Bandas:

```text
red
green
blue
nir
swir16
swir22
```

### Artefactos locales

```text
data/processed/model_2/training_manifest.csv
data/processed/model_2/training_manifest.partial.csv
data/processed/model_2/training_failures.csv
data/processed/model_2/sentinel_features.csv
data/raw/model_2/sentinel2/patches/
```

El manifest final necesita al menos 50 muestras preparadas con la configuración
actual.

---

## 7. training_manifest de Modelo 2

El manifest vincula muestra y escena.

Entre sus campos relevantes se encuentran:

```text
sample_id
Li_icpms
Longitude / Latitude
spatial_group
image_path
sentinel_scene_id
sentinel_cloud_cover
sentinel_datetime
```

Su función es conservar trazabilidad entre:

```text
muestra -> coordenada -> escena -> parche -> features
```

`Li_icpms` no entra al clasificador como feature. Se utiliza para construir la
clase de referencia de entrenamiento.

---

## 8. Target de Modelo 2

El modo actual es:

```text
high_lithium_quantile
```

Configuración:

```text
high_lithium_quantile = 0.75
```

Conceptualmente:

```text
Li >= q75 del entrenamiento -> clase positiva
Li <  q75 del entrenamiento -> clase negativa
```

El umbral exacto queda almacenado en el bundle entrenado.

Esto define una referencia relativa dentro del dataset de entrenamiento. No
equivale a definir un yacimiento, una ley económica ni una concentración
estimada por M2.

---

## 9. Features espectrales de Modelo 2

El extractor actual es:

```text
FEATURE_EXTRACTOR_VERSION = 3
```

Cambiar la versión invalida el cache antiguo de features para impedir que un
modelo nuevo mezcle representaciones distintas.

### 9.1 Estadísticas por banda

Para cada una de las seis bandas estándar se calculan:

```text
mean
std
p10
p50
p90
iqr
texture_mean
texture_p90
```

Con seis bandas:

```text
6 bandas x 8 features = 48 features
```

### 9.2 Textura

`texture_mean` y `texture_p90` resumen diferencias absolutas entre píxeles
vecinos en dirección horizontal y vertical.

No son una clasificación geológica por sí mismas. Funcionan como descriptores
simples de heterogeneidad espacial dentro del parche.

### 9.3 Índices y ratios derivados

Con el esquema estándar se calculan:

```text
NDVI
NDMI
NBR
NBR2
MNDWI
bare_soil_index
dry_bare_soil_index
nir_swir22_nd
red_swir16_nd
swir16_swir22_ratio
nir_swir16_ratio
red_blue_ratio
green_red_ratio
```

Cada índice/ratio se resume mediante:

```text
mean
std
p10
p50
p90
iqr
```

13 descriptores derivados x 6 estadísticas:

```text
78 features derivadas
```

Con las seis bandas estándar, el extractor v3 produce por tanto:

```text
48 features de bandas
+
78 features de índices/ratios
=
126 features espectrales
```

si todas las bandas estándar están presentes.

### 9.4 Qué significan estos índices dentro de LithiumScope

Se utilizan como descriptores del contexto superficial:

- vegetación;
- humedad;
- diferencias NIR/SWIR;
- suelo expuesto;
- contrastes espectrales;
- heterogeneidad espacial.

No se interpretan como sensores directos de litio.

---

## 10. Preprocesamiento de imágenes

Cada parche se carga y pasa por `preprocess_image`.

Configuración actual:

```text
normalize_per_band: false
nodata_fill: median
```

La extracción de features ocurre después del preprocesamiento.

En predicción se usa la configuración almacenada en el modelo para mantener
compatibilidad con el entrenamiento histórico.

---

## 11. Validación de Modelo 2

Cuando existe `spatial_group`, se utiliza validación espacial agrupada para
evitar que muestras del mismo bloque aparezcan simultáneamente en entrenamiento
y validación.

Configuración actual:

```text
5 folds externos
3 folds internos
prefer_spatial_groups: true
spatial_group_degrees: 0.5
```

La competencia ordena por:

1. ROC-AUC;
2. Average Precision;
3. Balanced Accuracy.

También se compara contra un baseline de prevalencia.

---

## 12. Threshold operativo de Modelo 2

El score continuo y la clasificación binaria son objetos distintos.

Los entrenamientos nuevos seleccionan un `operating_threshold` usando
predicciones OOF del entrenamiento y optimizando:

```text
Balanced Accuracy
```

Si no puede aprenderse un threshold válido, existe un fallback de:

```text
0.50
```

El threshold no se ajusta mirando la demostración externa.

---

## 13. Datasets auxiliares registrados

### Fregeneda–Almendra

```text
provider: Zenodo
record: 4575375
DOI: 10.5281/zenodo.4575375
```

Descripción registrada en el proyecto:

> Lithium-dedicated spectral library for the Fregeneda-Almendra field.

### GREENPEG

```text
provider: Zenodo
record: 6518319
DOI: 10.5281/zenodo.6518319
```

Archivos seleccionados:

```text
0-Database_files.zip
Metadata.pdf
Database report.pdf
```

Descripción registrada:

> European pegmatite spectral library; selected archive is large.

### Rol actual

Estas fuentes están registradas como referencias/datasets auxiliares, pero **no
se mezclan automáticamente** con las muestras andinas usadas por el pipeline
operativo M2.

Esto evita combinar dominios geológicos distintos sin una decisión científica
explícita.

---

## 14. Dataset de demostración integrada

Ruta versionada:

```text
examples/integrated_demo/cases.csv
```

Fuente declarada:

```text
Castro et al. 2021 via andes_paleoelevation
repository: siwill22/andes_paleoelevation
commit: cc00b161ace1be8bd9ed46d306a8ac3b2dc39ecc
source path: datafiles/geoch_Castro++2021.csv
license: MIT
```

Los 10 casos contienen variables geoquímicas compatibles con M1, coordenadas,
`Li_icpms` real y procedencia.

Las imágenes Sentinel-2 correspondientes se preparan dinámicamente desde las
coordenadas.

### Rol científico actual

Después de haber sido inspeccionado y utilizado para decidir cambios de modelos,
este conjunto debe tratarse como:

```text
conjunto externo conocido de aceptación/regresión
```

y no como una nueva validación completamente ciega.

El sistema mantiene una auditoría de solapamiento por identificador y
coordenadas contra el dataset local de entrenamiento.

---

## 15. Cache y reproducibilidad

### Modelo 1

Cada entrenamiento guarda:

- manifest del dataset;
- SHA-256;
- configuración;
- feature schema;
- algoritmo;
- transformación del target;
- métricas;
- perfil de aplicabilidad.

### Modelo 2

Cada entrenamiento guarda:

- manifest;
- SHA-256;
- features esperadas;
- bandas;
- versión del extractor;
- threshold de Li de referencia;
- threshold operativo;
- métricas;
- perfil de aplicabilidad.

La versión del extractor forma parte de la identidad técnica de un modelo M2.
Un modelo histórico puede seguir funcionando porque su bundle conserva la lista
exacta de features con la que fue entrenado.

---

## 16. Qué no debe confundirse

### Dataset vs. feature

- dataset: colección de observaciones o imágenes;
- feature: variable derivada o seleccionada que entra al estimador.

### Li real vs. feature

`Li_icpms`:

- es target de M1 durante entrenamiento;
- define la clase de referencia de M2 durante entrenamiento;
- es ground truth en evaluaciones externas;
- **no es feature de entrada de M1 ni de M2 durante predicción**.

### Score M2 vs. concentración

`prospectivity_score`:

- no estima ppm;
- no representa probabilidad de yacimiento;
- sirve para ranking/priorización relativa según el dominio del modelo.

### Coordenadas en M2

Las coordenadas se utilizan para:

- buscar Sentinel-2;
- construir grupos espaciales;
- emparejar casos y escenas.

El extractor espectral actual no añade latitud/longitud directamente como
predictors del clasificador M2.
