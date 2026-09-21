# Variables de entrada y salida

Este documento define el contrato de datos observable de LithiumScope para
entrenamiento, predicción individual y predicción integrada.

> Regla científica central: Modelo 1 y Modelo 2 son independientes. Modelo 1 no
> alimenta Modelo 2. La integración ocurre después de predecir, únicamente
> cuando los resultados pueden emparejarse por `case_id`.

---

## 1. Resumen

| Componente | Entrada principal | Salida científica |
|---|---|---|
| Modelo 1 | variables geoquímicas, geológicas y espaciales de una muestra | estimación de `Li_icpms` en ppm |
| Modelo 2 | parche Sentinel-2 multibanda | score de prioridad exploratoria relativa |
| Predicción integrada | resultados independientes de M1 y M2 con `case_id` común | correlaciones, concordancia, discrepancias y reporte conjunto |

El score del Modelo 2 **no es concentración de litio ni probabilidad de
yacimiento**.

---

## 2. Modelo 1 — entrada de entrenamiento

El target es:

```text
Li_icpms
```

Durante entrenamiento, el pipeline selecciona únicamente columnas disponibles
en el dataset y guarda el esquema final dentro del bundle del modelo.

### 2.1 Variables numéricas candidatas

#### Óxidos mayores

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

#### Elementos traza

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

#### Coordenadas candidatas

El código reconoce estos nombres:

```text
Longitude (X)
Logintude (X)
Longitude
longitude
Latitude (Y)
Latitude
latitude
```

El bundle entrenado conserva los nombres exactos utilizados. No debe asumirse
que todos aparecen simultáneamente en todos los datasets.

#### Edad

Se utiliza una sola variante disponible entre:

```text
Age (Ma)
Age (ma)
Age
age_ma
```

La edad se omite si su fracción de datos faltantes supera el límite configurado.

### 2.2 Variables categóricas candidatas

```text
Geologycal_age
Sample_type
Rock_type
Arc
Domain
```

Las categorías desconocidas durante predicción son aceptadas por el
`OneHotEncoder(handle_unknown="ignore")`.

### 2.3 Features derivadas automáticamente

El usuario no necesita entregarlas. Se generan dentro del pipeline:

```text
Alkali_Sum = Na2O + K2O

Mg_Number = MgO / (MgO + Fe2O3 + 1e-6)

A_CNK_proxy = Al2O3 / (CaO + Na2O + K2O + 1e-6)

K_Mg_ratio = K2O / (MgO + 1e-6)
```

### 2.4 Columnas usadas para control, no necesariamente como predictors

El entrenamiento puede usar una columna de suma geoquímica para control de
calidad:

```text
SUM (no water)
SUM(no water)
SUM (water)
```

Se aceptan muestras dentro del rango configurado:

```text
94 <= suma <= 102
```

`Li_icpms` es target durante entrenamiento. Nunca se incorpora al vector de
features.

---

## 3. Modelo 1 — entrada de predicción

Formato admitido:

```text
CSV
XLS
XLSX
```

La predicción utiliza el esquema guardado en el modelo activo.

Antes de predecir se aplican las mismas transformaciones estructurales:

1. limpieza de límites de detección;
2. normalización de valores faltantes;
3. limpieza de categorías;
4. generación de features derivadas;
5. conversión de variables numéricas;
6. imputación/preprocesamiento dentro del pipeline.

Si una columna esperada no está presente, LithiumScope la crea como valor
faltante y la deja al preprocesador del modelo. El diagnóstico registra qué
columnas faltaban originalmente y cuáles fueron generadas durante preparación.

### Importante

Para una predicción real de Modelo 1:

- `Li_icpms` **no es obligatorio**;
- si aparece, no se usa como predictor;
- las columnas derivadas no deben ser preparadas manualmente;
- cuantos más campos esperados falten, mayor puede ser la advertencia de
  aplicabilidad.

---

## 4. Modelo 1 — salida

El CSV de predicción conserva las columnas originales y añade:

| Campo | Significado |
|---|---|
| `Li_icpms_predicted` | concentración de Li estimada por M1, en ppm |
| `Li_icpms_interval_low_q90` | límite inferior del intervalo empírico q90, si existe |
| `Li_icpms_interval_high_q90` | límite superior del intervalo empírico q90, si existe |
| `out_of_training_range_fraction` | fracción de variables numéricas fuera del perfil q01–q99 de entrenamiento |
| `applicability_warning` | `OK` o `OUT_OF_DOMAIN` |

El umbral actual para marcar `OUT_OF_DOMAIN` es una fracción superior a
`0.25`.

También se genera un JSON de diagnóstico con:

- algoritmo;
- filas procesadas;
- columnas esperadas;
- columnas ausentes en el input;
- columnas derivadas generadas;
- columnas todavía ausentes después de preparar;
- número de filas OOD;
- fracción OOD media;
- error absoluto OOF q90 usado para el intervalo;
- hash del input;
- hash del modelo.

---

## 5. Transformación del target en entrenamiento de Modelo 1

La competencia puede evaluar dos representaciones del mismo target:

```text
identity -> Li_icpms original
log1p    -> log1p(Li_icpms)
```

`log1p` se prueba actualmente solo con:

- Random Forest;
- SVM-RBF;
- CatBoost.

La transformación es interna al entrenamiento. Toda predicción se transforma
de vuelta automáticamente a ppm antes de calcular RMSE, MAE y R² o antes de ser
entregada al usuario.

No existen dos tipos distintos de salida de Modelo 1: la salida científica sigue
siendo siempre Li estimado en ppm.

---

## 6. Modelo 2 — entrada de entrenamiento

Modelo 2 parte de un manifiesto de muestras emparejadas con imágenes
Sentinel-2.

El manifiesto necesita, como mínimo:

```text
Li_icpms
image_path
```

Y utiliza cuando está disponible:

```text
sample_id
spatial_group
Longitude / Latitude
sentinel_scene_id
sentinel_cloud_cover
sentinel_datetime
```

`Li_icpms` se usa únicamente para construir la etiqueta de referencia de
entrenamiento y evaluar el clasificador. No forma parte de las features
espectrales entregadas al estimador.

La etiqueta binaria de entrenamiento se deriva de:

```text
Li_icpms >= umbral de referencia -> clase positiva
Li_icpms <  umbral de referencia -> clase negativa
```

Por defecto el umbral de referencia se obtiene del cuantil 0,75 del dataset de
entrenamiento, salvo que exista un umbral fijo configurado.

---

## 7. Modelo 2 — entrada de predicción

Formatos admitidos:

```text
GeoTIFF / TIFF
NPY
```

El modelo activo guarda:

- bandas esperadas;
- lista exacta de features;
- configuración de normalización;
- perfil de aplicabilidad;
- threshold operativo aprendido.

Con el esquema estándar se esperan seis bandas:

```text
red
green
blue
nir
swir16
swir22
```

El orden y esquema reales deben tomarse siempre del bundle del modelo activo.

El input se preprocesa y se convierte automáticamente al vector de features
espectrales requerido.

---

## 8. Modelo 2 — salida

La predicción individual genera:

| Campo | Significado |
|---|---|
| `prospectivity_score` | score relativo generado por el clasificador |
| `priority` | `baja`, `media` o `alta` |
| `operating_positive` | clasificación binaria usando el threshold operativo guardado |
| `operating_threshold` | threshold OOF utilizado por el modelo |
| `algorithm` | algoritmo del bundle activo |
| `out_of_training_range_fraction` | fracción de features fuera del rango q01–q99 de entrenamiento |
| `applicability_warning` | `OK` o `OUT_OF_DOMAIN` |
| `warning` | recuerda que el score no es probabilidad de depósito |

Prioridad visual actual:

```text
score < 0.40        -> baja
0.40 <= score < .70 -> media
score >= 0.70       -> alta
```

Esta prioridad visual y `operating_positive` son conceptos distintos. El
threshold operativo de modelos nuevos se aprende usando predicciones OOF del
entrenamiento.

El JSON de diagnóstico incluye además:

- features esperadas;
- features faltantes;
- fracción OOD;
- umbral de Li usado para construir la referencia;
- cuantil de target;
- threshold operativo;
- objetivo usado para seleccionar el threshold;
- hashes de input y modelo.

---

## 9. Predicción integrada

La predicción integrada no crea una cadena M1 -> M2.

El flujo es:

```text
input M1 -> Modelo 1 -> resultado M1
                              \
                               emparejamiento por case_id
                              /
input M2 -> Modelo 2 -> resultado M2
```

Cuando existe un `case_id` válido en ambos lados se calculan, según
disponibilidad:

- Li real vs. Li predicho M1;
- Li real vs. score M2;
- Li predicho M1 vs. score M2;
- error absoluto M1 vs. score M2;
- OOD M1 vs. OOD M2;
- Pearson;
- Spearman;
- concordancia alta/baja;
- divergencia M1-alto;
- divergencia M2-alto.

Una discrepancia entre modelos es un resultado válido y no se corrige
forzadamente.

---

## 10. Demostración integrada

El CSV versionado de demostración contiene:

```text
case_id
variables compatibles con M1
Longitude
Latitude
Li_icpms
source_dataset
source_sample
```

`Li_icpms` en la demostración es **ground truth** para evaluación externa.

No se entrega al estimador de M1 como feature y tampoco se entrega a M2 como
feature. En M2 se usa después de predecir para calcular métricas externas contra
la referencia binaria definida por el modelo.

Sentinel-2 se obtiene automáticamente a partir de las coordenadas del mismo caso.

---

## 11. Fuente de verdad del contrato

El contrato definitivo de una ejecución concreta está en los artefactos del
modelo, no únicamente en este documento:

```text
models/<model_group>/trained/<run_id>/
├── model.joblib
├── metadata.json
├── feature_schema.json
├── training_config.yaml
└── dataset_manifest.json
```

En particular:

- `feature_schema.json` define el esquema real de M1;
- el bundle de M2 define `features`, `band_names` y normalización;
- `metadata.json` identifica algoritmo, transformación/extractor y métricas.

Este documento describe el contrato implementado actualmente, pero los
artefactos versionados permiten reproducir exactamente el contrato histórico de
cada entrenamiento.
