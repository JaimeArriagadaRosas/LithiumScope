# LithiumScope

> Herramienta local y modular para reproducir, extender y operacionalizar un estudio de predicción de litio en muestras de roca, incorporando una segunda línea de prospectividad espacial basada en imágenes.

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Arquitectura de LithiumScope" width="100%">
</p>

## 1. Antecedentes

LithiumScope nace a partir de un trabajo académico previo orientado a **predecir la concentración de litio (`Li_icpms`, ppm) en muestras de roca** utilizando información geoquímica, geológica y geoespacial.

El trabajo base parte de una tabla donde cada fila representa una muestra de roca y las columnas contienen, entre otras variables:

- óxidos mayores (`SiO2`, `TiO2`, `Al2O3`, `Fe2O3`, `MnO`, `MgO`, `CaO`, `Na2O`, `K2O`, `P2O5`);
- elementos traza como `Th_icpms`, `U_icpms`, `Rb_icpms`, `Cs_icpms`, `Nb_icpms`, `Ta_icpms`, `Pb_icpms`, `Ba_icpms`, `Sr_icpms`, `Zr_icpms`, `V_icpms` y `Hf_icpms`;
- coordenadas;
- edad geológica, tipo de muestra, tipo de roca, arco y dominio;
- `Li_icpms` como variable objetivo.

El documento de referencia informa **2.635 registros iniciales**, 787 mediciones válidas de `Li_icpms` y 681 muestras finales tras los filtros definidos por sus autores.

Los modelos comparados en el trabajo base fueron Random Forest, XGBoost, SVM, TabNet y un experimento mediante API de GPT. LithiumScope no copia un notebook ni conserva celdas de ejecución: **reconstruye cada etapa del procedimiento como un módulo Python explícito y testeable**.

### ¿Por qué reconstruir el proyecto?

Un notebook es excelente para exploración, pero puede volverse difícil de reutilizar como herramienta. LithiumScope persigue cuatro mejoras de ingeniería:

1. una etapa del procedimiento = un archivo `.py` con una responsabilidad clara;
2. un único punto de entrada local;
3. entrenamiento y predicción desacoplados;
4. trazabilidad mediante modelos guardados, metadatos, métricas y logs.

La meta inicial no es afirmar que LithiumScope supera al trabajo previo. La primera meta es **reproducir una baseline comparable de forma modular**. Solo después se evalúan mejoras metodológicas y nuevas fuentes de datos.

---

## 2. Preguntas que busca responder

LithiumScope separa deliberadamente dos problemas.

### Modelo 1 — Predicción geoquímica de litio

**Pregunta:** dadas las propiedades conocidas de una muestra, ¿qué concentración de `Li_icpms` estima el modelo?

**Entrada:** variables geoquímicas, geológicas y geoespaciales.

**Salida:** concentración estimada de litio en ppm.

### Modelo 2 — Prospectividad / prioridad espacial

**Pregunta:** a partir de muestras conocidas y de información espectral o multibanda del terreno, ¿qué sectores presentan características compatibles con los sectores asociados a mayor concentración de litio?

**Entrada:** imágenes preprocesadas vinculadas a muestras conocidas, y sus concentraciones de Li.

**Salida:** un **score de prioridad exploratoria**. Este score **no debe interpretarse como probabilidad de un yacimiento económicamente viable**.

La segunda pregunta es deliberadamente distinta de la primera. El objetivo no es usar una imagen para inventar concentraciones químicas que el sensor no observa; es agregar una herramienta para ayudar a decidir **dónde conviene revisar o muestrear después**.

---

## 3. Utilidad esperada

En un flujo de exploración, la utilidad de LithiumScope sería apoyar dos momentos diferentes.

**Modelo 1**

```text
muestra analizada
      ↓
variables geoquímicas/geológicas
      ↓
modelo entrenado
      ↓
Li_icpms estimado
```

**Modelo 2**

```text
muestras conocidas + imágenes del terreno
                   ↓
       patrones espectrales/espaciales
                   ↓
          score de prospectividad
                   ↓
      priorización de nuevas zonas
```

Esto no busca sustituir ICP-MS, trabajo de terreno ni interpretación geológica. La utilidad propuesta es **priorizar**: reducir el espacio de búsqueda, ordenar zonas candidatas y hacer reproducible la comparación entre evidencias.

---

## 4. Fuentes de datos consideradas

### Datos tabulares del Modelo 1

Mientras se obtiene o confirma la fuente oficial del proyecto académico, el repositorio incorpora un **mirror público reproducible** de la tabla `Mamani09_Table_DR2` como fuente bootstrap:

- [Machine-learning-Rocks-Categorisation](https://github.com/inshatazeen/Machine-learning-Rocks-Categorisation)

> Importante: este repositorio público contiene una tabla compatible con la estructura investigada, pero **no se asume que sea el repositorio oficial de los autores del trabajo base**. Cuando se disponga de la fuente oficial, debe configurarse como dataset principal.

### Fuentes espectrales de referencia del Modelo 2

El gestor de datasets conoce actualmente:

- **Fregeneda–Almendra Lithium Spectral Library** — Zenodo record `4575375`.
- **GREENPEG Spectral Library** — Zenodo record `6518319`.
- **Sentinel-2** — fuente dinámica de Copernicus Data Space.

Fregeneda–Almendra y GREENPEG pueden descargarse a través de Zenodo. GREENPEG es un conjunto grande; LithiumScope no fuerza una descarga de varios GB sin una decisión explícita.

Sentinel-2 es distinto: no es un único ZIP estático. Para convertirlo en datos de entrenamiento de Modelo 2 se requiere una etapa de adquisición que consulte escenas según coordenadas, fecha, nubosidad y resolución. Esa integración queda aislada de los modelos para que pueda evolucionar sin romper el resto del proyecto.

---

## 5. Proceso reproducido para el Modelo 1

La implementación sigue la organización descrita en el trabajo base, pero cada etapa se encuentra separada.

### 5.1 Carga

`step_01_load_data.py`

Carga CSV, XLS o XLSX y valida que existan registros.

### 5.2 Límites de detección

`step_02_detection_limits.py`

Reproduce las reglas documentadas:

```text
<10  → 5
>100 → 100
```

### 5.3 Valores faltantes

`step_03_missing_values.py`

Normaliza representaciones de ausencia. La decisión de incluir `Age (Ma)` se toma solo si la variable mantiene al menos 50 % de valores presentes, según el criterio documentado.

Las imputaciones que dependen de los datos se dejan dentro del pipeline de entrenamiento para evitar utilizar información de validación al calcular medianas.

### 5.4 Variable objetivo

`step_05_target_filtering.py`

- elimina muestras sin `Li_icpms`;
- conserva el intervalo central definido por los percentiles 2,5 y 97,5.

### 5.5 Control geoquímico

`step_04_quality_control.py`

Conserva las muestras cuyo `SUM (no water)` se encuentra entre **94 y 102**, de acuerdo con el procedimiento documentado.

### 5.6 Variables categóricas

`step_06_category_cleaning.py`

Normaliza y agrupa:

- `Geologycal_age`;
- `Sample_type`;
- `Rock_type`;
- `Arc`;
- `Domain`.

SVM utiliza una agrupación más compacta, mientras que Random Forest, XGBoost y TabNet conservan la agrupación general descrita en el documento.

### 5.7 Ingeniería geoquímica

`step_07_feature_engineering.py`

Se implementan las cuatro variables derivadas descritas:

```text
Alkali_Sum = Na2O + K2O

Mg_Number = MgO / (MgO + Fe2O3 + 1e-6)

A_CNK_proxy = Al2O3 / (CaO + Na2O + K2O + 1e-6)

K_Mg_ratio = K2O / (MgO + 1e-6)
```

`A_CNK_proxy` mantiene explícitamente el término *proxy*: no se presenta como el índice molar petrológico formal.

### 5.8 Preprocesamiento y validación

`step_08_preprocessing.py` y `step_09_validation.py`

- one-hot encoding para variables categóricas;
- categorías desconocidas ignoradas durante transformación;
- escalamiento para SVM;
- validación cruzada externa e interna 5×5, `seed=42`;
- optimización mediante Optuna dentro de cada conjunto de entrenamiento externo cuando Optuna está disponible.

---

## 6. Modelo 2: contrato inicial

Modelo 2 está físicamente separado de Modelo 1.

Su primera implementación funciona con un **manifiesto de pares muestra-imagen**:

```csv
sample_id,Li_icpms,image_path
A001,18.4,data/processed/model_2/images/A001.tif
A002,7.9,data/processed/model_2/images/A002.tif
```

Cada archivo debe representar un parche multibanda asociado a la muestra correspondiente.

El baseline actual extrae estadísticas por banda y aprende a distinguir muestras en el grupo de Li alto definido por un cuantil configurable. Esto permite validar toda la infraestructura de:

- carga de imagen;
- normalización;
- extracción de características;
- entrenamiento;
- persistencia;
- inferencia.

Antes de presentar resultados científicos, este baseline deberá reemplazarse o ampliarse con una estrategia espacial validada, un esquema de adquisición Sentinel-2 y controles de generalización geográfica.

---

## 7. Experiencia de uso

<p align="center">
  <img src="docs/assets/workflow.svg" alt="Flujo de ejecución de LithiumScope" width="100%">
</p>

LithiumScope tiene un único punto de entrada:

```bash
python main.py
```

o, después de instalar el paquete:

```bash
lithiumscope
```

El menú principal se mantiene pequeño:

```text
====================================================
                    LITHIUMSCOPE
====================================================

1. Entrenar modelos
2. Realizar predicción
3. Métricas y resultados
0. Salir
```

### Entrenar modelos

Permite elegir:

```text
1. Modelo 1 — Predicción de Li
2. Modelo 2 — Prospectividad espacial
3. Entrenar ambos
```

Modelo 1 permite seleccionar Random Forest, XGBoost, SVM o TabNet.

### Realizar predicción

El usuario **no tiene que escribir una ruta**. LithiumScope intenta abrir el selector de archivos nativo del sistema operativo.

En Windows esto abre el explorador habitual para seleccionar con doble clic:

- `.csv`, `.xls`, `.xlsx` en Modelo 1;
- `.tif`, `.tiff`, `.npy` en Modelo 2.

Si el entorno no dispone de interfaz gráfica, el programa utiliza una entrada de ruta por consola como fallback.

### Métricas

La opción 3 lee los metadatos de entrenamientos guardados y muestra las ejecuciones más recientes sin volver a entrenar.

---

## 8. Descarga automática de datasets

El código de adquisición vive en:

```text
src/lithiumscope/datasets/
├── registry.py
├── downloader.py
├── validator.py
└── checksum.py
```

`registry.py` es el contrato declarativo de fuentes.

`downloader.py` soporta actualmente:

- URLs directas;
- records de Zenodo mediante su API;
- detección de fuentes dinámicas;
- reutilización de archivos ya descargados;
- protección frente a descargas grandes no autorizadas.

Los archivos externos se guardan en:

```text
data/raw/model_1/
data/raw/model_2/
```

Los datos crudos y modelos entrenados están ignorados por Git.

---

## 9. CPU y GPU

La detección de hardware se encuentra centralizada en:

```text
src/lithiumscope/core/device.py
```

LithiumScope prioriza:

```text
CUDA → MPS → CPU
```

cuando la librería y el algoritmo correspondiente soportan ese backend.

No se fuerza una GPU en modelos que no obtienen una ventaja real de ella.

Ejemplos:

- Random Forest de scikit-learn: CPU;
- SVM de scikit-learn: CPU;
- XGBoost: CUDA cuando está disponible;
- TabNet: CUDA cuando está disponible;
- futuros modelos visuales: CUDA/MPS cuando corresponda.

La disponibilidad de GPU nunca es requisito para iniciar la aplicación.

---

## 10. Logs centralizados

Todo el proyecto utiliza el mismo sistema de logging.

```text
logs/
├── lithiumscope.log
├── training/
├── prediction/
└── errors/
```

Además del log general, cada entrenamiento y cada predicción puede abrir un log de ejecución independiente. Los errores no controlados terminan también en `logs/errors/errors.log`.

---

## 11. Persistencia y trazabilidad

Cada entrenamiento genera dos piezas:

```text
models/model_1/trained/random_forest_<timestamp>.joblib
models/model_1/metadata/random_forest_<timestamp>.json
```

El JSON registra, entre otros: algoritmo, fecha, número de muestras, variables, dispositivo, métricas, duración y parámetros. Así, una predicción posterior utiliza un modelo ya entrenado sin volver a ejecutar el entrenamiento.

---

## 12. Instalación

```bash
git clone https://github.com/JaimeArriagadaRosas/LithiumScope.git
cd LithiumScope
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instalación editable:

```bash
pip install -e .
```

Para XGBoost, Optuna, PyTorch y TabNet:

```bash
pip install -e ".[ml]"
```

Para GeoTIFF:

```bash
pip install -e ".[imagery]"
```

Desarrollo:

```bash
pip install -e ".[dev]"
pytest
ruff check src tests main.py
```

---

## 13. Estructura

```text
LithiumScope/
├── main.py
├── config/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/assets/
├── logs/
├── models/
├── results/
├── src/lithiumscope/
│   ├── cli/
│   ├── core/
│   ├── datasets/
│   ├── model_1/
│   │   ├── steps/
│   │   ├── training/
│   │   ├── prediction/
│   │   └── evaluation/
│   ├── model_2/
│   │   ├── steps/
│   │   ├── training/
│   │   ├── prediction/
│   │   └── evaluation/
│   └── persistence/
└── tests/
```

### Principio de diseño

```text
cada step_XX.py    → sabe hacer una etapa
pipeline.py        → sabe en qué orden ejecutarlas
trainer.py         → sabe entrenar
predictor.py       → sabe inferir
main.py / menu.py  → sabe qué pidió el usuario
```

---

## 14. Estado y próximos hitos

### Implementado

- estructura modular sin notebooks;
- menú local;
- selector gráfico de archivos;
- logs centralizados;
- gestor de rutas;
- descarga directa y Zenodo;
- detección CPU/GPU;
- etapas del Modelo 1 descritas en el informe;
- cuatro variables geoquímicas derivadas;
- Random Forest, SVM y conectores opcionales para XGBoost/TabNet;
- validación anidada 5×5;
- persistencia de modelos y metadata;
- predicción tabular;
- pipeline inicial de imágenes para Modelo 2;
- tests y GitHub Actions.

### Pendiente de validación científica

- confirmar el dataset/repositorio oficial del trabajo de origen;
- contrastar cada resultado con los scripts originales si son entregados;
- fijar la adquisición de imágenes Sentinel-2 para las coordenadas de las muestras;
- construir el manifiesto muestra ↔ imagen;
- definir validación espacial de Modelo 2;
- comparar el Modelo 1 reproducido contra la baseline documentada;
- evaluar si nuevas fuentes geoquímicas aumentan la generalización.

---

## 15. Referencias de datos y contexto

- Mamani et al. / dataset público utilizado como bootstrap: [repositorio público](https://github.com/inshatazeen/Machine-learning-Rocks-Categorisation).
- Fregeneda–Almendra Lithium Spectral Library: [Zenodo 4575375](https://doi.org/10.5281/zenodo.4575375).
- GREENPEG spectral library: [Zenodo 6518319](https://doi.org/10.5281/zenodo.6518319).
- Sentinel-2: [Copernicus Data Space](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2).

---

## 16. Representación de la baseline

La siguiente figura **no muestra resultados de LithiumScope**. Resume los valores de R² reportados en el documento académico de referencia y se conserva únicamente como objetivo de comparación durante la fase de reproducción.

<p align="center">
  <img src="docs/assets/baseline-reference.svg" alt="Baseline de referencia" width="100%">
</p>

| Modelo | R² promedio | RMSE promedio (ppm) |
|---|---:|---:|
| Random Forest | 0.499 | 5.361 |
| XGBoost | 0.467 | 5.528 |
| SVM | 0.348 | 6.124 |
| TabNet | 0.399 | 5.828 |
| GPT (evaluación distinta, 50 muestras) | -0.157 | 6.934 |

La comparación con GPT no debe interpretarse como equivalente a la validación cruzada anidada de los otros cuatro modelos, porque el esquema de evaluación documentado fue diferente.

---

## 17. Objetivo de ingeniería

LithiumScope se considera exitoso en su primera fase cuando puede demostrar esta secuencia:

```text
trabajo académico base
        ↓
reproducción modular
        ↓
métricas comparables
        ↓
modelo persistente
        ↓
predicción local reutilizable
        ↓
extensión de prospectividad espacial
```

La prioridad es mantener una separación clara entre **lo reproducido**, **lo modificado** y **lo experimental**.
