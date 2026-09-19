# Modelo 2: prospectividad espacial

Modelo 2 es una herramienta de priorización exploratoria. Su score no representa una probabilidad de yacimiento económicamente viable.

## Representación espectral

Los parches Sentinel-2 se limpian conservando su escala entre muestras. No se normaliza cada banda de cada parche a media cero y desviación uno antes de extraer las características, porque esa transformación eliminaría parte de la información de reflectancia que se pretende comparar entre ubicaciones.

Las características actuales incluyen:

- estadísticos robustos por banda: media, desviación, percentiles e IQR;
- NDVI;
- NDMI;
- NBR;
- diferencia normalizada red/SWIR16;
- ratios SWIR16/SWIR22, NIR/SWIR16, red/blue y green/red.

La estandarización necesaria para un algoritmo concreto debe ocurrir dentro del pipeline del modelo y, por tanto, dentro del conjunto de entrenamiento de cada fold.

## Caché de características

`data/processed/model_2/sentinel_features.csv` almacena las características derivadas.

El caché se invalida cuando cambia:

- el hash del manifest de entrenamiento;
- el conjunto y orden de bandas;
- la política de normalización;
- la versión del extractor de características.

Esto evita releer los 706 parches cuando la transformación no cambió sin permitir reutilizar features obsoletas.

## Validación y optimización

La validación externa utiliza grupos espaciales cuando existen grupos suficientes.

La optimización de hiperparámetros utiliza una validación interna que también respeta los grupos espaciales. Optuna persiste los estudios en SQLite y puede podar trials poco prometedores después de un periodo conservador de calentamiento.

Para SVM:

- durante tuning se usa `decision_function`, suficiente para ROC-AUC;
- `SVC(probability=True)` no se utiliza;
- los scores probabilísticos se obtienen mediante calibración sigmoidal en folds internos;
- cuando hay grupos espaciales, la calibración utiliza splits espaciales.

## Definición de la clase positiva

Se soportan dos fuentes de umbral:

1. `fixed_threshold_ppm`: valor fijo definido externamente;
2. `high_lithium_quantile`: cuantil del dataset cuando no hay umbral fijo.

El manifest registra explícitamente cuál fue utilizado.

La migración a un objetivo continuo de Li es una decisión metodológica separada y no se activa automáticamente, para conservar comparabilidad con las ejecuciones binarias anteriores.

## Release gate

Que un algoritmo sea el primero del ranking no implica automáticamente que Modelo 2 sea apto para versionar.

El release gate exige una mejora mínima sobre el baseline en:

- ROC-AUC;
- Average Precision.

Los mínimos se configuran en `config/model_2.yaml`. Si el entrenamiento termina pero no supera el gate, el run puede permanecer completo para análisis, pero no se considera candidato automático a tag.
