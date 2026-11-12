# Demostración integrada de diez casos — candidatos experimentales

Fecha: 12 de noviembre de 2026. Implementado **solo en el clon local de main**, pendiente de aprobación del usuario. Ninguna activación, push, tag ni release.

## Modelos disponibles

El menú de demostración permite seleccionar una pareja de candidatos **explícita y fijada en config/prediction.yaml**, evitando que una nueva fecha de modificación de un archivo cambie accidentalmente la selección:

- Modelo 1, SVM: `training_20261106_111653_m0300` (candidate_status = pending).
- Modelo 2, Random Forest: `training_20261112_123451_m0300` (candidate_status = pending).

Alternativamente, puede seleccionarse la pareja del release anterior. Los modelos activos en `models/active_models.json` no se modifican ni se reemplazan al usar los candidatos.

El Modelo 2 utiliza su umbral operativo de aproximadamente 0,4416 y **aplica a cada imagen el control QA95 de píxeles válidos conjuntos**, como hace su predictor individual. Una imagen insuficiente produce abstención motivada; nunca una puntuación artificial. El fichero `model_2/case_imagery_status.csv` registra los diez casos, su estado Sentinel, si se puntuó, si se abstuvo y por qué. Los casos sin imagen o rechazados no se cuentan como predicciones verificadas.

## Procedimiento desde CMD

```cmd
cd /d "D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Scientific-Lab"
"D:\Librerias\Python312\python.exe" main.py
```

Seleccionar **2. Realizar predicción**, luego **4. Demostración integrada automática**.

En el submenú escoger **1. Candidatos nuevos** y después:

- **2. Datos y evidencias sin PDF**: se ejecutan ambos modelos, se guardan predicciones, fuentes, tablas, figuras, métricas y manifiesto de versiones; el PDF no se genera ni abre. Útil para revisar primero los valores.
- **1. Datos, evidencias y PDF**: mismo análisis, generando y abriendo `report.pdf` al finalizar si el navegador está disponible.

Se utiliza el conjunto de diez casos externos versionado `examples/integrated_demo/cases.csv` (Castro et al. 2021), y se preparan sus imágenes Sentinel-2 mediante el caché configurado en `data/interim/model_2/integrated_demo`. Si la adquisición satelital falla o los datos no están disponibles, no se fabrican predicciones.

Los artefactos se generan bajo `results/predictions/demonstration_<fecha>_<hora>_<offset>/`, incluyendo:

- `model_1/predictions.csv` — Li estimado, concentraciones observadas y advertencias de dominio.
- `model_2/predictions.csv` — scores espectrales de los casos con imagen válida.
- `model_2/case_imagery_status.csv` y `model_2/diagnostics.json` — trazabilidad de casos fallidos o abstenciones QA.
- `cross_model/joined_cases.csv`, `correlations.csv`, `concordance.csv` — comparación entre modelos.
- `evaluation.xlsx`, `training_vs_external.csv`, `prediction_manifest.json`, `prediction.log`, figuras y mapas.
- Opcionalmente, `report.pdf`.

El manifiesto incorpora los ID, algoritmos y hashes SHA-256 de **ambos modelos seleccionados**, además de `pdf_rendered`. Los candidatos se usan solo en la sesión de demostración; las opciones individuales de predicción siguen usando los modelos activos.

## Renderizar PDF posteriormente sin reentrenar ni volver a descargar

Una vez terminada una demostración **sin PDF**, desde la raíz del repositorio:

```cmd
set "PYTHONPATH=%CD%\src"
"D:\Librerias\Python312\python.exe" -X utf8 -m lithiumscope.prediction.rebuild_pdf --run "demonstration_YYYYMMDD_HHMMSS_m0300"
```

Sustituir el nombre por el ID de la carpeta de resultados. La utilidad reconstruye el PDF a partir de predicciones, tablas, figuras y metadatos guardados. Actualiza el manifiesto de esa ejecución; no vuelve a ejecutar inferencia, no cambia modelos activos y no publica un release.

## Límites

La muestra demostrativa de diez casos es pequeña y su procedencia externa **no demuestra** independencia completa de los datos de entrenamiento sin un contraste adicional. El score del Modelo 2 ordena candidatos de estudio; no es probabilidad de encontrar o poder explotar un depósito. Los PDF futuros son evidencia de una **ejecución reproducible**, no una validación geológica definitiva.

Este cambio habilita la revisión y corrección posterior del informe. El empaquetado final de release conjunto Modelo 1 + Modelo 2 y los cambios visuales del PDF quedan fuera del alcance, a la espera de la aprobación explícita del usuario.
