# LithiumScope — demostración original de diez casos con candidatos nuevos

Fecha: 12/11/2026. Reparación local en main. Sin push, release ni promoción de modelos.

## Alcance autorizado

La opción original Realizar predicción → 4. Demostración integrada automática ejecuta directamente los diez casos, prepara Sentinel-2, realiza ambas predicciones, calcula métricas, escribe Excel, CSV, figuras e informe PDF de ReportLab y abre el PDF con el navegador predeterminado. No hay submenús ni opción de omitir el PDF.

**Único cambio de comportamiento en la demostración:** cargar los candidatos fijados en config/prediction.yaml:

- Modelo 1: SVM, training_20261106_111653_m0300.
- Modelo 2: Random Forest, training_20261112_123451_m0300.

Las predicciones individuales y la predicción completa siguen utilizando los modelos activos del release anterior. El registro models/active_models.json no se modifica. Los candidatos siguen pending.

El menú src/lithiumscope/cli/predict_command.py, rebuild_pdf.py y la canalización de análisis, tablas, reporte y apertura en integrated.py se restauraron desde origin/main. La única diferencia en integrated.py es la selección de modelos según session.mode == demonstration. El módulo de PDF y sus secciones y estilos permanecen intactos.

## Error corregido

La sesión demonstration_20261112_163353_m0300 falló porque build_overall_interpretation recibió un argumento render_pdf que no existe en su firma. La modificación anterior lo había propagado incorrectamente. Ese argumento y los submenús añadidos se retiraron. La sesión fallida no se eliminó.

## Ejecutar desde CMD

    cd /d "D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Scientific-Lab"
    "D:\Librerias\Python312\python.exe" main.py

Seleccionar Realizar predicción y luego 4. Demostración integrada automática; el reporte se genera y se abre automáticamente sin preguntas adicionales.

## Evidencia de ejecución real

Se ejecutó con ambos candidatos y las diez imágenes Sentinel-2. Sesión: results/predictions/demonstration_20261112_164710_m0300. Código de salida 0, diez predicciones M1, diez M2 y diez emparejadas. El programa confirmó que envió el PDF al navegador predeterminado.

Artefactos: report.pdf, evaluation.xlsx, prediction_manifest.json, model_1/predictions.csv, model_2/predictions.csv, cross_model/joined_cases.csv, figuras, mapas y logs. El manifiesto registra las versiones y SHA-256 de los dos modelos realmente utilizados.

Métricas exploratorias de los diez casos: Modelo 1 RMSE 3,230 ppm, MAE 2,746 ppm, R² 0,807; Modelo 2 ROC-AUC 0,562, AP 0,611, sin positivos bajo su umbral operativo. No son métricas del entrenamiento ni validación geológica definitiva.

La generación del PDF no implica aprobación editorial ni autorización para publicar un release. Las correcciones visuales del informe se revisarán posteriormente con el usuario.
