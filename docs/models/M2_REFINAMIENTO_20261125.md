# Refinamiento dirigido M2 — 25/11/2026

Evidencia local 20261125T130441_de33c9e8: 990/990 evaluaciones internas, 60/60 externas, 8472 predicciones, 102770 registros de trazabilidad. Selección interna verificada 30/30. Solo RF y SVM mejoraron AUC medio externo (0.5919→0.5990 y 0.5869→0.5924); solo SVM mejoró conjuntamente AP (0.3665→0.3716). PCA fue inestable. El fold 1 tiene 156 muestras de un solo grupo; fold 3 20 positivos de 138 observaciones. Se preservan los folds, las etiquetas TRAIN-dependientes y el QA TRAIN-only.

Segunda iteración: 26 políticas, 130 ajustes externos diagnósticos. Cada familia incluye referencia y regularización aislada; se prueba regularización con masa unitaria por firma (ponderada o comprimida), ablación V3 126 vs V4 170 y selección KBest48 para boosting según la lista declarada. Las firmas se calculan antes de transformar. No se ejecuta PCA ni aumentación SMOTE en este refinamiento. Las políticas conservan todas las observaciones TEST y comparan AUC/AP por fila, firma y grupo (peso inverso al tamaño). Se informa cada delta por fold, brecha TRAIN/TEST y peor fold. Las puntuaciones SVM siguen siendo márgenes; no se presentan como probabilidades, ni se ajustan umbrales sobre TEST.

La segunda iteración reutiliza folds ya observados. No demuestra superioridad independiente, no cambia el entrenamiento normal, no guarda estimadores, no selecciona ni promueve. Requiere cohorte congelada de 706 muestras, evidencia V4, snapshot y plan EDA idénticos. Las hipótesis de ablación pueden afectar el peso de firmas y se mantienen explícitamente separadas. No se añade opción de menú.

Ejecutar después de git pull:

    python -m lithiumscope.experiments.model_2.refinement --previous-run 20261125T130441_de33c9e8

Enviar el manifest.json, summary.csv y fold_metrics.csv de results/generalization/model_2_refinement. La validación predictiva con nuevos sitios independientes y evaluación integrada M1→M2 permanecen pendientes.
