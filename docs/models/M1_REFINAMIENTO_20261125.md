# Refinamiento dirigido de M1 — 25/11/2026

## Evidencia previa

Evaluación experimental local 20261125T111144_fe968d37: 1800/1800 ajustes interiores y 60/60 exteriores exitosos, 3420 observaciones OOF por algoritmo y modo. El menor RMSE medio por fold fue CatBoost 79.4087 ppm (referencia fija 85.4168), HGB 81.9934 (108.9570); RF obtuvo RMSE global 141.4558 con selección (141.8243 referencia). El log1p de HGB fue seleccionado en los cinco folds y eliminó 171 predicciones negativas exteriores. CatBoost y SVM conservaron una predicción negativa cada uno. TabNet log1p presentó una partición con RMSE superior a 14 millones de ppm.

## Diseño de segunda iteración

26 políticas predeclaradas incluyendo seis referencias, cinco folds exteriores, 130 ajustes diagnósticos. Comparación de hipótesis seleccionadas a partir de la primera evaluación: combinaciones de bootstrap espacial/rare-group, ausencia de coordenadas, raíz cuadrada, log1p acotado al máximo observado exclusivamente en TRAIN y escalado robusto de TabNet dentro de la partición de ajuste. La referencia permanece sin proyección; las alternativas usan proyección no negativa, con registro de negativos latentes. El límite superior explícito no constituye un máximo físico ni autoriza extrapolación. Se conservan observaciones y concentraciones originales. El protocolo verifica hashes del snapshot y plan EDA, alineación de filas/targets/folds del OOF anterior y hash de fuente; no requiere reconstruir las 360 preparaciones si sigue compatible.

Se reporta RMSE global y medio, peor fold, R², MAE, sesgo, error de altas concentraciones (umbral q75 de TRAIN), casos fuera del máximo de TRAIN, proyecciones y diferencias pareadas con bootstrap por grupo. No se selecciona ni publica un ganador automáticamente. Las hipótesis se originaron en los folds reutilizados: todo resultado es desarrollo exploratorio, no validación externa ni estimación confirmatoria. La adquisición de observaciones de alta concentración espacialmente independientes sigue pendiente.

## Ejecución local (después de terminar M2 y actualizar)

    python -m lithiumscope.experiments.model_1.refinement --previous-run 20261125T111144_fe968d37

Opcional: repetir --algorithm para limitar familias. No se añade ninguna opción al menú; esta herramienta dirigida se mantiene separada hasta revisar ambos modelos. No guarda predictores entrenados, no promueve, no modifica la competencia normal ni ejecuta workflows. Enviar el manifest.json y la salida de consola del directorio results/generalization/model_1_refinement. No ejecutar todavía si M2 sigue en curso.
