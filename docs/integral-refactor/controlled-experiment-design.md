# Protocolo experimental propuesto (sin ejecutar)

Los artefactos históricos son retrospectivos. No seleccionar una configuración en los folds externos y después presentar esos mismos folds como prueba independiente.

## Optimización
- Congelar datos, grupos, folds, versiones, espacio de hiperparámetros y criterio de optimización antes de ejecutar.
- Comparar 20 frente a 100 intentos por familia, con y sin pruning, semillas 42, 43 y 44. Presupuestos de intentos iguales no garantizan coste igual: registrar tiempo, memoria, ajustes completos y parciales.
- Mantener los presupuestos históricos diferenciados como brazo separado. Analizar parámetros en límites sin ampliar rangos a partir del TEST.
- Selección de configuración exclusivamente en validación interna; evaluar los folds externos una sola vez por protocolo preespecificado. Informar métricas por fold, OOF global e intervalos de diferencias pareadas por grupo.

## Predictores y preprocesamiento M1
- Comparar referencia de 33 variables con las cuatro ablaciones predefinidas, legacy frente a stable_v1; para TabNet, máscara de ausencia como variante adicional.
- Congelar muestras, objetivo, unidades, grupos y folds. No eliminar extremos ni columnas usando el TEST. Reportar cobertura, errores por fuente/estrato, sensibilidad de Li alto auxiliar y resultados sin coordenadas.
- Si se selecciona una variante utilizando resultados externos, reservar una evaluación independiente nueva.

## M2
- Separar 706 de 783 observaciones; contrastar representaciones en exactamente las mismas observaciones, QA, etiquetas por TRAIN, folds, modelos y presupuestos.
- El contraste retrospectivo RF anterior/v4 no aísla el efecto de red-edge. Comparar únicamente una modificación por brazo.
- Mantener la evaluación de prospectividad separada de la viabilidad de depósitos. No promover por métricas OOF.

## Condición de publicación
Ajuste definitivo solo con autorización. Congelar candidato y evaluar en conjunto geológico externo genuinamente independiente, con incertidumbre y dominio de aplicabilidad. No activar automáticamente ni publicar desde esta rama.
