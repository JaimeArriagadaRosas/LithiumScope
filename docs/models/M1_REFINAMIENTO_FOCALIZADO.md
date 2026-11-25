# Refinamiento focalizado de M1

Protocolo de desarrollo, no validación externa ni promoción. 15 políticas en cada
una de RF, HGB y CatBoost (45 configuraciones). Se conservan las 3420 observaciones,
los folds espaciales, los parámetros de algoritmo archivados por fold y las
predicciones OOF de referencia. El laboratorio verifica hashes, filas, coordenadas,
grupos, etiquetas, folds y checkpoints sellados antes de ajustar.

Los parámetros de algoritmo ya fueron escogidos usando los folds internos de la
competencia de referencia; quedan fijos en todos los brazos. No interpretar la
selección interna como una estimación independiente. Se evalúan todos los brazos
en la validación interna y se selecciona por RMSE agrupado con MAE no superior al
control. La evaluación externa de cada brazo es exploratoria y se registra por
separado de la política seleccionada internamente.

Incluye densidad inversa KDE en log1p (inspirada en DenseWeight, no reproducción
exacta), pesos fuente/grupo, copias adaptativas por cuantiles, potencia 0.75,
Yeo-Johnson, corrección aditiva de sesgo con OOF espacial dentro de TRAIN,
selección estable, log-razones, imputación por litología, interpolación SMOTER
restringida y dos combinaciones. No se generan valores de litio extrapolados.
Yeo-Johnson se acota al soporte de TRAIN para evitar singularidades inversas.
Los tratamientos no cambian el flujo productivo ni M2.

Ruta: 5 > 2 > 3 > 2. Seleccionar la competencia M1 reciente con checkpoints
sellados. Alternativa: python -m lithiumscope.experiments.model_1.targeted.runner
--previous-run training_20261125_115932_m0300_826123c1

Evidencias en results/generalization/model_1_targeted: manifiesto, métricas
internas y externas, predicciones, selección, trazabilidad, intervalos pareados
por grupo y errores. Son 1350 evaluaciones base con 5 folds externos y 5 internos, más 540
ajustes previstos de calibración cruzada (1890 ajustes en total si todos completan).
Los recuentos se calculan dinámicamente. No se guardan estimadores.
Los brazos alternativos sustituyen el tratamiento anterior; el control refit y el
OOF archivado se conservan separadamente. La corrección aditiva de sesgo es una
hipótesis experimental, no una garantía de extrapolación. La evaluación general
(opción 4 del menú anterior) no ejecuta implícitamente esta campaña focalizada.
No se adquieren nuevos datos automáticamente. Incorporar nuevas muestras reales
de alta concentración exige una cohorte nueva y validación independiente.

Los folds externos se obtienen del manifiesto histórico y se comprueban contra
los checkpoints, evitando diferencias de GroupKFold entre versiones. El plan
interno archivado se usa si su hash y su cohorte coinciden; si no existe, se
reconstruye exclusivamente dentro del TRAIN externo y se advierte expresamente.
Las particiones efectivas quedan exportadas con el resultado.
