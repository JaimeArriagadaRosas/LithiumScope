# Etapa de mejora de entrenamiento científico - M1 y M2

## Alcance de esta etapa

Se conserva exclusivamente **Modelo 1** (regresión de concentración de Li en roca total) y **Modelo 2** (clasificación exploratoria basada en variables espectrales de Sentinel-2). No se crea Modelo 3, no se incorporan registros de salmueras/costras al entrenamiento, no se reentrenan aquí los bundles activos y no se activan automáticamente nuevos candidatos.

Esta etapa modifica la **siguiente ejecución** iniciada con `python main.py` -> **1. Entrenar modelos** -> **3. Entrenar ambos**. La preparación de datos puede requerir red y ejecutar entrenamientos largos; aquí solo se verificó el código mediante pruebas y experimentos sintéticos.

## Cambios de Modelo 1

- Se conservan validación espacial externa e interna, selección de algoritmos, optimización y transformaciones del objetivo existentes.
- Cada fold externo guarda RMSE, MAE, R², MedAE y varianza explicada, además de **sesgo medio** y **percentil 90 del error absoluto**.
- Se añade una perturbaci?n **pareada** de ruido del 5 % de la desviación típica de las características numéricas calculada **solo con los datos de entrenamiento** del fold. Se guardan el desplazamiento medio absoluto y el RMSE perturbado como diagnósticos; no se usan para seleccionar al ganador. No se simula instrumental real ni se atribuye precisión externa.
- Al terminar la competición, el ganador provisional genera diagnósticos OOF por **grupo espacial** y bootstrap por grupos (300 repeticiones, semilla 42), incluyendo sesgo, MAE, RMSE, MedAE y percentil 90 de error.
- El rango `oof_absolute_residual_q90` previamente existente es descriptivo/calibrado sobre OOF del mismo proceso de selección; **no debe describirse como intervalo con cobertura conformal validada**.

## Cambios de Modelo 2

### Corrección de fuga de información

Anteriormente, el cuantil q75 de litio se calculaba con todo el conjunto antes de construir la evaluación cruzada. Ahora:

1. Se construyen folds exteriores con `GroupKFold`, sin usar Li ni etiquetas para asignarlos. **Los grupos espaciales son obligatorios**; no existe fallback aleatorio en este protocolo.
2. Cada fold calcula q75 exclusivamente con Li de su **partición exterior de entrenamiento**. Aplica el mismo valor fijo para etiquetar su partición exterior de prueba. Un fold sin ambas clases no se declara evaluable y provoca un error explícito.
3. La optimización con Optuna utiliza folds internos agrupados y recalcula las etiquetas de cada split usando únicamente el Li del **entrenamiento interno**, no su validación. La calibración espacial de SVM también se comprueba y no recurre a particiones aleatorias.
4. El baseline de prevalencia se calcula en **exactamente los mismos folds y etiquetas** que los candidatos.
5. El modelo candidato final se ajusta con **todos los datos de entrenamiento** y el q75 completo, válido solo para esa etapa final, después de terminar la evaluación cruzada. Este umbral no interviene en las métricas exteriores.

Los folds, sus índices, tamaños, umbrales y número de grupos se almacenan en `manifests/scientific_folds.json`. Los CSV de folds incluyen `label_threshold_ppm`, `group_overlap` y tamaños; las predicciones OOF registran `outer_fold` y Li.

### Métricas y selección

Se conservan ROC-AUC, Average Precision, Balanced Accuracy, Precision, Recall, F1 y Brier. Se agregan **especificidad, tasa de falsos positivos, tasa de falsos negativos, log loss y cantidad de ejemplos de cada clase**. El ganador provisional genera errores y métricas OOF por grupo geográfico más intervalos bootstrap por grupo.

**Advertencia metodológica importante:** las etiquetas OOF se definen con un q75 que puede variar por fold. Los valores agregados permiten describir la evaluación de ese protocolo, pero no son equivalentes a un test con un único umbral absoluto de Li ni a una validación externa independiente.

El candidato y su umbral operativo se siguen seleccionando usando OOF del procedimiento; esa métrica es **optimista por selección**. El umbral es *exploratorio* hasta contar con evaluación confirmatoria. Por ello el campo `release_gate.pass` permanece `false` en esta etapa, mientras `exploratory_gate_pass` informa si se superan los mínimos comparativos del entrenamiento. No se promociona un modelo automáticamente.

## Logs y recuperación

Al entrar al comando de entrenamiento (incluyendo preboot completo y preparación de fuentes), se abre una sesión:

```text
logs/training/session_<timestamp>/
    status.json                # running / partial / failed / completed / interrupted
    runs.json                  # rutas a model_1 y model_2 si se crearon
    console.log                # copia persistente de stdout y stderr
    events.log                 # eventos Python con hora, nivel y módulo
    model_1_competition_*.log  # eventos específicos M1
    model_2_competition_*.log  # eventos específicos M2
    failure_traceback.txt      # solo si un fallo escapa del comando
```

`status.json` contiene rutas a reportes disponibles dentro de los run directories; `run.json` vincula cada ejecución al directorio de logs cuando se inicia desde el menú. Si la aplicación se interrumpe antes de abrir un run, quedan `status.json`, `console.log` y, cuando corresponda, `failure_traceback.txt`. Un cierre abrupto del proceso (apagado, kill) puede impedir escribir el estado final, pero los registros previamente vaciados permanecen.

Por modelo, en `results/model_1/runs/<id>/` o `results/model_2/runs/<id>/` se mantienen los artefactos históricos (`run.json`, `tables/`, `figures/`, `exports/`, `checkpoints/`, `manifests/`) y se añade:

```text
scientific_diagnostics/
    oof_diagnostics.csv
    per_spatial_group.csv
    scientific_diagnostics.json
```

Este informe indica explícitamente `external_validation: NOT_EVALUATED` y documenta el sesgo potencial por seleccionar al ganador con las mismas métricas OOF.

## Trazabilidad y verificación

Se actualizó la firma científica del entrenamiento para incluir los cambios en entrenamiento y diagnósticos, evitando reutilizar checkpoints incompatibles. Se conservan las políticas de reanudación y los algoritmos del proyecto.

La suite automática se amplió desde 174 a **182 pruebas**, añadiendo casos para particiones agrupadas, q75 por fold, baseline, métricas agrupadas, logs y fallos. También se comprobó el protocolo de 5 folds sobre el manifiesto histórico M2 (706 muestras, 66 grupos), y una ejecución sintética de Random Forest con 3 folds. **No se realizó un entrenamiento completo de M1/M2 en esta etapa**, de modo que no existe evidencia de una mejora numérica ya obtenida.

## Aspectos no demostrados o pendientes de una nueva ejecución

- Incremento de precisión de M1 o M2 sobre muestras externas independientes.
- Cobertura probabilística conformal de M1, calidad geológica de nuevas etiquetas y validación física de correspondencia parche–afloramiento.
- Robustez espectral o temporal mejorada por un nuevo entrenamiento; se cuenta con baterías históricas de simulación que deberán compararse sin ajustar el modelo a esos mismos escenarios.
- Capacidad predictiva en salmueras, costras o sedimentos de la plataforma SERNAGEOMIN.
- Validación confirmatoria externa y publicación automática. El nuevo candidato permanece pendiente de evaluación formal.

## Interpretación de los experimentos históricos

En la rama ya constan 90 evaluaciones M1, 1596 evaluaciones M2, robustez espacial por 36 parches, 5 folds de corrección experimental M2 y un estudio GEOROC exploratorio. El análisis GEOROC presentó divergencias entre medición por muestra y por ubicación, y no certificó etiquetas externas independientes. Estos artefactos fundamentan los controles añadidos, **no** son una medida de mayor precisión del nuevo candidato.

**Conclusión de alcance:** se mejora el procedimiento de evaluación, generación de diagnósticos y auditoría del próximo entrenamiento; las mejoras predictivas deben medirse después de ejecutarlo.