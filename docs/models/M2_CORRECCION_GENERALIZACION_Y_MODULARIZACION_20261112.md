# LithiumScope — corrección de generalización y cierre de modularización

Fecha: 12 de noviembre de 2026. Rama de trabajo: `main`, exclusivamente local.

## Alcance y restricciones

Se mantiene íntegra la operación original del menú interactivo y del generador PDF. No se modifica `models/active_models.json`, no se reemplazan bundles existentes, no se borran muestras históricas ni checkpoints y no se publica un release. Las instalaciones y salidas de pruebas permanecen en D:. La demostración de diez casos continúa siendo exploratoria, no un conjunto externo independiente.

## Diagnóstico de sobreajuste, no de subajuste

El candidato Random Forest `training_20261112_123451_m0300` utiliza 126 características derivadas de seis bandas Sentinel-2, con 783 muestras y 767 elegibles para ajuste bajo QA95. En entrenamiento aparente alcanzó ROC-AUC=0,990865; sobre las mismas filas con predicciones OOF espaciales, ROC-AUC=0,644973. La brecha aproximada de 0,346 apunta a sobreajuste considerable. Esta comparación se hizo entre estimadores entrenados con poblaciones diferentes y la etiqueta original era específica de cada fold; no debe interpretarse como un estimador pareado exacto del error de generalización.

El estudio pareado adicional de cinco folds espaciales, sin tocar las 783 observaciones de evaluación, comparó el RF original frente a profundidad 8 y mínimo 10 muestras por hoja:

| Métrica OOF | Original | Regularizado |
|---|---:|---:|
| ROC-AUC | 0,649555 | 0,663126 |
| Average Precision | 0,381100 | 0,396195 |
| Brier sin calibrar (menor es mejor) | 0,203806 | 0,216693 |
| Brier calibración sigmoidal interna | 0,179843 | 0,178853 |
| Brecha AUC aparente media por fold | 0,366406 | 0,307031 |

Con bootstrap pareado por grupos geográficos (400 repeticiones), la mejora de AUC del RF regularizado es +0,0136, intervalo del 95 % de −0,0036 a +0,0322, que incluye cero. AP también tiene un intervalo que incluye cero. Por tanto, **la regularización es prometedora pero no se ha demostrado superioridad concluyente**.

## Nueva búsqueda experimental versionada

Se conserva intacto `random_forest.optuna_space` (perfil `legacy`). Se añadió `random_forest.optuna_space_regularized` y una política de selección explícita `search_policy.py`. En `config/model_2.yaml`:

- `optimization.random_forest_search_profile: regularized_v2`.
- `optimization.per_algorithm_trials.random_forest: 16`.
- Profundidad permitida: 6, 8, 10 o 12.
- Muestras mínimas por hoja: 6–16.
- Cantidad de árboles: 250–650.
- Subconjunto de variables por partición: `sqrt`, `log2` o 0,5.

La búsqueda contiene la configuración experimental de profundidad 8/hoja 10 sin imponerla como ganadora. Continúan los mismos seis algoritmos. El YAML de Modelo 2 participa en la firma de entrenamiento; este cambio configura un **nuevo experimento reproducible**, sin reutilizar los estudios Optuna históricos ni activar modelos automáticamente. El identificador de perfil se registra en los metadatos de futuros candidatos.

## Calidad e independencia

Al auditar los diez casos de demostración frente a GEOROC se identificaron **dos posibles duplicados**: A1015-22 y A1015-25, con concentraciones de litio coincidentes y distancias geográficas de pocos metros. Falta confirmación documental de identidad. Por ello, esa demostración no debe considerarse holdout independiente.

Experimentos de eliminación exclusiva del conjunto TRAIN, con las 783 muestras TEST conservadas:

| Datos de entrenamiento | ROC-AUC OOF |
|---|---:|
| QA95 completo, 767 elegibles | 0,644785 |
| Sin 77 GEOROC provisionales, 690 elegibles | 0,639168 |
| Sin sitios con etiquetas contradictorias, 693 elegibles | 0,642543 |

Ninguna eliminación general mejoró la discriminación espacial global. No se borraron ni reescribieron datos fuente. Se requieren trazabilidad de publicaciones, verificación de etiquetas físicas y un conjunto externo previamente congelado.

El `scientific_release_gate_current.json` de la carpeta de evidencias registra una decisión negativa: el conjunto de diez casos tiene solo dos positivos, un área geográfica pequeña, duplicados probables y evaluación ya consultada durante el desarrollo. El candidato continúa experimental, sin autorización de activación ni de release.

## Arquitectura y pruebas

Se descompusieron por responsabilidad las competencias de M1/M2, los bucles de validación cruzada, la adquisición de datos M2, Sentinel-2 y el empaquetado de releases. Las nuevas unidades mantienen las firmas públicas y puntos de entrada. Se conservan las secciones y lógica originales del PDF; las pruebas de regresión verifican que siga siendo PDF, no HTML.

**Resultado de la auditoría:** cero archivos propios por encima de 300 líneas. Se retiraron todas las excepciones de `config/code_size_legacy_exceptions.json`, ahora `{}`, y el workflow existente comprueba que no vuelvan a aparecer.

Pruebas completas sobre la última implementación: **352 aprobadas, cero fallos y 15 advertencias de bibliotecas/etiquetas únicas**. Ruff y control de tamaño: correctos. Prueba real pequeña de validación cruzada RF agrupada: aprobada. Preflight científico M2: hashes/contratos correctos, 706 muestras históricas preservadas, cinco folds, sin entrenar ni activar candidatos.

Los artefactos exploratorios se mantienen en:

`results/scientific_simulations/m2_generalization_20261112/`

Las pruebas de refactorización y el criterio de release se encuentran en:

`results/scientific_simulations/m2_generalization_20261112/refactor_completion/`

## Estado de cierre

La **implementación técnica** del perfil regularizado y la **modularización** pueden considerarse realizadas tras validar pruebas. El modelo nuevo **no está aprobado científicamente para sustituir el CatBoost anterior**. Queda pendiente una evaluación confirmatoria externa independiente con procedencia verificada, representación regional y suficientes positivos. Solo entonces sería legítimo seleccionar y autorizar un release.

No se realizó push ni modificación de `main` remoto. Los cambios se mantienen en el clon local.


## Verificación complementaria del optimizador real

Se ejecutó una búsqueda Optuna pequeña con dos trials sobre datos sintéticos de seis bandas, usando tres folds separados por grupos espaciales y el perfil `regularized_v2`. Terminó sin errores y seleccionó una configuración válida dentro de los límites establecidos (533 árboles, profundidad 8, 8 muestras por hoja y `max_features=0.5`). Evidencia: `refactor_completion/optuna_regularized_smoke/validated_parameters.json`. **Es exclusivamente una prueba funcional del optimizador, no una nueva evaluación científica del Modelo 2**.

El mismo directorio conserva `scientific_release_gate_current.json` con razones explícitas del bloqueo de release y el XML de las 352 pruebas de regresión.
