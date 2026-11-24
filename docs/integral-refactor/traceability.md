# Matriz de trazabilidad inicial

Identificadores según orden de aparición en la sección Diagnostico. Criterios de origen extraídos cuando identificables. Ninguna fila se considera verificada. La revisión íntegra y la asignación de componentes/tests siguen pendientes.

## M1-01 — Step 01 Histórico ↔ Step 03 Release — Diagnóstico de la carga del dataset
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 795.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Media. Decisión: Conservar y reforzar los contratos de entrada.**
- Criterio de aceptación (extracto): El criterio de aceptación será que las entradas admitidas produzcan la estructura esperada y que las ambiguas se rechacen o adviertan con diagnóstico reproducible.
- Componentes / corrección detallada / tests: datasets/readers/tabular.py; tests/datasets/test_tabular_contract.py
- Implementación / evidencia: Validacion de encabezados, ancho de filas, delimitador declarado, hoja y codificacion. Excepcion Mamani09 con SHA256 exacto y 4 encabezados vacios documentados; 1788 x 101 leidos. XLSX original no recuperado.
- Estado: parcialmente verificado.

## M1-02 — Step 02 Histórico ↔ Procedimiento transversal Release — Diagnóstico del análisis exploratorio (EDA)
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 801.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Conservar la infraestructura y ampliar el diagnóstico de datos.**
- Criterio de aceptación (extracto): El criterio de aceptación será poder explicar con evidencia la composición del conjunto final y sus diferencias respecto de la memoria; si el original sigue indisponible, no se afirmará una réplica exacta.
- Componentes / corrección detallada / tests: model_1/eda, core/pipeline_audit.py, core/scientific/eda_statistics.py, results/eda_artifacts.py; test_integral_eda.py.
- Implementación / evidencia: EDA descriptivo comparte preparación, perfiles por fuente, etapas y exclusiones, firmas y verificación de artefactos. Falta EDA de desarrollo por folds, censura/unidades y auditoría real.
- Estado: parcialmente verificado.

## M1-03 — Step 03 Histórico ↔ Step 04 Release — Diagnóstico de la limpieza de límites de detección
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 807.
- Diagnóstico de origen: **Estado: Defecto confirmado de cobertura. Prioridad: Alta. Decisión: Corregir la interpretación de mediciones censuradas.**
- Criterio de aceptación (extracto): El criterio de aceptación será interpretar correctamente cada formato admitido, no modificar categorías por error y cuantificar los registros reales afectados antes de decidir si corresponde reconstruir datos o repetir el entrenamiento.
- Componentes / corrección detallada / tests: datasets/validation/analytical.py; model_1/preparation/detection_limits.py; ingestion_transform.py
- Implementación / evidencia: Parser compartido, evidencia de tokens antes de coercion numerica, no se detectaron tokens censurados GEOROC en original/filtrado; pendiente convención por fuente y Mamani completo.
- Estado: parcialmente verificado.

## M1-04 — Step 04 Histórico ↔ Step 05 Release — Diagnóstico del tratamiento de datos faltantes
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 813.
- Diagnóstico de origen: **Estado: Defecto confirmado y limitación metodológica. Prioridad: Alta. Decisión: Corregir la elegibilidad de edad y controlar la imputación.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: steps/step_03_missing_values.py, pipeline.py, model_1.yaml; test_integral_physical_inputs.py.
- Implementación / evidencia: Disponibilidad de edad válida después de QC; esquema fijo antes de CV. 36 pruebas aprobadas; falta diagnóstico de imputación por fold y auditoría real.
- Estado: parcialmente verificado.

## M1-05 — Step 05 Histórico ↔ Step 07 Release — Diagnóstico del filtrado de la variable objetivo
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 819.
- Diagnóstico de origen: **Estado: Defecto confirmado de validación. Prioridad: Alta. Decisión: Corregir el contrato físico del target sin recortar extremos válidos.**
- Criterio de aceptación (extracto): El criterio de aceptación será que ningún target inválido avance al entrenamiento y que los extremos admitidos se mantengan.
- Componentes / corrección detallada / tests: datasets/validation/concentration.py, M1 filter_target, preflight y configuración; test_integral_physical_inputs.py.
- Implementación / evidencia: Rechazo de negativos/no finitos, extremos conservados, política explícita de ceros. 36 pruebas aprobadas; falta auditoría semántica de ceros/censura por fuente.
- Estado: parcialmente verificado.

## M1-06 — Step 06 Histórico ↔ Step 08 Release — Diagnóstico del control de calidad geoquímica
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 825.
- Diagnóstico de origen: **Estado: Correcto con limitación de comparabilidad química. Prioridad: Alta. Decisión: Conservar el QC y verificar la equivalencia de las sumas.**
- Criterio de aceptación (extracto): El criterio de aceptación será que el ingreso o descarte de cada muestra tenga una razón química trazable; deberá estudiarse el subconjunto de los 272 descartes de la etapa QC del release y los casos extremos antes de cambiar criterios de selección o atribuirles errores de predicción.
- Componentes / corrección detallada / tests: preparation/source_reconciliation.py; audit_m1_source_reconciliation.py; test_source_reconciliation.py
- Implementación / evidencia: Balance real: GEOROC 2744/2744 coinciden con suma de 10 oxidos; Mamani09 602/676, y 676/676 con FeO incluido (tolerancia descriptiva 0.05). Convenciones cuantificadas; falta certificar metodos/unidades y revisar equivalencia de QC. No se cambia filtro.
- Estado: parcialmente verificado.

## M1-07 — Step 07 Histórico ↔ Step 09 Release — Diagnóstico de la normalización y agrupación de categorías
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 831.
- Diagnóstico de origen: **Estado: Defecto confirmado de correspondencia semántica. Prioridad: Alta. Decisión: Corregir el clasificador de categorías y conservar las reglas comunes.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: preparation/category_cleaning.py; source_reconciliation.py; audit_m1_source_reconciliation.py
- Implementación / evidencia: 2327/2744 GEOROC preparados con Rock_type recuperado de fuente verificable; 417 ausentes o ambiguos conservados. Se crea candidato que cambia solo Rock_type, mismos grupos, target, 33 predictores y 3420 casos. Falta ablacion y revision geologica de reglas de agrupacion.
- Estado: parcialmente verificado.

## M1-08 — Step 08 Histórico ↔ Step 11 Release — Diagnóstico de la selección de variables predictoras
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 837.
- Diagnóstico de origen: **Estado: Correcto con evidencia experimental insuficiente. Prioridad: Alta. Decisión: Conservar el esquema explícito y evaluar el aporte de las variables.**
- Criterio de aceptación (extracto): El criterio de aceptación científico será demostrar, con resultados por grupo e incertidumbre, qué predictores aportan valor sin perjudicar la transferencia; no se eliminarán variables únicamente por una correlación o importancia aislada.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-09 — Step 09 Histórico ↔ Step 10 Release — Diagnóstico de la ingeniería de características
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 843.
- Diagnóstico de origen: **Estado: Correcto funcionalmente, limitado en interpretación geoquímica. Prioridad: Alta. Decisión: Preservar la referencia histórica y validar variantes científicas.**
- Criterio de aceptación (extracto): El criterio de aceptación será que las características no generen resultados engañosos y que su procedencia sea verificable; la adopción de variantes dependerá de ablaciones con splits fijos y no de mejoras obtenidas sobre TEST reutilizado.
- Componentes / corrección detallada / tests: preparation/feature_engineering.py; pipeline.py; test_geochemical_contract.py
- Implementación / evidencia: Contrato v2 rechaza operandos negativos/no finitos y denominadores <=1e-6; v1 conserva formulas. EDA real mantiene 3420 filas; 7 faltantes adicionales en derivados respecto a v1. Pendiente revision de convenciones Fe y ablaciones.
- Estado: parcialmente verificado.

## M1-10 — Step 10 Histórico ↔ Step 13 Release — Diagnóstico de la validación cruzada anidada y espacial
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 849.
- Diagnóstico de origen: **Estado: Correcto en aislamiento de grupos, metodológicamente limitado. Prioridad: Alta. Decisión: Conservar GroupKFold obligatorio y reforzar la independencia.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: core/scientific/spatial_audit.py; model_1/validation; EDA y preflight; tests/scientific
- Implementación / evidencia: Auditoria real: distancia minima 153.57 m; 24 observaciones a <=1 km de TRAIN. Grupos sin cruce no certifican independencia fisica. Falta holdout y auditoria completa de inner folds.
- Estado: parcialmente verificado.

## M1-11 — Step 11 Histórico ↔ Step 16 Release — Diagnóstico del preprocesamiento por algoritmo
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 855.
- Diagnóstico de origen: **Estado: Correcto con mejoras de cobertura. Prioridad: Alta. Decisión: Conservar pipelines ajustados dentro de cada fold y reforzar sus contratos.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-12 — Step 12 Histórico ↔ Steps 17 y 18 Release — Diagnóstico de la optimización mediante Optuna y espacios de hiperparámetros
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 861.
- Diagnóstico de origen: **Estado: Correcto operativamente, experimentalmente insuficiente. Prioridad: Alta. Decisión: Conservar el optimizador y auditar la suficiencia y reanudación de la búsqueda.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-13 — Step 13 Histórico ↔ Steps 19 y 20 Release — Diagnóstico del reentrenamiento por fold externo y early stopping de TabNet
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 867.
- Diagnóstico de origen: **Estado: Correcto para aislamiento del TEST, metodológicamente limitado en TabNet. Prioridad: Alta. Decisión: Conservar el runner y rediseñar la selección de épocas.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-14 — Step 14 Histórico ↔ Step 21 Release — Diagnóstico de la evaluación y almacenamiento de predicciones
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 873.
- Diagnóstico de origen: **Estado: Correcto con limitación de evaluación final. Prioridad: Alta. Decisión: Conservar las métricas y fortalecer la integridad y comunicación de los resultados.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: evaluation/metrics.py; evaluation/oof_evidence.py; artifacts.py; test_statistical_baselines.py
- Implementación / evidencia: Regresion valida forma/finitud, R2/EV indefinidos para verdad constante; TRAIN y TEST por fold, OOF identificado por fila/grupo/fold y Q75 TRAIN; CSV y Excel. Pruebas sinteticas y 3420 OOF archivados verificados; falta competencia real y diagnostico de confusion/robustez completos.
- Estado: parcialmente verificado.

## M1-15 — Step 15 Histórico ↔ Steps 26 y 28 Release — Diagnóstico del entrenamiento definitivo y generación de resultados
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 879.
- Diagnóstico de origen: **Estado: Correcto técnicamente, experimentalmente insuficiente para certificar el modelo final. Prioridad: Alta. Decisión: Conservar el entrenamiento definitivo y separar selección, prueba y publicación.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-16 — Step 01 Release — Diagnóstico de la adquisición e integración de fuentes
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 887.
- Diagnóstico de origen: **Estado: Metodológicamente limitado y con evidencia de duplicación residual. Prioridad: Alta. Decisión: Conservar la expansión y auditar la identidad química y física de cada observación.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: datasets/georoc_ingestion_schema.py; preparation/source_reconciliation.py; test_source_reconciliation.py; audit_m1_source_reconciliation.py
- Implementación / evidencia: Reconciliacion quimica y de convencion Fe resuelve cuatro ambiguedades de identificador: 2744/2744 preparados con ID unico; 2913/2986 brutos unicos, 73 discrepancias quimicas excluidas por pipeline historico. Se conserva procedencia; pendiente verificacion metodos/CRS, Mamani original y duplicados fisicos.
- Estado: parcialmente verificado.

## M1-17 — Step 02 Release — Diagnóstico del pipeline central de preparación
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 893.
- Diagnóstico de origen: **Estado: Correcto con mejoras de trazabilidad. Prioridad: Media. Decisión: Conservar la arquitectura y formalizar invariantes entre etapas.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-18 — Step 06 Release — Diagnóstico de la normalización geoespacial
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 899.
- Diagnóstico de origen: **Estado: Correcto estructuralmente, con riesgos de calidad espacial. Prioridad: Alta. Decisión: Conservar la canonicalización y validar conflictos de coordenadas.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-19 — Step 12 Release — Diagnóstico del preflight científico
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 905.
- Diagnóstico de origen: **Estado: Correcto como control estructural, insuficiente como puerta científica. Prioridad: Alta. Decisión: Ampliar las guardas y establecer políticas explícitas de advertencia o bloqueo.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: preflight_run_model_1_preflight.py; core/scientific_checks.py; audit_m1_source_reconciliation.py
- Implementación / evidencia: Auditoria real identifica cuatro filas de dos pares predictor+target identicos, retenidos pendientes de identidad documental; etiquetas y grupos preservados. Falta politica de replicas/duplicados y preflight que integre evidencia de fuente.
- Estado: parcialmente verificado.

## M1-20 — Step 14 Release — Diagnóstico del baseline estadístico
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 911.
- Diagnóstico de origen: **Estado: Correcto con cobertura experimental insuficiente. Prioridad: Media. Decisión: Conservar la media de TRAIN y añadir referencias complementarias.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: evaluation/baseline.py; evaluation/oof_evidence.py; audit_m1_statistical_baselines.py; test_statistical_baselines.py
- Implementación / evidencia: Media y mediana solo TRAIN; cobertura y grupos verificados, metricas fold/globales. Auditoria real 3420: RMSE medio 97.266438 vs 94.308920, MAE 27.295125 vs 19.963105; cinco algoritmos historicos superan ambos en RMSE medio. Falta evaluar candidato nuevo.
- Estado: parcialmente verificado.

## M1-21 — Step 15 Release — Diagnóstico de la competencia de algoritmos
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 917.
- Diagnóstico de origen: **Estado: Correcto operativamente, limitado como comparación de familias. Prioridad: Alta. Decisión: Conservar los seis candidatos y estandarizar las condiciones experimentales.**
- Criterio de aceptación (extracto): El criterio de aceptación será que los rankings solo agrupen resultados comparables bajo condiciones declaradas; la decisión de agregar o eliminar algoritmos dependerá de estabilidad y valor predictivo, no de que el algoritmo haya ganado una única ejecución.
- Componentes / corrección detallada / tests: training/competition.py; training/execution/cv_runner.py; core/checkpoint_identity.py; test_statistical_baselines.py
- Implementación / evidencia: Folds exactos comunes y conteo verificados para ranking, TRAIN/TEST y OOF, identidad de reanudacion incluye esquema/config/particiones/datos y versiones. Falta ejecutar competencia real de seis familias y evaluar equidad de presupuestos.
- Estado: parcialmente verificado.

## M1-22 — Step 22 Release — Diagnóstico de la selección del ganador
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 923.
- Diagnóstico de origen: **Estado: Limitación metodológica confirmada. Prioridad: Alta. Decisión: Conservar el ranking histórico y separar el ganador relativo del candidato científicamente elegible.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: evaluation/selection.py; competition_finalize_model_1_winner.py; config/model_1.yaml; test_statistical_baselines.py
- Implementación / evidencia: Regla provisional versionada exige superar ambos baselines en RMSE medio y preflight pass, con evidencia esencial; ranking relativo separado de elegibilidad de desarrollo. Test impide ajuste final si no elegible; release_gate falso sin holdout. Auditoria retrospectiva cinco de seis superan ambos. Falta limites de estabilidad/alto Li por proposito y holdout.
- Estado: parcialmente verificado.

## M1-23 — Step 23 Release — Diagnóstico de las pruebas de robustez
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 929.
- Diagnóstico de origen: **Estado: Correcto como prueba de sensibilidad, metodológicamente limitado. Prioridad: Media. Decisión: Conservar el ensayo sintético y diseñar perturbaciones coherentes con el dominio.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-24 — Step 24 Release — Diagnóstico de los diagnósticos científicos complementarios
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 935.
- Diagnóstico de origen: **Estado: Correcto retrospectivamente, con cobertura experimental insuficiente. Prioridad: Alta. Decisión: Conservar los diagnósticos y extenderlos a todos los candidatos.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M1-25 — Step 25 Release — Diagnóstico de las concentraciones extremas
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 941.
- Diagnóstico de origen: **Estado: Problema predictivo demostrado, causa experimentalmente insuficiente. Prioridad: Alta. Decisión: Investigar primero los registros de gran error y después evaluar soluciones de aprendizaje.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: scripts/scientific/audit_m1_historical_extremes.py; audit_m1_extreme_provenance.py; audit_m1_source_reconciliation.py
- Implementación / evidencia: 27240 OOF reconciliados; 2744 GEOROC con identidad unica tras comparar composicion; 16 mayores errores rastreados, nueve Macusani sin publicacion en TRAIN fold5 y max TRAIN 696 ppm. Candidato de litologia recuperada preserva cohorte; falta ablacion controlada de error alto y certificacion analitica/CRS.
- Estado: parcialmente verificado.

## M1-26 — Step 27 Release — Diagnóstico de la incertidumbre y el dominio de aplicabilidad
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 947.
- Diagnóstico de origen: **Estado: Defecto confirmado de diagnóstico y limitación de calibración. Prioridad: Alta. Decisión: Corregir la procedencia de entradas y validar el significado de las advertencias.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: predictor.py, pipeline.py, core/scientific/applicability.py; test_m1_input_provenance.py.
- Implementación / evidencia: distinguir derivadas, padding y faltantes; exigir perfil marginal completo; salida negativa/no finita marcada invalida; 91 pruebas aprobadas, 4 omitidas. Pendiente incertidumbre calibrada independiente y validacion multivariada.
- Estado: parcialmente verificado.

## M1-27 — Step 29 Release — Diagnóstico del versionamiento y publicación
- Origen: 01. Comparación M1 Historico y M1 Release.md, línea 953.
- Diagnóstico de origen: **Estado: Correcto técnicamente, con garantía científica insuficiente para promoción. Prioridad: Alta. Decisión: Conservar el versionamiento y establecer una puerta científica explícita para M1.**
- Criterio de aceptación (extracto): Consultar el texto completo de origen.
- Componentes / corrección detallada / tests: prediction/candidates/holdout/; evaluation/release_gate.py; test_prospective_holdout.py.
- Implementación / evidencia: Registro prospectivo y score/evaluacion de bundles congelados por hash; comparacion M1, bootstrap emparejado, sin activacion. Flujo sintetico probado. Falta holdout real, revision y aprobacion.
- Estado: parcialmente verificado.

## M2-01 — Step 01 Release — Diagnóstico de la configuración central
- Origen: 04. Diagnostico M2 Release.md, línea 427.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Media. Decisión: Validar y hacer efectiva la configuración.**
- Criterio de aceptación (extracto): El criterio de aceptación será que el manifiesto registre la configuración realmente utilizada y que ninguna opción científica válida sea ignorada silenciosamente.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M2-02 — Step 02 Release — Diagnóstico de la preparación de muestras geoquímicas
- Origen: 04. Diagnostico M2 Release.md, línea 433.
- Diagnóstico de origen: **Estado: Defecto confirmado. Prioridad: Alta. Decisión: Reforzar la preparación geoquímica y la identidad de las muestras.**
- Criterio de aceptación (extracto): El criterio de aceptación será que los valores físicamente inválidos se rechacen, las réplicas legítimas no se eliminen sin justificación y cada inclusión o exclusión pueda explicarse mediante una auditoría de datos por fuente.
- Componentes / corrección detallada / tests: datasets/validation/concentration.py, sample_source.py; test_integral_physical_inputs.py.
- Implementación / evidencia: Validación física compartida; 36 pruebas aprobadas; IDs estables, réplicas, conflictos y censura pendientes.
- Estado: parcialmente verificado.

## M2-03 — Step 03 Release — Diagnóstico de la construcción de grupos espaciales
- Origen: 04. Diagnostico M2 Release.md, línea 439.
- Diagnóstico de origen: **Estado: Metodológicamente limitado. Prioridad: Alta. Decisión: Validar el aislamiento espacial efectivo.**
- Criterio de aceptación (extracto): El criterio de aceptación será demostrar y registrar el aislamiento exigido por el protocolo, no solamente comprobar que los identificadores de grupos difieren.
- Componentes / corrección detallada / tests: core/scientific/spatial_audit.py; model_2/eda/pipeline.py; tests/scientific/test_spatial_audit_contract.py (daa0921).
- Implementación / evidencia: Auditoria real 706 muestras: distancia minima TRAIN elegible-TEST 1023.33 m, cero posibles solapamientos de parches de 640 m, 547 casos TEST con escena compartida, cero hashes compartidos. Falta sensibilidad a tamano y desplazamiento de celdas y dependencia geologica.
- Estado: parcialmente verificado.

## M2-04 — Step 04 Release — Diagnóstico de la adquisición de imágenes Sentinel-2
- Origen: 04. Diagnostico M2 Release.md, línea 445.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Controlar la calidad local y reproducibilidad de la adquisición.**
- Criterio de aceptación (extracto): El criterio de aceptación será obtener imágenes trazables y que superen controles de calidad local verificables, conservando un registro de las muestras que no puedan recuperarse y evitando introducir un sesgo de selección no cuantificado.
- Componentes / corrección detallada / tests: v4_integrity.py; v4_inputs.py; acquisition/sentinel2/rededge.py; import_rededge_evidence.py; test_v4_local_integrity_contract.py.
- Implementación / evidencia: 64 descriptores de escena STAC originales copiados y comprobados; hash de bandas por muestra verificado. Pendiente evaluar nueva adquisicion y temporalidad.
- Estado: parcialmente verificado.

## M2-05 — Step 05 Release — Diagnóstico de la extracción de parches satelitales
- Origen: 04. Diagnostico M2 Release.md, línea 451.
- Diagnóstico de origen: **Estado: Defecto confirmado de cobertura. Prioridad: Alta. Decisión: Garantizar compatibilidad geométrica y radiométrica de los parches.**
- Criterio de aceptación (extracto): El criterio de aceptación será reproducir una ventana físicamente consistente y comprobar las transformaciones con valores esperados.
- Componentes / corrección detallada / tests: tools/acquisition/sentinel2/patch_io.py; preparation/patch_contract.py; test_verified_patch_contract.py; test_http_tiles_rasterio_parity.py
- Implementación / evidencia: Lectura de adquisicion exige CRS metrico norte-arriba, mascara de nodata y forma; adquisicion nueva conserva hash/geometria, falla con banda totalmente invalida. Red-edge real auditado; falta comprobar coregistro fisico por banda y SCL nuevo, no modifica caches historicos.
- Estado: parcialmente verificado.

## M2-06 — Step 06 Release — Diagnóstico de la construcción del manifiesto de entrenamiento
- Origen: 04. Diagnostico M2 Release.md, línea 457.
- Diagnóstico de origen: **Estado: Defecto confirmado. Prioridad: Alta. Decisión: Reconstruir la validación y reconciliación del manifiesto.**
- Criterio de aceptación (extracto): El criterio de aceptación será que el manifiesto represente exactamente el conjunto aprobado y pueda recuperarse sin mezclar entradas obsoletas.
- Componentes / corrección detallada / tests: scripts/import_historical_data.py; datasets/validation/paths.py; test_historical_relocation.py
- Implementación / evidencia: 706 filas congeladas, 1425 archivos verificados, registro de rutas local. Falta contrato completo de nuevos manifiestos y adquisicion.
- Estado: parcialmente verificado.

## M2-07 — Step 07 Release — Diagnóstico de la carga de imágenes
- Origen: 04. Diagnostico M2 Release.md, línea 463.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Unificar los contratos de carga e inferencia.**
- Criterio de aceptación (extracto): El criterio de aceptación será que entradas equivalentes generen características equivalentes y que entradas incompatibles se rechacen con un diagnóstico comprensible, en lugar de producir silenciosamente un score sin respaldo.
- Componentes / corrección detallada / tests: preparation/patch_contract.py; prediction/spectral_contract.py; tools/acquisition/sentinel2/patch_io.py; test_verified_patch_contract.py
- Implementación / evidencia: Contrato estricto nuevo: NPY/TIFF 6x32x32, SHA, bandas ordenadas, sidecar, TIFF CRS metrico, resolucion, centro, mascaras y equivalencia de descriptores verificada. Bundle nuevo incluye contrato, legado sin el conserva ruta historica. NPY no certifica geometria independiente. Falta adquisicion y scoring real.
- Estado: parcialmente verificado.

## M2-08 — Step 08 Release — Diagnóstico de el preprocesamiento de imágenes
- Origen: 04. Diagnostico M2 Release.md, línea 469.
- Diagnóstico de origen: **Estado: Defecto confirmado. Prioridad: Alta. Decisión: Preservar la validez de píxeles y corregir la imputación.**
- Criterio de aceptación (extracto): El criterio de aceptación será que no se genere un vector aparentemente completo a partir de información insuficiente y que se pueda cuantificar la proporción de datos reales utilizados.
- Componentes / corrección detallada / tests: preparation/features/physical_v5.py; test_physical_v5.py
- Implementación / evidencia: Variante V5 sin imputacion y con mascara conjunta; registra soporte real por banda, indice y textura, NaN para ausencia total. Auditoria real 706/706, 690 elegibles con politica provisional 95%. Se preserva legado; falta integracion y ablacion controlada.
- Estado: parcialmente verificado.

## M2-09 — Step 09 Release — Diagnóstico de la extracción de características espectrales
- Origen: 04. Diagnostico M2 Release.md, línea 475.
- Diagnóstico de origen: **Estado: Defecto confirmado de contrato. Prioridad: Alta. Decisión: Validar el esquema completo y versionar el extractor.**
- Criterio de aceptación (extracto): El criterio de aceptación será preservar la reproducción del v3 y garantizar que los candidatos nuevos y sus predicciones utilicen la misma representación validada; no se modificará un modelo serializado anterior para aparentar compatibilidad.
- Componentes / corrección detallada / tests: preparation/features/physical_v5.py; preparation/patch_contract.py; physical_inputs.py; scripts/scientific/audit_m2_physical_v5.py
- Implementación / evidencia: V5 experimental 164 descriptores, 706 fuentes verificadas; contrato estricto nuevo impide 6x2x2, orden/geometry incorrectos, valida hash y equivalente TIFF/NPY. Legado sin contrato preservado. Falta ablacion V5 e inferencia con modelo real compatible.
- Estado: parcialmente verificado.

## M2-10 — Step 10 Release — Diagnóstico de las estadísticas de reflectancia
- Origen: 04. Diagnostico M2 Release.md, línea 481.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Media. Decisión: Ampliar el análisis de reflectancia y calidad.**
- Criterio de aceptación (extracto): El criterio de aceptación será reproducir los valores de referencia cuando los datos sean válidos y distinguir los resúmenes fiables de los que procedan de información incompleta.
- Componentes / corrección detallada / tests: preparation/features/physical_v5.py; scripts/scientific/audit_m2_physical_v5.py
- Implementación / evidencia: Reflectancia por escala y offset exactos, sin recorte de negativos; soporte explicito por banda e indice. Auditoria por escena y soporte, 706 muestras; falta ablacion y SCL real.
- Estado: parcialmente verificado.

## M2-11 — Step 11 Release — Diagnóstico de la extracción de textura
- Origen: 04. Diagnostico M2 Release.md, línea 487.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Media. Decisión: Evaluar textura con máscaras y escala espacial consistente.**
- Criterio de aceptación (extracto): El criterio de aceptación será calcular correctamente las diferencias válidas y reproducir el comportamiento histórico en parches completos; la contribución de la textura se medirá comparando reflectancia sola frente a reflectancia más textura con las mismas particiones, sin atribuir automáticamente los contrastes a fracturas o mineralización.
- Componentes / corrección detallada / tests: preparation/features/physical_v5.py; test_physical_v5.py
- Implementación / evidencia: Textura media y P90 solo pares adyacentes validos, soporte explicito, sin pares NaN. Tests de gradientes y mascara. Falta ablacion reflectancia vs textura y certificacion de resolucion real.
- Estado: parcialmente verificado.

## M2-12 — Step 12 Release — Diagnóstico de la ingeniería de índices espectrales
- Origen: 04. Diagnostico M2 Release.md, línea 493.
- Diagnóstico de origen: **Estado: Defecto confirmado. Prioridad: Alta. Decisión: Eliminar redundancias y tratar índices indefinidos.**
- Criterio de aceptación (extracto): El criterio de aceptación será demostrar la validez de cada fórmula y la ausencia de redundancias no intencionales; no se afirmará que reducir seis columnas mejora necesariamente ROC-AUC o AP hasta medirlo.
- Componentes / corrección detallada / tests: preparation/features/physical_v5.py; test_physical_v5.py; audit_m2_physical_v5.py
- Implementación / evidencia: Verificadas seis columnas NBR equivalentes de las 706 filas; V5 elimina duplicado, divisiones indefinidas NaN, prueba de 14 indices y mascara. Matriz 164-D separada; falta medir ablacion con mismo protocolo y preservar inferencia versionada.
- Estado: parcialmente verificado.

## M2-13 — Step 13 Release — Diagnóstico de la construcción del dataset de entrenamiento
- Origen: 04. Diagnostico M2 Release.md, línea 499.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Establecer un preflight científico para la matriz de entrenamiento.**
- Criterio de aceptación (extracto): El criterio de aceptación será que el entrenamiento solo comience con un dataset trazable y científicamente elegible, acompañado de un informe de preflight y composición final.
- Componentes / corrección detallada / tests: model_2/data/feature_cache.py; pipeline.py; core/scientific_checks.py; tests/model_2/test_verified_feature_cache.py; da55aa2, c8f653f, ab50a93.
- Implementación / evidencia: Matriz congelada v4 de 706 x 170; identidad, hashes, esquema, orden y reconstruccion de 44 descriptores comprobados; QA 690 elegibles; preflight de clases y folds internos documentado. Falta preflight integrado exhaustivo para otras cohortes y nuevas adquisiciones.
- Estado: parcialmente verificado.

## M2-14 — Step 14 Release — Diagnóstico de la caché de características espectrales
- Origen: 04. Diagnostico M2 Release.md, línea 505.
- Diagnóstico de origen: **Estado: Defecto confirmado de integridad. Prioridad: Alta. Decisión: Fortalecer la firma y reutilización de características.**
- Criterio de aceptación (extracto): El criterio de aceptación será recuperar resultados intactos cuando la firma coincida y regenerar o rechazar cualquier caché incompatible, documentando el motivo y evitando reejecuciones innecesarias.
- Componentes / corrección detallada / tests: v4_integrity.py; v4_inputs.py; acquisition/sentinel2/rededge.py; import_rededge_evidence.py; test_v4_local_integrity_contract.py.
- Implementación / evidencia: Firma v4 vincula manifiesto, matriz, QA y evidencia de cubos/sidecars y hashes de escena; re-extraccion local antes de reutilizar. Pendiente concurrencia e interrupcion.
- Estado: parcialmente verificado.

## M2-15 — Step 15 Release — Diagnóstico de la generación de etiquetas de clasificación
- Origen: 04. Diagnostico M2 Release.md, línea 511.
- Diagnóstico de origen: **Estado: Metodológicamente limitado. Prioridad: Alta. Decisión: Definir etiquetas sin dependencia del conjunto de evaluación.**
- Criterio de aceptación (extracto): El criterio de aceptación será que la definición de etiquetas sea reproducible y no dependa de información de evaluación no disponible durante el desarrollo.
- Componentes / corrección detallada / tests: training/optimization/optimizer.py; training/calibration/operating.py
- Implementación / evidencia: Cuantil Optuna calculado solo en cada TRAIN interno; corte geoquimico fijo de la particion exterior para seleccionar politica operativa sobre sus predicciones internas. Rechaza folds monoclase antes de ajustar. Falta holdout prospectivo y comparacion controlada.
- Estado: parcialmente verificado.

## M2-16 — Step 16 Release — Diagnóstico de la validación cruzada espacial estratificada
- Origen: 04. Diagnostico M2 Release.md, línea 517.
- Diagnóstico de origen: **Estado: Defecto confirmado y limitación metodológica. Prioridad: Alta. Decisión: Exigir validación espacial en todos los niveles.**
- Criterio de aceptación (extracto): El criterio de aceptación será que ningún estimador pueda comunicar resultados de validación espacial sin satisfacer las condiciones verificadas del protocolo; los experimentos aleatorios podrán mantenerse como controles diferenciados, no como equivalentes científicos.
- Componentes / corrección detallada / tests: core/scientific/spatial_audit.py; model_2/validation; EDA y preparacion cientifica
- Implementación / evidencia: Auditoria real: minimo 1023.33 m, 547/706 TEST con escena compartida, 0 hashes compartidos y 0 solapamientos potenciales de parches de 640 m segun distancia. Falta protocolo independiente por escena/sitio.
- Estado: parcialmente verificado.

## M2-17 — Step 17 Release — Diagnóstico de el baseline de clasificación
- Origen: 04. Diagnostico M2 Release.md, línea 523.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Media. Decisión: Consolidar baselines y comparaciones reproducibles.**
- Criterio de aceptación (extracto): El criterio de aceptación será disponer de referencias reproducibles que permitan interpretar tanto discriminación como decisiones operativas sin favorecer artificialmente a un candidato.
- Componentes / corrección detallada / tests: validation/spatial.py; competition_prepare_scientific_manifests.py; test_train_only_operating_policy.py
- Implementación / evidencia: Baseline prior utiliza las mismas filas TRAIN elegibles por QA que candidatos; TEST intacto, OOF identificado, soporte y metricas globales. Pendiente baseline espacial complementario.
- Estado: parcialmente verificado.

## M2-18 — Step 18 Release — Diagnóstico de la competencia de algoritmos
- Origen: 04. Diagnostico M2 Release.md, línea 529.
- Diagnóstico de origen: **Estado: Defecto confirmado de control de estados. Prioridad: Alta. Decisión: Separar la validez del entrenamiento de la exportación y ampliar la competencia controlada.**
- Criterio de aceptación (extracto): El criterio de aceptación será distinguir candidatos válidos, descartados y errores auxiliares sin contradicciones, conservando la comparación completa.
- Componentes / corrección detallada / tests: core/candidate_evidence.py; training/execution/competition_evaluate_algorithm_candidates.py; tests/runtime/test_candidate_evidence.py (ab50a93).
- Implementación / evidencia: Una fila por candidato; evidencia esencial separada de figuras auxiliares; ranking excluye evidencias numericas incompletas. Pruebas focalizadas aprobadas. Falta comparar pesos de clases y presupuestos con experimentos controlados.
- Estado: parcialmente verificado.

## M2-19 — Step 19 Release — Diagnóstico de la optimización mediante Optuna
- Origen: 04. Diagnostico M2 Release.md, línea 535.
- Diagnóstico de origen: **Estado: Experimentalmente insuficiente. Prioridad: Alta. Decisión: Reforzar Optuna y comprobar suficiencia de la búsqueda.**
- Criterio de aceptación (extracto): El criterio de aceptación será una optimización auditable, reproducible y con evidencia de suficiencia o limitación del presupuesto; los máximos de 312 trials y 936 ajustes internos no se comunicarán como ejecuciones efectivamente completadas.
- Componentes / corrección detallada / tests: training/optimization/optimizer.py; core/run_resume.py
- Implementación / evidencia: Preflight valida cobertura y ambas clases en todos los folds antes de Optuna; no omite folds monoclase. Cuenta fallidos/podados/completos. Falta estudio de convergencia y presupuesto con datos reales.
- Estado: parcialmente verificado.

## M2-20 — Step 20 Release — Diagnóstico de la calibración de SVM
- Origen: 04. Diagnostico M2 Release.md, línea 541.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Evaluar calibración de todos los candidatos bajo aislamiento espacial.**
- Criterio de aceptación (extracto): El criterio de aceptación será justificar con evidencia la política de calibración de cada candidato y conservar la distinción entre score exploratorio y probabilidad calibrada de la etiqueta binaria.
- Componentes / corrección detallada / tests: training/calibration/operating.py; training/calibration/policy.py; test_train_only_operating_policy.py
- Implementación / evidencia: Politica operativa cross-fit de estimadores calibrados o legacy dentro de desarrollo, misma construccion para final. Preflight real viable; 18 particiones internas, minimo en calibracion anidada 17 positivos y 62 negativos. Falta comparacion empirica de calibracion por algoritmo; se conserva legacy por defecto.
- Estado: parcialmente verificado.

## M2-21 — Step 21 Release — Diagnóstico de la ejecución del entrenamiento
- Origen: 04. Diagnostico M2 Release.md, línea 547.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Fortalecer el runner, los checkpoints y los diagnósticos de aprendizaje.**
- Criterio de aceptación (extracto): El criterio de aceptación será reproducir la ejecución sin mezclar estados incompatibles y disponer de evidencias suficientes para estudiar las causas de las diferencias de rendimiento.
- Componentes / corrección detallada / tests: core/checkpoints.py; core/scientific_checks.py; cv_runner.py; tests/runtime/test_checkpoints.py
- Implementación / evidencia: Checkpoint versionado con SHA256 y escritura atomica; rechaza alteracion, indices invalidos, predicciones incompletas/no finitas. Metricas principales recalculadas al reanudar. OOF cubre cada observacion exactamente una vez. Pendiente curvas de aprendizaje y comparacion real.
- Estado: parcialmente verificado.

## M2-22 — Step 22 Release — Diagnóstico de el cálculo de métricas de clasificación
- Origen: 04. Diagnostico M2 Release.md, línea 553.
- Diagnóstico de origen: **Estado: Defecto confirmado. Prioridad: Alta. Decisión: Validar las métricas y completar la evaluación binaria.**
- Criterio de aceptación (extracto): El criterio de aceptación será que cada métrica sea reproducible desde predicciones identificadas, que las agregaciones indiquen su denominador y que las evaluaciones incompletas no sean declaradas superiores.
- Componentes / corrección detallada / tests: evaluation/metrics.py, evaluation/thresholds.py, test_binary_metrics_contract.py.
- Implementación / evidencia: metricas binarias validadas, monoclase no comparable; confusion, accuracy, sensibilidad, MCC/Kappa, ECE y fiabilidad. 100 pruebas aprobadas, 3 omitidas. Falta reconstruccion de metricas historicas y agregaciones OOF etiquetadas.
- Estado: parcialmente verificado.

## M2-23 — Step 23 Release — Diagnóstico de los resultados de la competencia histórica
- Origen: 04. Diagnostico M2 Release.md, línea 559.
- Diagnóstico de origen: **Estado: Información insuficiente y evidencia experimental limitada. Prioridad: Alta. Decisión: Recuperar evidencias y establecer una comparación científica controlada.**
- Criterio de aceptación (extracto): El criterio de aceptación será poder fundamentar mejoras con evidencia comparable y cuantificar su incertidumbre.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M2-24 — Step 24 Release — Diagnóstico de la selección del ganador
- Origen: 04. Diagnostico M2 Release.md, línea 565.
- Diagnóstico de origen: **Estado: Metodológicamente limitado. Prioridad: Alta. Decisión: Separar la elección del ganador de su evaluación independiente.**
- Criterio de aceptación (extracto): El criterio de aceptación será distinguir un ganador interno de un modelo cuya superioridad y generalización han sido evaluadas independientemente; no se garantizará que el algoritmo final deba seguir siendo CatBoost.
- Componentes / corrección detallada / tests: core/candidate_evidence.py; competition_evaluate_algorithm_candidates.py; competition.py
- Implementación / evidencia: Ranking filtra metricas indefinidas y evidencia esencial fallida. Figura fallida no invalida evaluacion numerica. Falta holdout no utilizado para seleccionar.
- Estado: parcialmente verificado.

## M2-25 — Step 25 Release — Diagnóstico de la optimización del umbral operativo
- Origen: 04. Diagnostico M2 Release.md, línea 571.
- Diagnóstico de origen: **Estado: Metodológicamente limitado. Prioridad: Alta. Decisión: Optimizar y evaluar el umbral en conjuntos separados.**
- Criterio de aceptación (extracto): El criterio de aceptación será comparar BA y demás métricas a 0,5 y al corte optimizado sobre las mismas observaciones independientes, sin presentar el 0,5818 OOF retrospectivo como validación independiente del umbral.
- Componentes / corrección detallada / tests: training/calibration/operating.py; training/execution/cv_runner*.py; competition_finalize_scientific_winner.py; tests/model_2/test_train_only_operating_policy.py
- Implementación / evidencia: Umbral operativo por fold ajustado exclusivamente con cross-fits internos de TRAIN y evaluado sin reajuste sobre TEST, exporta scores internos, umbral y metricas de ambas politicas. Corte final procede de OOF de desarrollo al corte Li final; no certifica holdout. Pruebas de aislamiento, reanudacion y objetivo. Falta validacion real y externa.
- Estado: parcialmente verificado.

## M2-26 — Step 26 Release — Diagnóstico de el Release Gate
- Origen: 04. Diagnostico M2 Release.md, línea 577.
- Diagnóstico de origen: **Estado: Correcto con mejoras de control. Prioridad: Alta. Decisión: Separar entrenamiento, elegibilidad y promoción del release.**
- Criterio de aceptación (extracto): El criterio de aceptación será que ningún candidato se active o presente como aprobado por el solo hecho de haberse entrenado o guardado; los criterios de publicación externos deberán verificarse por separado.
- Componentes / corrección detallada / tests: prediction/candidates/holdout/; evaluation/release_gate.py; test_prospective_holdout.py.
- Implementación / evidencia: Registro prospectivo, endpoint comun, score local, revelacion comprometida y metricas/uncertainty emparejadas. Gate valida dominios y soporte; revision documental obligatoria; promocion bloqueada. Flujo sintetico M2 v3 aprobado. Sin holdout real.
- Estado: parcialmente verificado.

## M2-27 — Step 27 Release — Diagnóstico de el entrenamiento definitivo
- Origen: 04. Diagnostico M2 Release.md, línea 583.
- Diagnóstico de origen: **Estado: Correcto con mejoras. Prioridad: Alta. Decisión: Validar el ajuste definitivo y su contrato operativo.**
- Criterio de aceptación (extracto): El criterio de aceptación será disponer de un modelo reproducible, con contrato completo e indicadores de rendimiento independiente del ajuste definitivo; el release anterior permanecerá disponible hasta que un candidato corregido supere la evaluación y la promoción autorizada.
- Componentes / corrección detallada / tests: pendientes de revisión y contraste con código.
- Implementación / evidencia: ninguna.
- Estado: pendiente.

## M2-28 — Step 28 Release — Diagnóstico de el perfil de aplicabilidad
- Origen: 04. Diagnostico M2 Release.md, línea 589.
- Diagnóstico de origen: **Estado: Defecto confirmado. Prioridad: Alta. Decisión: Corregir el perfil de aplicabilidad y gestionar incertidumbre de inferencia.**
- Criterio de aceptación (extracto): El criterio de aceptación será informar correctamente el grado de respaldo de una predicción y rechazar o advertir entradas incompatibles, sin presentar los percentiles marginales como intervalos de predicción.
- Componentes / corrección detallada / tests: prediction/spectral_contract.py, predictor.py y evaluador integrado; test_spectral_inference_contract.py.
- Implementación / evidencia: Abstención ante esquema o características faltantes; perfil ausente/incompleto marcado insuficiente; M2 v4 requiere metadatos y bandas del mismo scene. 66 aprobadas, 4 omitidas. Falta prueba real de adquisición/inferencia v4 y validación independiente de perfiles.
- Estado: parcialmente verificado.

## M2-29 — Step 29 Release — Diagnóstico de la generación de resultados gráficos
- Origen: 04. Diagnostico M2 Release.md, línea 595.
- Diagnóstico de origen: **Estado: Correcto con mejoras de trazabilidad. Prioridad: Media. Decisión: Mejorar las representaciones diagnósticas y sus contratos.**
- Criterio de aceptación (extracto): El criterio de aceptación será que cada figura pueda reproducirse a partir de sus tablas y que no confunda un mapa de puntos conocidos con una superficie de prospectividad continua ni una simulación ilustrativa con un resultado real.
- Componentes / corrección detallada / tests: evaluation/oof_evidence.py; evaluation/maps.py; evaluation/plots.py; test_candidate_evidence.py
- Implementación / evidencia: OOF registra ID, posicion, fold, etiqueta, corte Li, score, clase a 0.5 y elegibilidad TRAIN; mapa por ID con escala 0-1; histogramas con bins comunes; tabla de fiabilidad. Falta verificar figuras de entrenamiento real.
- Estado: parcialmente verificado.

## M2-30 — Step 30 Release — Diagnóstico de el almacenamiento y trazabilidad
- Origen: 04. Diagnostico M2 Release.md, línea 601.
- Diagnóstico de origen: **Estado: Correcto con mejoras de integridad. Prioridad: Alta. Decisión: Consolidar artefactos, evidencias y versionamiento reproducible.**
- Criterio de aceptación (extracto): El criterio de aceptación será poder reconstruir cada decisión de entrenamiento y predicción sin depender del orden de filas, recuperar evidencias desde el menú de resultados y mantener inalterado el release histórico.
- Componentes / corrección detallada / tests: core/checkpoints.py; core/checkpoint_identity.py; core/run_resume.py; evaluation/oof_evidence.py; test_checkpoints.py
- Implementación / evidencia: Checkpoints con sello y vinculacion a X/y/grupos/folds/config/versiones/QA/endpoint, nombres de Optuna por identidad; lectores historicos solo lectura. Pruebas M1/M2 sinteticas de reanudacion e incompatibilidad. Falta reanudacion real, integridad completa de runs y bundles.
- Estado: parcialmente verificado.
