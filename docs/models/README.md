# Modelos, versiones y evidencia experimental

**Entrada vigente a la documentación de modelos.** Los documentos por fecha se conservan como bitácoras históricas; leer sus afirmaciones de «pendiente», «en ejecución» o «no publicado» en el contexto de su fecha de creación, no como estado actual del repositorio.

## Orientación actual

- [Modelos publicados](MODEL_RELEASES.md): contratos de GitHub Releases e instalación.
- [Versiones y procedencia](VERSIONING.md): diferencias entre identidad de entrenamiento, publicación y estado activo local.
- [Método del Modelo 2](MODEL_2_METHODOLOGY.md): etiquetas, features, validación y gates.
- [Flujos de predicción](PREDICTION_WORKFLOWS.md): M1, M2, demostración e integración.
- [Pipelines científicos](../architecture/SCIENTIFIC_PIPELINES.md): componentes y contratos.
- [Selección M1/M2 del 13-11-2026](M1_M2_SELECCION_Y_PDF_MENU_20261113.md): evidencia y propuesta **histórica previa a la publicación**.

## Situación de modelos de referencia (17-11-2026)

| Modelo | Ejecución | Situación |
|---|---|---|
| M1 SVM ampliado | `training_20261106_111653_m0300` | Integrado en el release público del 14-11-2026 |
| M2 CatBoost histórico (126 features) | `training_20260920_195933_m0300` | Integrado en ese release |
| M2 Random Forest v4 (170 features) | `training_20261113_200558_m0300` | Experimental; no reemplazó a CatBoost publicado |

Un equipo puede mantener activos otros modelos mediante `models/active_models.json`; el release público no cambia ese archivo automáticamente.

## Documentos experimentales: cómo encontrarlos

Los documentos con prefijo `M1_`, `M2_`, `GEOROC_`, `VNEXT_`, o fechas en su nombre describen experimentos específicos. Conservar sus resultados sin reescribir retrospectivamente la conclusión original. En particular:

- M1: [Auditoría integral](M1_AUDITORIA_INTEGRAL_DATOS_PROCEDIMIENTOS_Y_MODELOS_20261113.md), [comparación release/candidato](M1_CUATRO_FASES_DIAGNOSTICO_RELEASE_VS_CANDIDATO_20261113.md) y [defensa de métricas](M1_DEFENSA_TESIS_METRICAS_DATOS_Y_RELEASE_20261113.md).
- M2: [Protocolo canónico](M2_PROTOCOLO_CANONICO_20261112.md), [706 históricos](M2_ABLACION_706_HISTORICOS_20261113.md), [diagnóstico frente a release](M2_DIAGNOSTICO_706_VS_RELEASE_Y_HIPOTESIS_20261113.md), [competencia seis familias](M2_DIAGNOSTICO_COMPETICION_SEIS_MODELOS_20261113.md) y [v4](M2_EXPERIMENTO_FINAL_ESPECTRAL_V4_20261113.md).
- Continuidad: [cierre de CI](CI_CIERRE_Y_CONTINUIDAD_20261113.md).

## Advertencias científicas

- Comparar M1 y M2 con métricas de sus tareas: RMSE/MAE/R² frente a ROC-AUC/AP/Brier.
- No comparar cifras de folds o poblaciones distintas como si fueran pares independientes.
- El score de M2 no prueba litio en una ubicación ni viabilidad económica.
- Candidatos v4 necesitan coincidencia exacta de su esquema espectral de inferencia.
- El resultado de una simulación o de un piloto GEOROC no equivale a validación externa certificada.

Para otras auditorías, consultar el índice de [evidencia científica](../scientific/README.md).
