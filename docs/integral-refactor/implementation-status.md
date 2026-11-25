# Estado de implementación por criterio

Corte de código: 0c1ad5ce3c3d412bea80f20f32374bca2a6663ad. Conteo de implementación solicitado, **independiente de la ejecución de pruebas**. I = implementación principal incorporada al código (no necesariamente todos los detalles ni aceptación); P = implementación parcial con trabajo de código concreto pendiente. Cada criterio vale una unidad, incluidos requisitos superpuestos; P no suma como implementado. La clasificación es revisión documental y estática, no resultado de una suite. Los originales completos permanecen locales; se usan los extractos y matrices versionados. Consola adicional excluida del denominador textual de 166.

| Diagnóstico | Total | Implementación principal | Parcial | Avance de implementación |
|---|---:|---:|---:|---:|
| M1 | 27 | 27 | 0 | 100,0 % |
| M2 | 30 | 30 | 0 | 100,0 % |
| ARQ | 109 | 81 | 28 | 74,3 % |
| **Total** | **166** | **138** | **28** | **83,1 %** |

Los seis cierres acotados y la última suite histórica se conservan por separado en HANDOFF.md. Los experimentos son ejecutables pero no ejecutados; el rendimiento no está demostrado. La revisión podrá reclasificar criterios si surgen faltantes.

| ID | Implementación | Evidencia principal o trabajo faltante |
|---|---|---|
| M1-01 | Principal incorporada | Lectura tabular estricta y codificaciones. |
| M1-02 | Principal incorporada | EDA común, perfiles antes/después, manifiestos. |
| M1-03 | Principal incorporada | Contrato de censura y detección. |
| M1-04 | Principal incorporada | Imputación y transformaciones por fold. |
| M1-05 | Principal incorporada | Validación de objetivo y límites físicos. |
| M1-06 | Principal incorporada | Control de calidad y reconciliación de fuentes. |
| M1-07 | Principal incorporada | Normalización categórica y comparación de variantes. |
| M1-08 | Principal incorporada | Selección de predictores y ejecutor de ablación. |
| M1-09 | Principal incorporada | Ingeniería versionada y comparación de variables. |
| M1-10 | Principal incorporada | Particiones espaciales anidadas y auditoría de distancias. |
| M1-11 | Principal incorporada | Preprocesamiento por algoritmo y comparación controlada. |
| M1-12 | Principal incorporada | Contrato Optuna y comparación 20/100 trials. |
| M1-13 | Principal incorporada | Adaptador TabNet con validación espacial. |
| M1-14 | Principal incorporada | Métricas, predicciones OOF y diagnósticos por grupo. |
| M1-15 | Principal incorporada | Contrato de ajuste final, separado de promoción. |
| M1-16 | Principal incorporada | Adquisición y armonización; fuentes históricas protegidas. |
| M1-17 | Principal incorporada | Pipeline común de preparación y auditoría. |
| M1-18 | Principal incorporada | Validación geoespacial y reconciliación de coordenadas. |
| M1-19 | Principal incorporada | Registro documental de conflictos, identidad de fila y hashes de documentos, decisiones revisadas sin automatizar interpretación geológica y comparación pareada de exclusiones SOLO TRAIN. Evidencia documental real y ejecución pendientes. |
| M1-20 | Principal incorporada | Baselines estadísticos independientes. |
| M1-21 | Principal incorporada | Competencia de seis familias. |
| M1-22 | Principal incorporada | Política versionada de elegibilidad de desarrollo con baseline pareado, fracción de folds ganados, peor fold, error/sesgo de Li alto, sesgo por fuente y validez física; mantiene ranking relativo y gate externo. |
| M1-23 | Principal incorporada | Diagnósticos de robustez y comparación emparejada. |
| M1-24 | Principal incorporada | Curvas, errores y diagnósticos científicos. |
| M1-25 | Principal incorporada | Auditoría específica de concentraciones extremas. |
| M1-26 | Principal incorporada | Perfil multivariado de covarianza regularizada con cobertura de entrada y categorías inéditas; estudio de intervalos por grupos separados para ajuste/calibración/evaluación, preflight por defecto; P90 OOF etiquetado descriptivo. |
| M1-27 | Principal incorporada | Persistencia, contratos y promoción explícita. |
| M2-01 | Principal incorporada | Configuración validada y versionada. |
| M2-02 | Principal incorporada | Reconciliación por fuente, identidad original y réplica explícita; cuarentena de conflictos, censura excluida sin imputación, unidades y auditoría de inclusión. Cohorte congelada intacta. |
| M2-03 | Principal incorporada | Seis protocolos predeclarados: rejillas, componentes por escena y embargo TRAIN, con preflight, cobertura y comparación por algoritmo. |
| M2-04 | Principal incorporada | Recuperación SCL histórica por escena/posición/hash, máscaras preservadas, adquisición temporal escalonada con ventanas preregistradas y comparación de escena única frente a mediana BOA en cohorte y folds idénticos. Adquisición y evidencia real pendientes. |
| M2-05 | Principal incorporada | Adquisición nueva en cuadrícula métrica común de 640m/32px, cobertura completa, CRS, alineación nativa, remuestreo y SCL con hash y trazabilidad. |
| M2-06 | Principal incorporada | Manifiestos nuevos con firma fuente/configuración, verificación de patch y SCL, reconciliación de etiquetas, retiro de entradas obsoletas, historial de fallos y protección de cohorte congelada. |
| M2-07 | Principal incorporada | Carga de imágenes y rechazo de entradas inválidas. |
| M2-08 | Principal incorporada | Preprocesamiento físico v5 aislado, con comparación de representaciones. |
| M2-09 | Principal incorporada | Extractores versionados v4/v5 y comparación emparejada. |
| M2-10 | Principal incorporada | Estadísticas físicas de reflectancia. |
| M2-11 | Principal incorporada | Ablación pareada reflectancia 60 vs reflectancia+textura 80, con QA y versión v4/v5 explícitas. |
| M2-12 | Principal incorporada | Índices versionados, divisiones inválidas y redundancia. |
| M2-13 | Principal incorporada | Construcción de matriz con contrato de esquema. |
| M2-14 | Principal incorporada | Caché con identidad y comprobaciones de compatibilidad. |
| M2-15 | Principal incorporada | Generación de etiquetas separada de umbral operativo. |
| M2-16 | Principal incorporada | Validación espacial anidada con soporte de clases. |
| M2-17 | Principal incorporada | Baseline espacial geodésico suplementario con mismos folds, QA TRAIN, empates y OOF; no participa en ranking. |
| M2-18 | Principal incorporada | Competencia de seis familias. |
| M2-19 | Principal incorporada | Presupuesto y optimización Optuna controlados. |
| M2-20 | Principal incorporada | Calibración anidada y comparación por familia. |
| M2-21 | Principal incorporada | Ejecución, recuperación e identidad de checkpoints. |
| M2-22 | Principal incorporada | Métricas binarias y probabilísticas. |
| M2-23 | Principal incorporada | Auditoría de competencia histórica. |
| M2-24 | Principal incorporada | Ranking y elegibilidad separados. |
| M2-25 | Principal incorporada | Optimización del umbral en desarrollo interno. |
| M2-26 | Principal incorporada | Gate de elegibilidad independiente. |
| M2-27 | Principal incorporada | Contrato de entrenamiento definitivo, sin promoción implícita. |
| M2-28 | Principal incorporada | Perfil de aplicabilidad y abstención. |
| M2-29 | Principal incorporada | Visualizaciones y evidencias de evaluación. |
| M2-30 | Principal incorporada | Almacenamiento y trazabilidad por ejecución. |
| ARQ-001 | Principal incorporada | Inventario dinámico de métricas y resultados. |
| ARQ-002 | Principal incorporada | Submenú de consulta de evidencias. |
| ARQ-003 | Parcial | Laboratorio incorporado; falta completar casos de adquisición automática autorizada. |
| ARQ-004 | Principal incorporada | Cancelación previa conserva limpieza de sesión. |
| ARQ-005 | Principal incorporada | Submenú de predicción reorganizado. |
| ARQ-006 | Principal incorporada | Reubicación inicial conservada; cambios científicos posteriores separados. |
| ARQ-007 | Parcial | 27 diagnósticos con código principal, directorios específicos de diagnóstico, preparación y evaluación; quedan ajustes transversales e integración de pipeline. |
| ARQ-008 | Parcial | Adquisición y procesamiento separados; falta terminar centralización de consultas Mamani09/GEOROC. |
| ARQ-009 | Principal incorporada | Preparación científica centralizada. |
| ARQ-010 | Principal incorporada | EDA con statistics, visualization y reporting. |
| ARQ-011 | Principal incorporada | Reutilización verificable de EDA durante entrenamiento. |
| ARQ-012 | Principal incorporada | Validación y controles de particiones. |
| ARQ-013 | Principal incorporada | Algoritmos, preprocesamiento, optimización y ejecución separados. |
| ARQ-014 | Principal incorporada | Evaluación, OOF, métricas, baselines y diagnósticos de errores. |
| ARQ-015 | Principal incorporada | Directorio diagnostics M1 con dominio, incertidumbre, concentraciones altas, robustez, métricas históricas y entradas de curvas; conserva compatibilidad de rutas antiguas y procedimientos compartidos. |
| ARQ-016 | Parcial | Comparaciones históricas y nuevos ejecutores; falta completar consolidación de metodología. |
| ARQ-017 | Principal incorporada | Entrada, procedencia de predictores, inferencia y aplicabilidad. |
| ARQ-018 | Parcial | Flujo interactivo operativo; falta culminar orquestación de adquisición y validación independiente en pipeline. |
| ARQ-019 | Principal incorporada | Resultados EDA, runs y metodología diferenciados. |
| ARQ-020 | Principal incorporada | Infraestructura compartida y compatibilidad histórica. |
| ARQ-021 | Parcial | Requisito global: depende de restantes diagnósticos M1. |
| ARQ-022 | Parcial | 30 diagnósticos con código principal, diagnóstico separado y adquisición registrada; quedan preprocessing, integración y flujos residuales. |
| ARQ-023 | Principal incorporada | Ruta científica de adquisición nueva registrada con SCL, corregistro y radiometría por asset; sin cambiar adquisición histórica. |
| ARQ-024 | Principal incorporada | Preparación con contrato común de identidad, censura, réplicas, unidades y reconciliación de checkpoints. |
| ARQ-025 | Principal incorporada | Extractores espectrales versionados y variantes físicas. |
| ARQ-026 | Principal incorporada | Compatibilidad de caché, manifiestos y características. |
| ARQ-027 | Principal incorporada | EDA con estadísticas, visualización y exportación separadas. |
| ARQ-028 | Principal incorporada | Reutilización de preparación y EDA compatible. |
| ARQ-029 | Principal incorporada | Preflight y validación espacial anidada. |
| ARQ-030 | Parcial | Familias y ejecución separadas; falta consolidar preprocessing específico en M2. |
| ARQ-031 | Principal incorporada | Calibración interna y comparación por familia. |
| ARQ-032 | Principal incorporada | Umbral geoquímico separado de umbral operativo. |
| ARQ-033 | Principal incorporada | Métricas binarias/probabilísticas y OOF. |
| ARQ-034 | Principal incorporada | Selección y elegibilidad científica separadas. |
| ARQ-035 | Principal incorporada | Directorio diagnostics M2 con auditoría de competencia, sobreajuste espectral, dominio, independencia, calidad y estabilidad, incertidumbre y entradas de curvas; rutas anteriores compatibles. |
| ARQ-036 | Principal incorporada | Estudios por representación, textura, presupuesto, calibración y sensibilidad de bloques, escenas y buffers con evaluación opt-in. |
| ARQ-037 | Principal incorporada | Inferencia versionada y abstención por perfil inválido. |
| ARQ-038 | Parcial | Flujo interactivo existente; faltan piezas de adquisición, manifiestos e independencia prospectiva. |
| ARQ-039 | Principal incorporada | Resultados EDA, runs y metodología separados. |
| ARQ-040 | Principal incorporada | Componentes comunes y compatibilidad. |
| ARQ-041 | Parcial | Requisito global: depende de restantes diagnósticos M2. |
| ARQ-042 | Parcial | Cálculos de casos y deltas trasladados y preparados fuera del renderer; falta retirar o delimitar compatibilidad histórica de helpers de formato y verificar límite completo. |
| ARQ-043 | Principal incorporada | Reporting movido a raíz. |
| ARQ-044 | Principal incorporada | Predicción centrada en entradas, cálculos y resultados. |
| ARQ-045 | Parcial | Evaluación y holdout separados, scoring de candidatos en ejecución; restantes componentes de prediction pendientes de auditoría completa. |
| ARQ-046 | Principal incorporada | Subdirectorios principales de predicción. |
| ARQ-047 | Principal incorporada | Preparación de errores absolutos, clases operativas, intervalos físicos y deltas históricos en prediction/analysis/report_inputs.py; renderer consume valores calculados, conserva presentación. |
| ARQ-048 | Principal incorporada | Evaluación comparativa de candidatos y holdout prospectivo trasladados a evaluation/candidates y evaluation/holdout; persistencia conserva activación y distribución empaquetado, scoring de casos en prediction/execution, compatibilidad antigua. |
| ARQ-049 | Principal incorporada | Inferencia compartida con modelos y compatibilidad. |
| ARQ-050 | Principal incorporada | Exportaciones separadas de inferencia. |
| ARQ-051 | Principal incorporada | Control de 300 líneas y separación de archivos. |
| ARQ-052 | Principal incorporada | Control de directorios en fuentes y tests; scripts auxiliares requieren auditoría adicional. |
| ARQ-053 | Principal incorporada | Datasets organizado por responsabilidades. |
| ARQ-054 | Principal incorporada | Gestión, readers, adapters, harmonization, validation y provenance. |
| ARQ-055 | Parcial | Adquisición centralizada en gran parte; faltan algunos flujos de proveedores. |
| ARQ-056 | Principal incorporada | Traslado de normalización, filtrado, lectura y validación GEOROC a datasets. |
| ARQ-057 | Principal incorporada | Normalización y armonización compartidas, sin lógica de entrenamiento. |
| ARQ-058 | Principal incorporada | Alias de compatibilidad y separación de adquisición/procesamiento. |
| ARQ-059 | Parcial | Experiments reorganizado parcialmente; falta completar rutas antiguas. |
| ARQ-060 | Principal incorporada | Laboratorios específicos M1, M2, integrados y compartidos. |
| ARQ-061 | Parcial | Redistribución avanzada; queda revisión de adquisición, análisis y reporting experimentales. |
| ARQ-062 | Parcial | Adquisición reusable parcialmente trasladada; faltan auxiliares experimentales. |
| ARQ-063 | Parcial | Lectura y validación compartidas; quedan auxiliares antiguos en experiments. |
| ARQ-064 | Parcial | Exportaciones y reporting experimentales todavía parcialmente duplicados. |
| ARQ-065 | Principal incorporada | Experimentos aislados de comportamiento predeterminado y promoción. |
| ARQ-066 | Parcial | Métricas genéricas y curvas desacopladas de modelos; falta comprobar independencia de todo core. |
| ARQ-067 | Principal incorporada | Paquete core/reproducibility para semilla, huella e integridad; configuración, execution, logging y scientific separados; recursos permanecen en resources por requisito específico. |
| ARQ-068 | Principal incorporada | visualization.py restaurado a raíz de core junto a exceptions.py, paths.py y states.py; ruta anterior en presentation preservada mediante alias. |
| ARQ-069 | Principal incorporada | Auditoría de tamaño trasladada a tools/inspection. |
| ARQ-070 | Principal incorporada | Retención consolidada. |
| ARQ-071 | Principal incorporada | Parámetros específicos separados de detección de hardware. |
| ARQ-072 | Principal incorporada | Contratos genéricos de métricas de regresión y clasificación centralizados en core/scientific; curvas y comparaciones comparten esas implementaciones, rutas de modelos compatibles. |
| ARQ-073 | Parcial | Auditoría AST ampliada a import_module literal, ciclos y destinos inexistentes; falta inventario y corrección integral de dependencias. |
| ARQ-074 | Principal incorporada | Familias en archivos por algoritmo. |
| ARQ-075 | Principal incorporada | Implementaciones separadas de preprocesamiento y ejecución. |
| ARQ-076 | Principal incorporada | Algoritmos M1/M2 independientes. |
| ARQ-077 | Principal incorporada | Resources independiente. |
| ARQ-078 | Principal incorporada | Detection, compatibility, allocation y monitoring. |
| ARQ-079 | Principal incorporada | Funcionalidades históricas trasladadas con alias. |
| ARQ-080 | Principal incorporada | Inventario hardware y compatibilidad por backend. |
| ARQ-081 | Principal incorporada | Política central en resources.yaml. |
| ARQ-082 | Principal incorporada | Estimación de carga y presupuesto por algoritmo; sin prometer máximos. |
| ARQ-083 | Parcial | Límites internos y secuencialidad controlada; falta planificación paralela adaptativa. |
| ARQ-084 | Principal incorporada | Monitorización CPU/RAM/GPU/VRAM y tiempos. |
| ARQ-085 | Principal incorporada | Probes y benchmark controlado; falta medir rendimiento real local. |
| ARQ-086 | Principal incorporada | Preboot consulta infraestructura de recursos. |
| ARQ-087 | Principal incorporada | Gestión interna sin opciones obligatorias nuevas. |
| ARQ-088 | Principal incorporada | Detección adaptable multiplataforma; validación física pendiente. |
| ARQ-089 | Principal incorporada | Tests reorganizados por responsabilidades principales. |
| ARQ-090 | Parcial | Faltan varios directorios específicos solicitados y revisión de rutas. |
| ARQ-091 | Principal incorporada | Pruebas unitarias, integración, regresión y científicas ampliadas. |
| ARQ-092 | Parcial | Se incorporan pruebas por etapa; ejecución diferida por restricción de CI y acceso local. |
| ARQ-093 | Principal incorporada | Mapa de pruebas por los 166 criterios. |
| ARQ-094 | Principal incorporada | Trazabilidad de requisitos, implementación, pruebas y evidencias. |
| ARQ-095 | Parcial | Pruebas de aliases, contratos de métricas y detección de ciclos literales añadidas; falta ejecución y cierre de ciclos residuales. |
| ARQ-096 | Principal incorporada | Casos científicos válidos, inválidos y límites. |
| ARQ-097 | Parcial | Benchmark sintético disponible; faltan comparaciones completas de adquisición, entrenamiento e inferencia. |
| ARQ-098 | Principal incorporada | CI ampliada y matriz existente; nuevas ejecuciones diferidas. |
| ARQ-099 | Principal incorporada | Infraestructura de logging modular. |
| ARQ-100 | Parcial | Sesiones centralizadas; quedan rutas históricas de adquisición/experimentos. |
| ARQ-101 | Principal incorporada | Identificadores únicos y correlación de errores. |
| ARQ-102 | Principal incorporada | Contexto estructurado de algoritmo, fold y trial. |
| ARQ-103 | Principal incorporada | Monitorización vinculada a operaciones. |
| ARQ-104 | Parcial | Identidad por operación y atomicidad; falta prueba y control completo multiproceso. |
| ARQ-105 | Principal incorporada | Rotación, retención y protección de evidencias. |
| ARQ-106 | Principal incorporada | Resultados de tests con run ID, JUnit y cobertura. |
| ARQ-107 | Principal incorporada | Aislamiento y comprobación de hashes antes/después; auditoría real pendiente. |
| ARQ-108 | Principal incorporada | Logging interno sin saturar consola. |
| ARQ-109 | Parcial | Requisito global: faltan implementaciones y regresión de cierre. |
