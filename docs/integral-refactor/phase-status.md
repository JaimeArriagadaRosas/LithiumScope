# Estado por fases — desarrollo mediante GitHub

## Medición verificable

- 166 criterios documentados: 27 M1, 30 M2, 109 arquitectura, con requisitos superpuestos.
- 6/166 cierres acotados previos (3,6 %); 160/166 siguen abiertos. No confundir con porcentaje de implementación, cobertura de líneas o cantidad de tests.
- 57/57 diagnósticos científicos tienen evidencia parcial histórica.
- 166/166 criterios con asociaciones a módulos de pruebas (437 asociaciones). Las asociaciones no son pruebas de aceptación completa.
- La última ejecución integral observada sigue siendo b0c3857: 647 aprobadas, 7 omitidas, 57 % de cobertura de líneas, CI 7/7. No se han ejecutado los tests nuevos de esta rama tras ese commit.
- Revisión de implementación principal, separada del cierre: 129/166 (77,7 %), 37 parciales. Véase implementation-status.md; no certifica aceptación.

## Lote de trabajo posterior a 6dc2385

| Fase | Cambios publicados | Estado de implementación | Validación |
|---|---|---|---|
| 1. Particiones internas | 870efdde: auditoría nested de Optuna, calibración y umbral; soporte y distancias | Incorporada; revisar viabilidad por algoritmo y early stopping real | Pendiente |
| 2. Evidencias locales | eaba9f21: JSON parcial estricto, identidad de regresión, no equiparar árbol modificado a commit limpio | Incorporada | Pendiente |
| 2. Predicción y PDF | 822674e0: imágenes preparadas fuera del renderizador, sesiones únicas, PDF atómico | Incorporada; revisar restantes cálculos en el renderer | Pendiente |
| 2. Documentos | f642a348: inspección estructural PDF con pypdf opcional, formato finito y evidencia local | Incorporada; revisión visual real aparte | Pendiente |
| 2. Selección y abstención | 05c19925: anexo histórico opcional, identificadores únicos y concordancia no evaluable | Incorporada | Pendiente |
| 3. Curvas de aprendizaje | 267aa026: ejecutor seis familias, preflight por defecto, agregación por fold | Incorporada; política de hiperparámetros predeterminados, no optimización de curvas | Pendiente |
| 3. Arquitectura | 9300f927: auditoría AST de importaciones, ciclos potenciales y manifiestos atómicos | Incorporada; corregir hallazgos y completar traslados | Pendiente |
| 3. Integridad de pruebas | ca8c57eb: hashes de artefactos protegidos antes/después de suite | Incorporada; no verifica todos los parches ni sustituye aislamiento | Pendiente |

La auditoría de importaciones es estática: no prueba que un ciclo detectado sea un fallo de ejecución ni detecta todas las importaciones dinámicas. El informe estructural PDF no prueba la apariencia visual. Los controles de hashes no reparan ni restauran archivos.

## Siguiente lote antes de pedir la ejecución local

1. Completar revisión estática de módulos de experimentos, adquisición, diagnósticos y renderizado restantes; resolver dependencias indebidas sin duplicar implementaciones.
2. Incorporar ejecución reproducible de ablaciones M1 (variables/preprocesamiento), M2 (representaciones/QA/cohortes), comparación de presupuesto y calibración por familia; declarar configuraciones antes de observar TEST. La representación física v5 no se impone al release ni se presume superior.
3. Reforzar trazabilidad de procedencia, control de georreferencia y QA de parches, checks de concurrencia/recursos, correlación de logs y tests de fallos de E/S. Revisar pruebas estáticas que todavía consulten rutas antiguas.
4. Completar auditoría individual de los 166 criterios, separando implementaciones terminadas de aceptación pendiente, y ampliar pruebas útiles donde falten.
5. Solo entonces solicitar git pull y opción 5.4. Evaluar informe del usuario, corregir fallos y pedir artefactos puntuales si hicieran falta. Las curvas reales, entrenamientos, independencia externa y revisión visual requieren evidencia local y no se ejecutan automáticamente en 5.4.
6. Retirar opción temporal tras validación y ejecutar CI normal. Sin activar modelos ni publicar release por el solo resultado de pruebas.

No se ha utilizado Desktop Commander en este lote ni se han ejecutado workflows. Todos los commits son intermedios y llevan [skip ci].

## Lote siguiente, hasta a07e02c8

- 19577d72: evaluación pareada con particiones externas comunes e intervalos por grupos.
- 40d2cd6a: ejecutores M1/M2 de ablación, presupuesto y calibración, preflight por defecto.
- 5a128fe3: estimación de demanda computacional por algoritmo y picos VRAM.
- 482ff622: contexto científico y correlación de errores.
- 56a74610: EDA de ambos modelos separado en estadísticas, visualización y reporting.
- a07e02c8: traslado de procesamiento GEOROC a datasets y adquisición a tools, alias históricos y huella de armonización.

| Diagnóstico | Total | Implementación principal | Parcial | Avance de implementación |
|---|---:|---:|---:|---:|
| M1 | 27 | 24 | 3 | 88,9 % |
| M2 | 30 | 23 | 7 | 76,7 % |
| ARQ | 109 | 71 | 38 | 65,1 % |
| **Total** | **166** | **118** | **48** | **71,1 %** |

Pendientes prioritarios de código: criterios P en implementation-status.md, en particular QA de imágenes y manifiestos, dominios, baselines espaciales, diagnósticos por modelo, módulos restantes de adquisición/experimentos/reporting, concurrencia y pruebas. No ejecutar entrenamientos ni pedir el pull hasta agotar implementaciones posibles.

## Lote de continuación: controles M2 (24-11-2026)

- 1b45d9f9: referencia geográfica nearest-TRAIN con QA, empates promediados y tablas OOF/fold/globales; ablación reflectancia 60 frente a reflectancia+textura 80 mediante el estudio pareado existente, v4 o v5 con artefactos verificados.
- 03d98833: sensibilidad espacial con seis protocolos predeclarados, grupos por rejilla y componentes de escena, buffer de 1/5 km exclusivamente sobre TRAIN y endpoint fijo independiente obligatorio. Preflight por defecto; ejecución opt-in y comparación con baseline prior y familias configuradas. Las particiones internas agrupan, pero no aplican el embargo adicional exterior. No selecciona automáticamente protocolo ni modelo.
- Ajustes de prueba de frontera, alineación de etiquetas, validación de identidades y registro de trazabilidad en el commit documental siguiente. No se ejecutó CI ni entrenamiento.

| Diagnóstico | Total | Implementación principal | Parcial | Avance de implementación |
|---|---:|---:|---:|---:|
| M1 | 27 | 24 | 3 | 88,9 % |
| M2 | 30 | 26 | 4 | 86,7 % |
| ARQ | 109 | 72 | 37 | 66,1 % |
| **Total** | **166** | **122** | **44** | **73,5 %** |

Criterios reclasificados por incorporación principal: M2-03, M2-11, M2-17, ARQ-036. Mantienen aceptación pendiente. Próximos: reconciliación de réplicas y censura por fuente (M1-19/M2-02), QA local SCL, geometría y manifiestos (M2-04/05/06), estabilidad y aplicabilidad M1 y arquitectura restante. No solicitar aún 5 > 4 ni iniciar entrenamientos definitivos.

## Lote de continuidad: identidad y adquisiciones registradas

- 7c3d2a01: reconciliación M2 por fuente y réplica, exclusión de concentraciones censuradas sin asignar un valor artificial, cuarentena de conflictos y auditoría.
- 34236a84: protección explícita de la cohorte congelada, firma de preparación y adquisición, checkpoints con verificación de bytes, actualización de etiquetas, descarte de entradas obsoletas, escritura atómica e historial de fallos.
- 1b0c9037: adquisición nueva estricta de seis bandas y SCL, cuadrícula común y corregistro, QA local, metadatos radiométricos y hashes; elección determinista. Se exige intervalo cerrado y prerregistrado (el intervalo abierto de la configuración histórica no se cambia). No se reacquirieron parches históricos; SCL histórica y composición temporal permanecen pendientes.
- 80c7840f: auditoría M1 de identidad y conflictos sin cambiar la cohorte; identificadores M2 estables por fuente y réplica, unidades explícitas y trazabilidad.
- Este commit: traslado de seis tests experimentales a tests/scientific/experiments, actualización de asociaciones y ajuste de tamaño de preflight M1. Los tests no se ejecutaron.

Políticas nuevas: las mediciones censuradas M2 se excluyen en preparación de nuevas muestras; unidades incompatibles se excluyen, unidades desconocidas permanecen marcadas como no verificadas. Conflictos sin réplica explícita se aíslan, no se resuelven por primera fila. Se preserva el DN en las imágenes nuevas con máscara SCL; el extractor versionado es responsable de aplicar escala/offset. La reconstrucción forzada de una cohorte no congelada reutiliza cachés compatibles y no sobrescribe automáticamente un caché corrupto; requiere recuperación explícita. No se cambia el release.

| Diagnóstico | Total | Implementación principal | Parcial | Avance de implementación |
|---|---:|---:|---:|---:|
| M1 | 27 | 24 | 3 | 88,9 % |
| M2 | 30 | 29 | 1 | 96,7 % |
| ARQ | 109 | 74 | 35 | 67,9 % |
| **Total** | **166** | **127** | **39** | **76,5 %** |

Reclasificados M2-02/05/06 y ARQ-023/024. Aceptación todavía pendiente. M1-19 y M2-04 conservan estado parcial. Próximos: estabilidad y colas altas M1, incertidumbre y dominio multivariado, composición temporal y SCL histórica, arquitectura residual, recursos y concurrencia. No solicitar todavía 5 > 4 ni ejecutar entrenamientos.

## Lote de continuidad: selección e incertidumbre M1

- 973c5f42: nueva política de elegibilidad de desarrollo previa a selección final, conserva ranking relativo. Exige superar baseline medio/mediano en RMSE medio (gate existente), ganar al menos 60 % de folds frente a mejor referencia RMSE, no exceder el peor fold de esa referencia, ni MAE/sesgo absoluto de Li alto ni peor sesgo por fuente respecto a las mejores referencias por indicador; no admite predicciones negativas. Son límites de desarrollo conservadores, no tolerancias geológicas ni aprobación de release. No se ajustaron tras observar resultados.
- 76169f7c: perfil multivariado Ledoit-Wolf de referencia de desarrollo, cobertura mínima 75 % de variables numéricas, corte empírico Q99 de distancias y categorías inéditas; disponible para nuevas bundles sin modificar releases. Estudio de intervalos conformales por máximos de error de cada grupo, ajuste/calibración/evaluación disjuntos, semilla y alpha 0,1 predefinidos, hiperparámetros por defecto, ejecución opt-in. La cobertura teórica requiere intercambiabilidad de grupos, no demostrada para los datos espaciales. La cohorte histórica no es holdout externo auténtico. El intervalo OOF existente sigue siendo descriptivo.
- Este commit: reorganización de tres módulos de tests M1 y actualización de asociaciones. No se ejecutaron pruebas ni ajustes. Se reclasifican M1-22 y M1-26 como implementación principal; M1-19 sigue parcial por resolución de conflictos reales. M2-04 sigue parcial por SCL histórica y composición temporal.

| Diagnóstico | Total | Implementación principal | Parcial | Avance de implementación |
|---|---:|---:|---:|---:|
| M1 | 27 | 26 | 1 | 96,3 % |
| M2 | 30 | 29 | 1 | 96,7 % |
| ARQ | 109 | 74 | 35 | 67,9 % |
| **Total** | **166** | **129** | **37** | **77,7 %** |

Siguiente: recuperar SCL histórica y diseñar composición temporal controlada sin reemplazar imágenes congeladas, resolución de conflictos con evidencia por fuente y arquitectura residual, concurrencia, logging y regresión. No solicitar aún 5 > 4.
