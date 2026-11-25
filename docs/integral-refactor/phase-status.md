# Estado por fases — desarrollo mediante GitHub

## Medición verificable

- 166 criterios documentados: 27 M1, 30 M2, 109 arquitectura, con requisitos superpuestos.
- 6/166 cierres acotados previos (3,6 %); 160/166 siguen abiertos. No confundir con porcentaje de implementación, cobertura de líneas o cantidad de tests.
- 57/57 diagnósticos científicos tienen evidencia parcial histórica.
- 166/166 criterios con asociaciones a módulos de pruebas (396 asociaciones). Las asociaciones no son pruebas de aceptación completa.
- La última ejecución integral observada sigue siendo b0c3857: 647 aprobadas, 7 omitidas, 57 % de cobertura de líneas, CI 7/7. No se han ejecutado los tests nuevos de esta rama tras ese commit.
- No se certifica un porcentaje de implementación integral sin revisar la aceptación individual de los requisitos.

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
