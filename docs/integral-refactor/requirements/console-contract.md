# Contrato de consola — referencia y comprobación pendiente

Fuente: Diagnóstico de Repositorio, L39-1261. El transcript original permanece local; no se publican registros de ejecución, datos, rutas o métricas ilustrativas como evidencia. La verificación de equivalencia visual/textual completa sigue abierta.

## Anclas de navegación del original

- L53: **Inicio de LithiumScope y menú principal**
- L95: Seleccione una opción [0-5]:
- L104: **Opción 1. Realizar predicción**
- L116: Seleccione una opción [0-4]:
- L119: **Opción 1.1. Predicción completa**
- L183: Presione ENTER para continuar
- L188: **Opción 1.2. Demostración integrada automática**
- L251: Presione ENTER para continuar
- L256: **Opción 1.3. Modelo 1 — Predicción de concentración**
- L283: Presione ENTER para continuar
- L286: **Opción 1.4. Modelo 2 — Prospectividad espacial**
- L311: Presione ENTER para continuar
- L316: **Opción 1.0. Volver**
- L319: Seleccione una opción [0-4]: 0
- L324: **Opción 2. Entrenar modelos**
- L340: **Opción 2.1. Entrenar Modelo 1**
- L507: Presione ENTER para continuar
- L512: **Opción 2.2. Entrenar Modelo 2**
- L724: Presione ENTER para continuar
- L729: **Opción 2.3. Entrenar ambos**
- L733: **Opción 2.0. Volver**
- L737: Se ha limpiado el registro de entrenamiento. Presione ENTER para continuar
- L742: **Opción 3. Métricas y resultados**
- L755: Seleccione una opción [0-5]:
- L760: **Opción 3.1. Visualizar PDF**
- L774: Presione ENTER para continuar
- L777: **Opción 3.2. Visualizar Excel**
- L791: Presione ENTER para continuar
- L794: **Opción 3.3. Métricas y estado de los modelos**
- L827: Presione ENTER para continuar
- L830: **Opción 3.4. Resultados del análisis exploratorio**
- L859: Presione ENTER para continuar
- L864: **Opción 3.5. Resultados de tests y simulaciones**
- L891: Presione ENTER para continuar
- L894: **Opción 3.0. Volver**
- L897: Seleccione una opción [0-5]: 0
- L902: **Opción 4. Cargar modelo versionado**
- L944: Presione ENTER para continuar
- L949: **Opción 5. Laboratorio de desarrollo y diagnóstico**
- L960: Seleccione una opción [0-3]:
- L963: **Opción 5.1. Análisis exploratorio**
- L975: Seleccione una opción [0-3]:
- L981: Seleccione una opción [0-3]: 1
- L1007: Presione ENTER para continuar
- L1013: Seleccione una opción [0-3]: 2
- L1041: Presione ENTER para continuar
- L1046: **Opción 5.2. Ejecución de tests y simulaciones**
- L1059: Seleccione una opción [0-3]: 1
- L1079: Presione ENTER para continuar
- L1115: Presione ENTER para continuar
- L1120: **Opción 5.3. Salud y diagnóstico del entorno**
- L1159: Presione ENTER para continuar
- L1164: **Opción 5.0. Volver**
- L1167: Seleccione una opción [0-3]: 0
- L1172: **Opción 0. Salir**
- L1245: Presione ENTER para continuar

## Validación necesaria
- Preboot, carga de dependencias/recursos y ENTER antes de limpiar pantalla.
- Menú principal 1 predicción, 2 entrenamiento, 3 resultados, 4 versiones, 5 Laboratorio de diagnóstico, 0 salida.
- Predicción 1 completa, 2 demostración integrada automática directa, 3 M1, 4 M2, 0 volver.
- Entrenamiento 1 M1, 2 M2, 3 ambos, 0 volver; cancelación limpia solo sesión no iniciada.
- Resultados 1 PDF, 2 Excel, 3 modelos/métricas, 4 EDA, 5 tests/simulaciones, 0 volver.
- Versiones: confirmación, cancelación y carga verificable.
- Laboratorio vigente: 1 EDA, 2 preparación y tratamiento, 3 ejecución de tests y simulaciones, 4 salud read-only, 0 volver.
- En 5 > 3: 1 Validación del software, 2 Comparaciones de modelos, 3 Evaluación de tratamiento de datos, 0 Volver. Sin Ejecutar todo ni entrada OASIS. Las comparaciones reutilizan evidencia verificable y piden autorización para escenarios pendientes; no entrenan ni promueven modelos.
- Salida y entradas inválidas: sin pérdida de evidencia; contrastar mensajes y transiciones.
- No reproducir números ilustrativos como resultados reales.
- Pendiente: pruebas de flujos completos, revisión visual y contraste de todos los mensajes de L39-1261.
