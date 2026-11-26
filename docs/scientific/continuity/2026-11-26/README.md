# Continuidad científica — LithiumScope, 26 de noviembre de 2026

> **Estado vigente M1 — optimización integral:** [Plan](M1_PLAN_OPTIMIZACION_INTEGRAL.md) aprobado. Fases 1–3 cerradas operativamente; [entrenamiento](M1_AUDITORIA_ENTRENAMIENTO.md) conserva HGB cuadrático y [postentrenamiento](M1_AUDITORIA_POSTENTRENAMIENTO.md) conserva Ridge como candidato (7.034486/.635544, guardas descriptivas), mezcla 50/50 no elegible por cola. 115 pruebas, 9 hashes de stacking verificados. Fase 4: [integración metodológica](M1_INTEGRACION_METODOLOGICA.md) implementada, 33 predictores canónicos sin recuperación asistida por Li, 121 pruebas esperadas, 80 ajustes base +10 Ridge, pendiente `lithiumscope.experiments.model_1.integral.integration`. Fases 5–7 condicionadas a resultados; [state.json](state.json) → `m1.integral_optimization`. M2, producción, main y release congelados.

> **Cierre histórico del laboratorio de datos:** [M1_CIERRE_DATOS_RESULTADOS.md](M1_CIERRE_DATOS_RESULTADOS.md) cierra fases 5–6: 85 pruebas, 15 fits, nueve hashes verificados. Mejor descriptivo C2 7.053262/.633404, no cumple objetivo RMSE ni guarda de cola. Conservar A; ninguna fila nueva aprobada. Fase 7 no ejecutada. [Fuentes y metodología](M1_FUENTES_Y_CIERRE.md) y [state.json](state.json). No repetir campañas; M2 congelado.

> **Registro histórico — cierre de refinamiento M1:** [M1_EXPANSION.md](M1_EXPANSION.md) (última sección) y [state.json](state.json) prevalecen sobre las notas históricas siguientes. Campaña académica final completa: 47/47 pruebas focalizadas, 35 ajustes de árboles y 20 estimadores locales, siete hashes verificados. Mejor RMSE medio académico 7.074911 ppm, R² medio .631231 (HGB combinado); objetivo OASIS parcialmente alcanzado (RMSE <5.361 pendiente; R² >.499 superado). CatBoost H lidera RMSE espacial experimental 111.579554 ppm. No repetir campañas históricas ni promover modelos. Siguiente línea: evaluar adquisición dirigida de datos independientes y congelar protocolo antes de nuevos ajustes. M2 congelado. Última suite completa anterior a expansión: 1002 aprobadas, 3 omitidas, 0 fallos, cobertura 57.05%; no equivale a una suite completa posterior. No hay nueva ejecución local requerida por el cierre documental.

**Entrada canónica para retomar el trabajo en otro chat.** Leer este documento, 
RESULTADOS.md, PLAN_CIERRE.md y EVIDENCIAS.md antes de modificar el código.
Rama: `feat/m1-targeted-data-refinement`. Base de esta documentación: 
`f97a2daf9266e6dae2206870988bfea122892ebf`. Main permanece en 
`a3b77a02d98e626558b81163d94fcf92da1ddfb5`. No fusionar ni publicar sin aprobación.

## Actualización de cierre anterior (26/11; registro histórico)

**[CIERRE_EXPERIMENTAL.md](CIERRE_EXPERIMENTAL.md) registra el cierre anterior; el estado vigente de M1 está en la actualización superior.** Fases 0 y 1 completas; fase 2 cerrada como cribado validado y decisión aprobada de no incorporar datos. Fases 3 y 4 cerradas sin ejecución ni fit adicional; se conservan los RF de 25/11. Suite: 1001/1004 aprobadas, 3 omitidas, cero fallos/errores, cobertura 57.02%. Comparador global complete. Página 2 validada. Restan PDF final con evidencia recuperada, reconstrucción y cierre operativo de fase 6. No publicar ni fusionar.

## Decisiones vigentes

- Objetivo principal: superar descriptivamente el release con validación espacial
  defendible y evidencias reproducibles. OASIS es una referencia académica, no
  un test compartido. No perseguir mejoras indefinidas.
- M2: conservar candidato Random Forest de 25/11, 706 muestras y 170 predictores.
  No reentrenar ni buscar nuevos tratamientos. Completar posprocesamiento de
  evidencia histórica y SCL sin ajustar modelos.
- M1: mantener la cohorte real de 3420 filas y 244 grupos como referencia congelada.
  El RF actual está guardado; balanceo 2700 y selección interna son alternativas
  experimentales, no artefactos de producción. Mantener 2043 filas reales
  equivalentes como mínimo propuesto, no requisito oficial.
- La adquisición de GEOROC, EDA y preparación son responsabilidades independientes.
  Reutilizar caché validada. No descargar ni repetir análisis completos sin necesidad.
  No mezclar automáticamente candidatos con el dataset de entrenamiento.
- No alterar archivos originales, no reconstruir OOF inventando identidades,
  no sustituir evidencia de desarrollo por validación externa independiente.
- No ejecutar otra vez campañas 45/21 configuraciones ni piloto OASIS completado.
  Conservar resultados desfavorables, no solo los mejores.
- Mantener arquitectura por módulos, máximo 300 líneas por archivo Python
  y 20 archivos Python por directorio; pruebas y documentación sincronizadas.
- Consola: conservar resumen aprobado, progreso de pytest en una línea y
  limpieza previa al resumen. Último test completo: 987/984 aprobados/0 fallos/
  0 errores/3 omitidos; cobertura 57.08%, commit b964ab3. La revisión f97a2da
  de línea dinámica no se ha validado todavía en CMD real.
- El release se reinstaló localmente durante la búsqueda de OOF; no asumir que
  la versión activa coincide con los candidatos. La demo debe mostrar identidad
  de cada modelo, estado y hash. No activar automáticamente candidatos.

## Rutas del menú actuales

| Ruta | Operación |
| --- | --- |
| 5 > 1 > 1 | EDA M1 y adquisición de fuentes candidatas |
| 5 > 2 > 1 | Preparación y tratamientos M1 |
| 5 > 3 > 1 | Tests completos |
| 5 > 3 > 5 | Comparación de versiones guardadas |
| 5 > 3 > 3 > 2 | Balanceo y expertos M1, ya ejecutado |
| 5 > 3 > 6 | Piloto OASIS, ya ejecutado |
| 1 > 2 | Demostración integrada y PDF |

## Estado de la evidencia

El archivo ZIP de respaldo conserva 158 adjuntos originales disponibles en la
sesión de trabajo, con SHA-256 individual, 191321208 bytes originales.
El ZIP pesa 30459757 bytes. SHA-256:
`275890d4065cc409579a2ea45aec2d130a6a07295304693a36d22836f91fba0b`.
El inventario dentro del ZIP tiene SHA-256
`07d73a023c56f6381ab13d238bc892984ad1f6cf243889b204064e36e82d2861`.

**El ZIP no está subido a GitHub.** El repositorio es público; los originales
incluyen datos de terceros, rutas locales e información de entorno. La conexión
remota estaba desconectada y el conector GitHub no acepta un archivo del entorno
de trabajo directamente. Se entregó el ZIP verificable al usuario. Conservarlo
en almacenamiento privado; no publicar datos brutos sin revisar permisos.
Aquí quedan los resultados agregados, hashes críticos y plan de restauración.
No afirmar que un archivo está disponible en el repositorio solo porque aparece
en el inventario. No solicitar de nuevo lo ya archivado si el ZIP está accesible.

## Referencias de implementación

- [Ciclo de datos](../../../models/DATA_LAB_LIFECYCLE.md)
- [Refinamiento focalizado](../../../models/M1_REFINAMIENTO_FOCALIZADO.md)
- [Consola de tests](../../../models/TEST_CONSOLE.md)

Siguiente paso: restauración verificada, comparador por identidad y recuperación
de informes M2. Las fases 1–6 fueron aprobadas posteriormente, con precisión de fase 5:
recuperar ambas etapas de decisión aun con dominio insuficiente y mantener
etiquetas de alcance breves. Fase 6 es solo validación. Publicación, promoción
y merge no autorizados. Experimentos y fit final siguen condicionados a evidencia.

## Implementación posterior a la aprobación

- Decisión integrada: selección exploratoria y segunda etapa siempre visible.
- Comparador M1: identidad congelada completa y resultados por fold; R² global
  e intervalo agrupado. Comparador M2: referencia reconstruida a corte 21.6 ppm
  separada de métricas publicadas de protocolo propio.
- Restaurador de los 14 archivos críticos a resultados locales, sin sobrescribir.
- Anexo dinámico: comparación reconstruida y piloto académico recalculado
  desde predicciones, sujeto a identidad.
- Sin reentrenamiento, merge o release. Queda probar el código y recuperar
  posprocesamiento M2 en el entorno local. No se ha ejecutado en ese equipo.

## Ejecución local pendiente del cierre anterior (registro histórico; no repetir por defecto)

1. Actualizar la rama con `git pull --ff-only`.
2. Menú 1 > 2: regenerar PDF con las comparaciones restauradas (run 20261126T141852_5f6cc5d4) y verificar los anexos y la página 2.
3. Comprobar reconstrucción a partir de artefactos congelados y cerrar fase 6. Revisar resultados y decidir release por separado.

Las restauraciones, suite, comparación y auditoría de candidatos ya se ejecutaron. No repetir por defecto. El detalle de los skips, los hashes y las limitaciones está en CIERRE_EXPERIMENTAL.md.
