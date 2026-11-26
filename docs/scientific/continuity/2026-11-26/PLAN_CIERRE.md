# Plan de cierre aprobado — 26 de noviembre de 2026

## Fase 0. Preservar la evidencia

Restaurar únicamente archivos verificados del ZIP en los destinos indicados
en EVIDENCIAS.md. No sobreescribir archivos existentes con otro hash. Registrar
origen, hash, tamaño, run_id y estado. Las carpetas candidatas, experimentales
y productivas siguen separadas. No volver a solicitar adjuntos recuperables.

## Fase 1. Corrección de comparación y consola

1. M1: si los hashes del dataset coinciden, verificar cobertura de identificador
   de fila, posición, fuente, Li real, coordenadas, grupo y fold; usar unión 1:1.
   Mantener fallback conservador por clave física para cohortes distintas.
   Rechazar IDs duplicados, etiquetas alteradas, folds incompatibles y OOF de
   un modelo no correspondiente. Recalcular métricas y CIs agrupados; guardar
   tabla completa de 3420, resumen global/por fold, intervalos, hash y método.
2. M2: separar estado de comparación descriptiva del release y estado de
   referencia reconstruida. No forzar el comparador estricto de datasets
   diferentes. Verificar SHA-256 del CSV reconstruido, sample_id, fold, Li,
   grupo, corte fijo 21.6 y correspondencia exacta de 706 casos. Registrar
   diferencias de protocolo y métricas originales vs reconstruidas.
3. Recuperar posprocesamiento M2 con OOF de referencia y SCL; comprobar
   metadatos y coincidencia de las 783 filas sin volver a ajustar modelos.
   Faltantes deben producir evidencia parcial explícita.
4. Validar el cambio de progreso de pytest f97a2da: reescritura en una línea,
   no envolver terminal, limpieza al terminar o cancelar, fallback redirigido.
   Resumen, JUnit, cobertura, log y eventos permanecen. Aplicar misma idea a
   otras operaciones solo si no se pierde información diagnóstica.
5. Tests unitarios de casos positivos y negativos; ejecutar suite una vez
   al finalizar cambios, no repetir campañas científicas completadas.

Aceptación: comparador guardado con M1 3420/3420 y M2 dos alcances distintos,
evidencias M2 recuperadas si identidades verificadas, cero fallos de tests.

## Fase 2. Auditoría geológica y selección de datos M1

Usar exclusivamente EDA y archivos locales ya descargados. Reusar adquisición
y no descargar de nuevo. Comparar datos candidatos con la cohorte preparada
y entre fuentes: ID, publicación+nombre de muestra, coordenadas, Li, perfiles
geoquímicos, unidades, censura, QC de óxidos y duplicados aproximados.
Identificar 0.5° grupos canónicos; auditar materiales, predictores faltantes,
litología y cambio de dominio. Registrar razón de aceptación/rechazo por fila.
Separar márgenes convergentes del conjunto intraplaca. Proteger dominio andino.
Las 69 muestras altas no son automáticamente útiles ni independientes.
Reservar grupos y/o publicaciones candidatas que nunca se usen para escoger
tratamientos, cuando haya suficiente cobertura. Ningún dato nuevo de M2.

Aceptación: catálogo candidato reproducible, fuente y grupo de cada fila,
inventario de duplicados, cobertura por banda, reporte de cambio de dominio,
cohorte propuesta explícita y registro sin cambios en dataset productivo.

## Fase 3. Experimento mínimo y condicionado

Antes de otro fit, decidir si los candidatos aportan cobertura suficiente.
Si no, cerrar con RF actual o balanceo documentado. Si sí, propuesta de cuatro
brazos RF, sobre los mismos cinco TEST históricos íntegros:
A control actual; B balanceo moderado/selección interna reproducible;
C B + muestras convergentes compatibles; D B + extensión alta intraplaca
como estudio de transferencia, no como aprobación de dominio.

Usar el mismo contrato de predictores, hiperparámetros congelados o búsqueda
interna igual para todos, selección únicamente dentro de TRAIN. Excluir de
cada TRAIN externo/interno nuevas filas con grupo o identidad coincidente
con su TEST correspondiente; considerar separación bibliográfica para evitar
dependencia. Deduplicación antes de particionar, sin usar etiquetas TEST
para decidir la extensión; todas las decisiones de elegibilidad predefinidas
y registradas. Nunca mover TEST histórico, no retirar nueve valores extremos.
Mantener holdout adicional separado si hay suficientes datos nuevos.
El control debe reproducir el resultado archivado; reportar cualquier
diferencia. No repetir HGB/CatBoost ni expertos salvo evidencia concreta.
Presupuesto de fits y criterio de parada se fijan antes de ejecutar.
No usar el recorte OASIS como solución productiva.

Métricas: RMSE medio/global, MAE, R2 medio/global, sesgo alto, peor fold,
error por fuente y banda, TRAIN/TEST y OOF completo. Comparación emparejada
por grupo, IC95 de diferencias, estabilidad. Criterio principal: mejora
frente a RF actual y release, MAE no degradado materialmente (propuesta:
tolerancia máxima 0.1 ppm preacordada), no predicciones negativas, sin
pérdida de representatividad del dominio y mínimo 2043 reales equivalentes.
Si no mejora o degrada dominio, mantener actual; no iniciar otra búsqueda
indefinida. No declarar superioridad estadística por escoger el mejor
resultado en los mismos folds. Si no hay holdout independiente, declararlo.

## Fase 4. Entrenamiento final de M1 y publicación condicionada

Solo si un tratamiento supera criterios: integrar contrato determinista de
balanceo/selección (no copiar modelo de un fold), ajustar dentro de cada
TRAIN, registrar selección final sin utilizar TEST, ajustar candidato final
una vez y guardar identidad/hashes/versión. Verificar reproducibilidad y
predicción integrada con M2 RF ya guardado. Mantener release anterior,
no publicar ni cambiar main hasta autorización. Si no hay mejora material,
conservar RF actual evita reentrenamiento innecesario. La campaña balanceada
no produjo modelos guardados.

## Fase 5. PDF, decisión integrada y defensa

Aprobación posterior a la revisión de los PDF del 13 y 26 de noviembre:
- Mantener página 2 con las dos etapas. Separar validez numérica de aplicabilidad.
- Priorizar M1 aplicable y completar hasta tres con predicciones válidas de alcance
  exploratorio. Usar q90 uniforme si disponible; en otro caso Li estimado.
- Comparar los scores M2 aplicables si hay al menos dos; si no, usar los scores
  válidos disponibles para ranking exploratorio. Mostrar las tres vistas reales
  cuando existan; si faltan imágenes, mostrar su estado, no inventarlas.
- Mantener empates y falta real de datos como casos sin ganador. Etapa 2 visible.
- Etiquetas breves de alcance; no repetir advertencias en la página 2. Mantener
  las limitaciones científicas en las secciones de diagnóstico y evaluación.
- Separar conteo marginal del diagnóstico final (incluye multivariado).
- No cambiar predicciones, umbrales ni etiquetas para mejorar la demostración.
- Validar PDF del release y candidatos, nueva generación y reconstrucción.
  No usar estos diez casos para seleccionar o ajustar hiperparámetros.


Anexo dinámico vinculado al hash de cada modelo realmente utilizado:
cohorte y EDA, tratamientos, ranking, OOF, TRAIN/TEST, comparación release,
OASIS y limitaciones, candidatos no incorporados, tablas y figuras verificadas.
Recuperar también evidencia de piloto (no asumir que el renderizador actual
ya la lee) y evitar mezclar un modelo activo del release con evidencia RF.
Comprobar render real, portada, tablas, figuras, páginas, unidades, manifiesto
y reconstrucción sin rutas vivas. Revisar que la salida no oculte las
limitaciones de M2 ni presente proxy SCL como exposición de roca certificada.
Generar evidencia final y documentación de instalación; suite y CI antes
de cualquier decisión futura; no repetir investigación ya completada.

## Fase 6. Validación final sin publicación automática

Validar tests, integridad de artefactos y operación real. El release es una decisión
posterior e independiente: solo si el conjunto final supera la referencia publicada
bajo criterios científicos y funcionales aprobados. Considerar todos los candidatos
experimentales desde el release anterior; no asumir que el más reciente es mejor.
No empaquetar, publicar, promover ni fusionar main por completar las fases.

## Criterio global de cierre

M2 queda cerrado en optimización; faltan evidencia e integración documental.
M1 ya supera descriptivamente el release, con comparación emparejada
exploratoria favorable para selección interna. OASIS: R2 mayor, RMSE mayor;
sin exigencia de superar todas sus métricas. Una última evaluación
condicionada de datos reales basta para decidir. Evitar trabajo adicional
sin beneficio científico medible.
