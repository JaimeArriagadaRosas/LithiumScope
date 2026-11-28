# M1 — último refinamiento de EDA y tratamiento de datos

Fecha: 28/11/2026. Rama: feat/training-refinement.
Estado: implementación diagnóstica de fases 1–4 publicada tras aprobación; pendiente validación local y revisión de evidencia. Fases 5–6 condicionadas a esa revisión.
No ejecutar campañas, descargar fuentes, ajustar modelos, modificar datos,
activar workflows ni fusionar main. M2 y producción permanecen congelados.

## Evidencia y punto de partida

EDA local eda_20261128T100117_090336Z_db98fe5f, firma
5ee7e90ecf49f0bda83755cf679cc457dbf714d18af6f6bbb7c4c8f392d2870b,
fuente training_combined.csv SHA256
94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94.
4765 filas originales, 3420 preparadas, 33 predictores, 244 grupos.
16 tablas adjuntas verificadas anteriormente; no se dispone aquí de los
originales completos ni de las particiones GEOROC para probar recuperación.
EDA principal completo, EDA global parcial por dos bloqueos de fuentes
candidatas (Empty header at column 172). No fit ni modificación de datos.

La población y el hash coinciden con el cierre anterior. El cierre fue
operativo: no repetir expansión, balanceo, síntesis, pérdida, calibración
o búsqueda masiva ya evaluada. No borrar evidencia negativa.
Referencias: [cierre de datos](M1_CIERRE_DATOS_RESULTADOS.md),
[fuentes](M1_FUENTES_Y_CIERRE.md), [expansión](M1_EXPANSION.md),
[aumento](M1_AUGMENTATION.md), [optimización](M1_PLAN_OPTIMIZACION_INTEGRAL.md),
[recuperación espacial pendiente](M1_RECUPERACION_ESPACIAL.md).

## Inventario de fuentes extra y alcance real del EDA

| Fuente | Evidencia anterior | Decisión |
| --- | --- | --- |
| GEOROC Andean Arc, DOI 10.25625/PVFZCE, 3 CSV | 2998 candidatas, 2990 claves históricas exactas, 2750 coincidentes con preparadas; 14 altas >696 coinciden con histórico | No anexar duplicados; recuperación lateral experimental |
| GEOROC Granite/Rhyolite, DOI 10.25625/2JETOA, 6 CSV | 8234 candidatas, 221 claves históricas exactas, 205 preparadas; 69 altas intraplaca sin soporte inicial | Transferencia experimental, no producción |
| Chen 2020, DOI 10.1038/s41467-020-19106-z | 3680 Li positivos, 1588 compatibles en al menos un fold, 1482 <=20 ppm, cuatro >40, ninguno >60 | C1/C2/D ensayados; no admisión productiva |
| Meixner 2020, DOI 10.1007/s00126-019-00915-2 | 80 principales con Li positivo, solo SiO2 entre 22 químicos canónicos; cero coincidencias sitio+SiO2 | No incorporación; identidad no resuelta |
| Fernández-Turiel 2020, DOI 10.20350/DIGITALCSIC/10527 | 6 roca total con 22 químicos y Li 19.96–29.64; cero compatibles en los cinco folds | No incorporación |
| Abat 2025 Macusani, DOI 10.60520/IEDA/113778 | 29 registros, 22 Li positivo 310–4160, 8 QC, 20 posibles duplicados y publicación compartida | Cuarentena; no independencia |

Conteos de etapas distintas, no sumar como observaciones nuevas.
Las fuentes extra históricamente descargadas son diferentes del GEOROC ya
integrado en la cohorte canónica (2744 GEOROC y 676 Mamani09 preparados).
Las 9 particiones de expansión fueron verificadas localmente en el estudio
del 26/11; no se ha comprobado su presencia actual en disco en esta sesión.
Se recuperaron 2686 enlaces retrospectivos a muestras históricas,
2531 observaciones académicas enriquecidas y 58708 valores adicionales,
sin nuevas filas canónicas. La recuperación asistida por Li no certifica
identidad disponible durante inferencia. CatBoost H con transferencia
atenuada: RMSE espacial global 111.579554 frente a RF 152.302214; no elegible
por guardas. Chen ponderado C2: RMSE medio académico 7.053262 frente a A
7.074911, pero RMSE cola >40 25.093647 frente a 24.888575; no aprobado.
Cero nuevas filas aprobadas para producción.

En el estado previo a esta implementación, el registro source_studies/registry.py habilitaba automáticamente solo
georoc_candidate_eda para M1. Los otros estudios (Chen, Meixner, whole_rock,
directed_intake, UNAB) se invocan de forma independiente y requieren entradas
declaradas. En la ejecución entregada ambos conjuntos GEOROC candidatos
fueron bloqueados antes de completar su perfil. Los demás estudios no
fueron ejecutados por el EDA automático. No implica que sus evaluaciones
anteriores se hayan perdido. Las evidencias previas se conservan localmente
con hashes documentados; no afirmar disponibilidad actual sin verificar.

## Plan aprobado (seis fases)

### 1. Corrección funcional del EDA
Inspeccionar encabezados originales, separar columnas auxiliares vacías
de columnas científicas y tratar pies bibliográficos con contrato explícito.
No permitir encabezados inválidos indiscriminadamente ni modificar originales.
Validar anchura, alias y mapeos, registrar hashes, particiones y errores.
Separar estado del EDA canónico de fuentes candidatas. Reusar evidencias
compatibles sin repetir adquisiciones o campañas.

Extensión de trazabilidad de fuentes: inventario integral de fuentes
registradas/adquiridas, originales disponibles, identidad y manifiesto,
estado de EDA previo/actual, compatibilidad, solapamiento, cobertura de
cola, decisión de uso y composición efectiva TRAIN por fold. Distinguir
fuente no evaluada, bloqueada, evaluada histórica, reutilizada y rechazada.
No presentar un inventario como nuevo EDA ni incorporar automáticamente.
Antes de reejecutar un estudio, verificar identidad y evidencia previa.
Las fuentes extra deben ser visibles desde el flujo M1 aunque se mantengan
estudios independientes; no ejecutar ajustes o descargas por defecto.

#### Ejecución secuencial y cierre del ciclo de cada dataset (precisión aprobada)

El comando 5 > 1 > 1 debe recorrer un dataset a la vez, en orden explícito
y reproducible del registro de fuentes. No depender del orden del sistema
de archivos. Orden inicial propuesto: combinado canónico (con perfiles
separados de Mamani09 y GEOROC histórico), GEOROC Andean Arc,
GEOROC Granite/Rhyolite, Chen, Meixner, Fernández-Turiel y Macusani.
Las particiones de una misma fuente se verifican individualmente y se
consolidan antes de pasar a la siguiente fuente. No contar el combinado
y sus componentes como poblaciones independientes. Inventariar además
cualquier otro dataset M1 adquirido y registrado que se descubra en el
catálogo local, sin admitir archivos arbitrarios ni descargar por defecto.

Para cada fuente activa, imprimir nombre, posición i/N, disponibilidad,
identidad y estado. Si falta un original o una partición, enumerar lo que
falta y continuar con el siguiente dataset; no dar por rechazada una
fuente ausente. Si falla el lector o la validación, registrar el error
con contexto, preservar evidencia parcial y continuar. Un manifiesto
parcial no acredita una evaluación completa. No sustituir el dato faltante
por otro de procedencia distinta. Reutilizar perfiles anteriores completos
y verificados cuando su identidad y metodología sean compatibles.

Cada dataset disponible requiere un perfil individual adecuado a su
formato y contenido (no aplicar indiscriminadamente el esquema canónico):
- Inventario de archivos, hojas/particiones, dimensiones, mediciones,
  unidades, métodos, material, referencias, licencia, CRS y precisión.
- Li medido, ausente, censurado, inválido, distribución por bandas y cola;
  cobertura de predictores, calidad de óxidos y química, categorías,
  redundancia, faltantes, extremos y soporte multivariado/espacial.
- Identidad de muestras, duplicados y solapamiento con histórico y
  otras fuentes; separar recuperaciones de mediciones, nuevas filas y
  posibles conjuntos de evaluación independiente.
- Aporte marginal al balance y cobertura por concentración, química,
  grupo y publicación, no solo al número total de filas. Estimar
  elegibilidad por TRAIN/fold en la fase de preparación, sin usar TEST
  para seleccionar ni incorporar muestras.
- Indicar uso efectivo anterior, tratamientos, resultados comparativos,
  guardas satisfechas/incumplidas, limitaciones y recomendación razonada.
  Aprovechar resultados históricos compatibles para evitar repetir fits.
  Si no hay evidencia de rendimiento suficiente, dejar la fuente en
  evaluación; el EDA por sí solo no prueba una mejora predictiva.

Mantener un registro persistente de decisiones por fuente y versión/hash.
Estados de ejecución (disponible, faltante, bloqueado, completo, reutilizado)
son independientes de la disposición científica (canónico, candidato,
en evaluación, admitido para un uso definido, cuarentena, rechazado).
La recomendación EDA no equivale a aceptación productiva ni autoriza fit.
Antes de adoptar datos, la preparación y evaluación de tratamientos
demuestran aporte marginal frente al control congelado, con guardas de
cola, generalización y paridad de inferencia. Registrar resultados
negativos y positivos, y distinguir incorporación de filas de
enriquecimiento/transferencia o uso solo diagnóstico.

Una fuente con rechazo definitivo respaldado por evidencia y decisión
explícita se retira de la lista activa del EDA: no descargar, volver a
leer, perfilar ni evaluar automáticamente en ejecuciones posteriores.
Conservar su ficha de cierre en el catálogo histórico separado del flujo
activo, con identidad/versión, perfil completo, resultados, motivos
específicos, fecha, decisión y enlaces/hashes de evidencias. No eliminar
originales ni documentación por defecto. Cuarentena, fallo de lectura
y ausencia no equivalen a rechazo definitivo. La retirada se aplica
también a las entradas automáticas del registro, sin eliminar funciones
científicas que tengan otros usos vigentes. Una fuente rechazada solo
puede reabrirse de forma explícita y justificada (por ejemplo, una versión
realmente nueva o evidencia primaria nueva); nunca por reiniciar el EDA.
Si una fuente tiene usos diferenciados, cerrar cada uso por separado y
retirarla del flujo únicamente cuando ninguno siga vigente. No reabrir
en bloque las campañas cerradas el 26/11: consolidar primero evidencia
existente y decisiones definitivas, y analizar solo brechas pendientes.

Consola: mostrar progreso y un resumen por fuente al finalizar su turno;
al terminar, listar resultados, ausentes, bloqueos, evidencias reutilizadas,
candidatos pendientes y decisiones de cierre. Las fuentes ya retiradas
no forman parte del denominador activo; pueden figurar únicamente como
referencia al catálogo histórico, sin procesarse. El informe global
distingue estado del EDA canónico y estado de las fuentes activas.
Probar orden determinista, continuidad ante ausencia/error, integridad
de caché, tratamiento de particiones, aislamiento entre fuentes y exclusión
persistente de rechazados. La evaluación de datos y el entrenamiento
siguen siendo etapas separadas, sin ajustes automáticos desde el EDA.

### 2. Refinamiento del control de calidad
Auditar 109 filas con suma de óxidos faltante; 160 bajo 94, 3 sobre 102.
Reconstruir únicamente desde mediciones originales completas compatibles,
conservar base de hierro, agua, unidades, censura y linaje.
No imputar óxidos para aprobar QC; no relajar umbrales 94–102.
Examinar ocho ceros de objetivo antes de QC y distinguir censura/medición.
No recuperar objetivos faltantes sin identidad independiente y Li medido.
Mantener cohorte congelada como control; cualquier recuperación es candidata.

### 3. Identidad y separación espacial
Auditar repetición de coordenadas y nombres, publicación, similitud química,
intervalos geográficos y CRS. No inferir fuga de proximidad o archivo común.
242/244 u otros recuentos de protocolos distintos no se confunden.
La ejecución actual tiene 244 grupos, 24 TEST a <=1 km y 471 a <=10 km;
política mínima actual 0 m y CRS no certificado. Conservar folds históricos.
Proponer sensibilidad de bloques/buffer más estrictos, seleccionada sin TEST,
con cobertura por fuente, grupo y cola. No cambiar evaluación para favorecer
métricas; registrar pérdidas de TRAIN y limitaciones de precisión.

### 4. Ampliación EDA
Perfiles por fuente, banda, publicación/grupo, correlación intra-fuente,
faltantes conjuntos, categorías unknown y asociación de procedencia,
colinealidad, soporte de TRAIN, extrapolación química/geográfica, calidad
de etiquetas, disponibilidad en inferencia, identidad y linaje.
Separar estadísticas descriptivas globales y decisiones aprendidas en TRAIN.
Exportar tablas y alertas estructuradas con identidad/versión; sin ajuste.

### 5. Tratamientos experimentales condicionados
Solo si fases 1–4 justifican una brecha: recuperación química verificable,
indicadores de ausencia si no cubiertos, variantes de predictores y
transformación composicional ILR de óxidos mayores (prioridad baja).
Antes de ILR verificar inventario de técnicas anteriores, cero/censura,
componentes positivos y tratamiento de faltantes dentro de TRAIN, base
ortonormal y paridad de inferencia. No confundir log-razones ya probadas
con ILR. No acumular tratamientos automáticamente.
Predeclarar hipótesis, control, folds, presupuesto y guardas por cada brazo;
no realizar ningún fit hasta aprobación de ejecución y evidencia suficiente.
Mantener TEST histórico completo; nueva cohorte/protocolo en brazo separado,
sin solapamiento de identidades. No repetir SMOTER, balanceos, expansión
masiva ni búsquedas de hiperparámetros cerradas.

### 6. Validación y cierre
Comparar RMSE/MAE/R², sesgo y cola por fold, fuente/grupo y soporte.
No seleccionar con TEST ni reclamar independencia de folds reutilizados.
Exigir trazabilidad, ausencia de fuga, paridad de preparación/inferencia,
reproducibilidad y regresión local. Conservar control si no mejora guardas;
registrar resultados negativos y cerrar. No promover sin autorización.

## Flujo funcional y continuidad

5 > 1 > 1 EDA M1 (canónico y evidencia de fuentes);
5 > 2 > 1 preparación y planes TRAIN;
5 > 3 > 3 evaluación experimental de tratamiento de datos.
Mantener responsabilidades separadas, sin nuevos menús innecesarios.
Arquitectura: 300 líneas Python/archivo y 20 Python/directorio; tests por
propósito y referencias canónicas; validación local sin workflows.

Tras cerrar el refinamiento, retomar la recuperación espacial 4B ya
implementada (10 fits presupuestados), previa verificación de compatibilidad
de identidades y artefactos. Si un tratamiento cambia la matriz, no mezclar
OOF de cohortes/configuraciones distintas; conservar comparación histórica
y definir un control pareado nuevo antes de reentrenar. Fase 5 de candidato
de referencia continúa bloqueada por elegibilidad espacial, fase 6 de
fine-tuning no iniciada. No iniciar nueva búsqueda masiva de Optuna.

## Estado de aprobación

Implementación autorizada. No se autoriza todavía ejecutar fits nuevos,
promover modelos o modificar datos canónicos sin revisar la evidencia.

## Entrega de implementación (fases 1–4)

- Normalización temporal de exportaciones GEOROC: columnas sin encabezado
  solo si vacías en filas científicas; pies bibliográficos explícitos,
  inventario de todas las columnas y huellas originales. El archivo de
  datos real no se ha probado en esta sesión.
- EDA secuencial del combinado y seis fuentes registradas: Andean Arc,
  Granite/Rhyolite, Chen, Meixner, Fernández-Turiel y Macusani. No descarga.
  Faltantes, discrepancias de hash y errores se distinguen de rechazo.
  Perfiles específicos por formato, evidencia parcial y caché verificada.
  No se certifica cobertura de fuentes no registradas.
- Inventario en results/model_1/source_catalog/executions/eda_*/source_inventory.json;
  perfiles candidatos en results/model_1/candidate_eda. Manifiestos y
  referencias históricas con SHA256. Retiro explícito por versión y
  justificación, sin borrar archivos. Ninguna fuente retirada ahora.
- Auditorías canónicas nuevas: qc_recovery_review, qc_recovery_summary,
  target_analytical_review, target_analytical_summary, identity_review,
  possible_identity_pairs, outer_chemical_support, outer_identity_support_summary,
  spatial_buffer_sensitivity y tablas de representatividad, correlación
  por fuente, faltantes conjuntos y preparación composicional.
  Las sumas reconstruibles son hipótesis de revisión, no filas recuperadas.
  Distancias nominales y duplicados posibles no acreditan independencia.
- Pruebas de regresión añadidas; no ejecutadas aquí. M2, originales,
  modelos y main sin cambios. No workflows.

Retirar solo tras decisión científica explícita y perfil íntegro:
`python -m lithiumscope.experiments.data_lab.source_studies.source_catalog retire FUENTE RUTA_PERFIL --reason "motivo" --decision-reference "evidencia" --all-uses-closed`.
Reabrir explícitamente mediante subcomando reopen y --reason. Mantener
evidencias y catálogo histórico; no ejecutar estos comandos por defecto.

### Punto de control local

En feat/training-refinement, git pull --ff-only. Ejecutar primero
5 > 3 > 1 para suite completa; después 5 > 1 > 1. Revisar informe
eda_verification_*.json, source_inventory.json, manifiestos y tablas.
Si la lectura GEOROC falla, conservar error y raw_structure_audit si existe.
No ejecutar 5 > 2 > 1 ni campañas de tratamiento por este punto de control:
la preparación general existente no incorpora aún las decisiones de este
refinamiento. Pendientes: verificar originales de exclusiones QC, comparar
evidencia entre fuentes y resolver elegibilidad; definir tratamientos
condicionados TRAIN sin repetir campañas; ejecutar y cerrar fase 6 solo con
resultados reales. No hay mejora predictiva ni cierre científico demostrado.


## Punto de control local posterior (28/11, 11:10 UTC)

Ejecución canónica eda_20261128T110953_162096Z_c46e99f5; firma registrada en el manifiesto original.
34/34 CSV entregados verificados por SHA256. Cohorte 3420 × 33, 244
grupos, hash congelado sin cambios. Cero ajustes. Los cuatro estudios
dirigidos (Chen, Meixner, Fernández-Turiel, Macusani) completaron su
perfil tras corregir cuantiles de booleanos. Las nueve particiones GEOROC
verificadas fallaron por la leyenda inline "Abbreviations: MET: ...",
no por calidad científica. Corregido reconocimiento conservador y prueba
sintética de 172 columnas; pendiente confirmar en archivos reales.

De 109 sumas QC faltantes, 76 filas no tienen ningún óxido numérico
disponible en el combinado y 33 tienen nueve: 25 sin SiO2, seis sin P2O5,
una sin Fe2O3 y una sin MnO. No existe reconstrucción directa certificada.
Nueva tabla qc_missing_oxide_patterns y referencias de procedencia
declaradas, no certificadas, para una revisión puntual de los originales.
No imputar para superar QC.

14 pares de identidad potenciales; cero atraviesan TRAIN/TEST según el
criterio aplicado. 269/3420 TEST fuera del soporte químico; las nueve
observaciones >696 ppm del fold 5 carecen también de soporte químico.
No certifica ausencia de fuga ni requiere modificar folds congelados.
No ampliar indiscriminadamente el EDA canónico; priorizar completar
GEOROC y consolidar aporte marginal, evidencia histórica y decisiones.

Consola M1 compacta, bloques por cohorte, calidad, soporte y alertas;
cada fuente informa progreso, número de archivos verificados, estado,
conteos y duración. Exporta source_comparison.csv junto al inventario,
con hash y definiciones de QC y conteos; N/D no se interpreta como cero.
Los conteos de fuentes no equivalen a muestras independientes ni admisión.
Se preserva el detalle completo en tablas, manifiestos y logs.

Siguiente ejecución: 5 > 1 > 1; aportar eda_verification y el nuevo
source_inventory.json/source_comparison.csv, y los manifiestos/tablas
de fuentes candidatas necesarios para la evaluación. Si persiste bloqueo
GEOROC, entregar raw_structure_audit.json de cada fuente afectada.
Validación software completa pendiente; no ejecutar tratamientos ni fits.


## Cierre funcional del EDA local (28/11, 11:32 UTC)

Commit ejecutado 2cde42c698e2e3477c693bc9bb5c200a1ebb4674.
Informe eda_verification_20261128T113220_b0df91c9.json, estado COMPLETADO.
EDA eda_20261128T113124_016337Z_5451af22, firma
81df1b4f5adc451b82024367a0b78feb3c200705bb76aaf23057d478287a86f8.
Inventario eda_20261128T113220_328539Z_f771bdcf/source_inventory.json;
source_comparison.csv SHA256
68b87ef7fc2fe328e5a217b233602b13ae2c6f536de82a18904d726b68d7fb92
recalculado y coincidente. Los seis registros de fuentes coinciden entre
inventario e informe. 17 originales adicionales reportados verificados
por la ejecución local, 2 perfiles nuevos, 4 reutilizados, 11 referencias
históricas verificadas (10 estudios distintos por ruta). Cero bloqueos,
cero retiros, cero nuevas filas aprobadas, cero fits. 35 CSV y seis figuras
declarados en manifiesto canónico; no se adjuntaron sus bytes en esta
entrega para recalcular sus hashes. Fuente congelada y protocolo de cinco
folds con identidades sin cambios. La nueva presentación funciona.

| Fuente | Filas perfiladas | Li >40 | Li >696 | QC sin alerta identidad | >696 QC sin alerta |
|---|---:|---:|---:|---:|---:|
| Andean Arc | 2801 | 219 | 14 | 6 | 0 |
| Granite/Rhyolite | 7996 | 2287 | 117 | 6694 | 69 |
| Chen | 3680 | 102 | 0 | N/D | N/D |
| Meixner | 82 | 19 | 0 | N/D | N/D |
| Fernández-Turiel | 6 | 0 | 0 | N/D | N/D |
| Macusani | 29 | 22 | 16 | N/D | N/D |

Conteos de perfil, no identidades independientes ni elegibilidad. QC
GEOROC y QC Chen tienen definiciones distintas. Se conservan las cifras
anteriores de expansión (2998 y 8234); el perfil actual reporta 2801 y
7996 (197 y 238 menos). No atribuir la diferencia a una causa sin
contrastar las tablas de ingestión, filtros y claves de fila; el mero
recuento de 69 altas coincidente no demuestra misma membresía.

### Recomendaciones de disposición, pendientes de decisión explícita

- Andean Arc: cerrar expansión masiva; seis QC sin alertas requieren
  verificación individual si se consideran nuevas filas. Ninguna >696.
  Mantener únicamente revisión de procedencia/enriquecimiento y recuperación
  de exclusiones con identidad independiente de Li, no repetir campaña
  retrospectiva de 2686 enlaces.
- Granite/Rhyolite: cerrar repetición de transferencia amplia y pesada.
  69 altas QC sin alerta (53 >1000); el estudio anterior encontró 69
  intraplaca sin soporte químico y fallos de estabilidad pese a menor RMSE
  global. Reconciliar membresía exacta y soporte TRAIN antes de considerar
  cualquier hipótesis nueva. No admitir 6694 filas ni 69 altas por conteo.
  Si no hay información nueva relevante, mantener cerrado ese uso.
- Chen: candidato a retiro del EDA activo, no repetir C1/C2/D. 1588
  compatibles históricas, cuatro >40, ninguna >60; C2 mejora RMSE medio
  académico .021649 ppm pero empeora cola .205072. No aporta >696.
- Meixner: candidato a retiro del EDA activo, insuficiencia de química
  canónica (1/22) e identidad no resuelta; no usar estándares, lixiviación
  ni Li como clave de recuperación de objetivos faltantes.
- Fernández-Turiel: candidato a retiro del EDA activo; seis registros
  con buena trazabilidad pero sin soporte compatible en preflight histórico
  de cinco folds y ninguno >40. No rechazo de calidad de la publicación.
- Macusani: mantener cuarentena por publicación compartida 30009 y
  potenciales duplicados. 16 >696 descriptivas, 8 QC históricas en total,
  20 posibles duplicados y 29 bloqueadas por publicación. Uso restringido
  a auditoría de procedencia/mediciones, no expansión ni holdout independiente.

No se ha ejecutado retiro: todos siguen review_required en el inventario
real. El retiro persistente requiere decisión explícita y evidencia local
íntegra, cierre de todos los usos; conservar fichas y originales.

### Siguiente punto de decisión (fase 5)

Cerrar ampliación descriptiva general, sin más campañas de adquisición
ni repetición de EDA. Priorizar, sin fit:
1. Reconciliar dos vistas GEOROC por clave y razón de filtrado usando
   evidencias existentes. Revisar las seis andinas y 69 félsicas solo
   si una diferencia real modifica elegibilidad. Recuperar bibliografía
   y mediciones de las 33 exclusiones QC parcialmente observadas desde
   originales si hay identidad inequívoca sin Li, sin imputación para QC.
2. Verificar inventario previo de tratamientos antes de definir brazos:
   ablación de categorías asociadas a fuente, indicadores de ausencia
   y transformación ILR de óxidos (3363 completos positivos; 44 con cero,
   13 incompletos). No repetir tratamientos equivalentes. Contrastar
   mediciones/ceros, bases de composición y paridad de inferencia.
3. Solo ante una hipótesis nueva y admisible, diseño pareado TRAIN-only
   con control congelado, folds internos, guardas de cola, MAE, sesgo
   y estabilidad espacial; presupuesto fijo previo a ejecución. Los
   TEST observados siguen siendo desarrollo reutilizado, no holdout.
4. Si no existe ganancia de información justificable, mantener cohorte
   y metodología, cerrar tratamiento y retomar recuperación espacial 4B
   pendiente (10 fits según protocolo predeclarado, previa compatibilidad).

La suite completa posterior a las modificaciones sigue pendiente.
Este cierre funcional no certifica mejora predictiva, nuevas filas,
independencia espacial, método, CRS ni cierre de fases 5–6.
