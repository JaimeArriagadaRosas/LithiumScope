# M2 — Experimento final de expansión espectral v4, seis algoritmos

**Corte:** 13-11-2026. **Estado:** extractor, adquisición y protocolos experimentales implementados. Solo pilotos de adquisición real ejecutados. **NO** se ha completado el entrenamiento de 170 variables ni confirmado una ganancia sobre el release.

## 1. Función de M1 y M2

**M1** estima litio geoquímico y entrega tres candidatos. **M2** examina principalmente sus imágenes Sentinel-2 y ayuda a escoger el Top-1. La matriz X de M2 no incorpora las 33 variables químicas de M1; las concentraciones de Li se utilizan para entrenamiento supervisado y evaluación, no como entradas predictoras. La validación final M1→Top3→M2→Top1 sobre candidatos reales de M1 **todavía está pendiente**, distinta de las ternas descriptivas históricas.

Son inamovibles las seis familias: **Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost y SVM-RBF**. No se eliminó ninguna. El menú de entrenamiento M2 conserva la modalidad histórica de 706 muestras, el extractor v3 y todas las versiones anteriores. No se reescribe ni activa ningún release.

## 2. Nuevas imágenes y 170 características

La representación v3 mantiene **126 características** de las bandas Sentinel-2 originales B02/B03/B04/B08/B11/B12, incluidos resúmenes de banda, textura e índices. Son valores históricos preservados byte a byte al construir el bloque base.

La v4 añade exactamente **44 características** derivadas de **cuatro bandas reales** de la **misma escena STAC y fecha que el parche original**: B05 rededge1 705 nm, B06 rededge2 740 nm, B07 rededge3 783 nm, B8A nir08 865 nm. Todas tienen resolución nativa 20 m: se extrae el mismo parche geográfico 640 m remuestreado a la rejilla común 32×32.

Se incorporan **32 estadísticas/texturas de bandas** (8×4) y **12 estadísticas de dos índices red-edge** (6×2). Las cuatro bandas nuevas se convierten de DN a reflectancia con los parámetros individuales scale/offset guardados en el STAC, conservando máscaras de nodata. La v3 conserva sus 126 valores previos calculados en DN; la matriz resultante es explícitamente de dos bloques y no sustituye por sorpresa el sistema original. La corrección radiométrica completa de las seis bandas v3 constituye otro experimento distinto, no oculto en este.

Los parches y las fuentes se guardan con SHA-256. La matriz de 170 debe contener **todas las 706 filas** para permitir competencia. No se fabrican espectros ni etiquetas con IA. Si los datos nuevos tienen cobertura menor que 95 %, esas observaciones se excluyen solo de TRAIN; los TEST históricos se conservan. Una imagen con cobertura casi nula suspende la fase en lugar de inventar números.

**Piloto completado:** SAR-00-08, escena S2C_18LXJ_20260829_0_L2A, cuatro bandas y 44 features reales, QA 100 %. Prevalidación de construcción 3/3 parches reales, sin errores. El estado registrado es **incomplete_no_training** y no existe aún matriz completa ni modelo v4 entrenado.

## 3. Nuevas muestras geoquímicas: alcance real

- Inventario GEOROC: 2.744 filas revisadas; 1.094 provisionales para revisión, 1.040 con filtro preliminar válido.
- Cohorte anterior: 706 originales y 77 provisionales adquiridas; 23 pendientes de 100 preseleccionadas.
- Un inventario nuevo identifica **30 grupos** de 0,5° realmente nuevos frente a los 783 actuales. De ellos, 23 son pendientes de descarga y siete agregados a la cola, tras quitar grupos repetidos. No representan automáticamente 30 muestras utilizables.
- Piloto de sitios nuevos: GEOROC_0d76ae9c0415cebc se rechazó al no haber una escena apta dentro de la ventana fija; **GEOROC_88fd6dcc16bae597** se adquirió y superó QA de seis y cuatro bandas con escena S2B_19KER_20261103_0_L2A. **Una** muestra nueva aceptada técnicamente, etiquetada **provisional y no certificada en terreno**.
- Los candidatos guardan DOI, tipo de roca, Li original, coordenadas, grupo y hashes; no se agregan automáticamente al manifiesto original ni al modelo activo.
- Criterio de calidad: ≥95 % píxeles espectralmente válidos, bloqueo SCL ≤25 %, proxy no vegetado SCL5 ≥50 %, y cuatro bandas red-edge QA95. El proxy no certifica litología.

**Restricción:** con solo 30 grupos potenciales y una adquisición real válida no existe el supuesto doble de muestras independiente. Primero revisar calidad y distribuir nuevos sitios geográficos, particularmente la franja 20°S–30°S, donde el desempeño actual es débil.

## 4. Brazos experimentales y sobreajuste

| Brazo | Familias | Cambios controlados | Comparación |
|---|---|---|---|
| E0 — v3 | Las 6 | 126 variables, registros OOF del 13/11 | Baseline archivado |
| E1 — v4 fijo | Las 6 | 170 variables, hiperparámetros de cada fold v3 congelados | Efecto aislado de 44 variables |
| E2 — v4 anidado | Las 6 | 170 variables, nueva búsqueda Optuna por TRAIN interno | Competición completa |
| E3 — sitios nuevos | Las 6 | 170 variables + nuevos grupos solo en TRAIN, mínimo diez aceptados, embargo 5 km frente a TEST | Efecto aislado de muestras nuevas |
| E4 — multitemporal/geología | Las 6 | **Pendiente:** cambios estacionales y una categoría litológica amplia | No implementado |
| E5 — M1→M2 aplicado | Las 6 | **Pendiente:** ternas reales generadas por M1 | No confundir con ranking de ternas OOF |

Overfitting de RF ya comprobado en estudio previo: AUC TRAIN aparente ≈0,991 vs OOF espacial ≈0,645. No se repite como si fuera hallazgo nuevo. El estudio previo probó ganancia DN ±3 %, KBest48 y embargo 5 km: la ganancia artificial **perjudicó AUC**; por ello **no se activa** la aumentación como remedio automático. Los seis conservan sus controles de regularización y CV anidada, sin reducir familias.

## 5. Contraste contra release y documentación incluso si falla

E1 y E2 puntúan las mismas 706 observaciones y folds espaciales que el histórico. Además comparan contra el **procedimiento CatBoost del release de septiembre reconstruido en validación cruzada**, nunca contra el binario publicado evaluado sobre sus propios ejemplos de TRAIN. Se registran AUC, AP, Brier, falsas alarmas, positivos no detectados, brechas cuando corresponda, intervalos exploratorios por grupos y selección descriptiva de tres sitios.

El generador de informe **m2_v4_document_result.py** se niega a crear conclusiones si no han finalizado los seis algoritmos. Una vez ejecutados, guarda CONCLUSION_CIENTIFICA.md con resultados reales, ventajas/regresiones, hipótesis y decisión de no promoción automática.

## 6. Comandos reproducibles (CMD desde la raíz del clon)

Activar Python de D:
```cmd
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"
```

Primero adquirir todas las nuevas bandas para las 706 originales. El proceso es reanudable desde caché. Si quedan fallos de descarga o QA, no entrenar:
```cmd
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_rededge_v4_build --cohort 706
```

Solo cuando cohort_706/build_status.json indique complete:
```cmd
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_rededge_v4_competition --cohort 706
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_v4_document_result --phase fixed
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_rededge_v4_nested --cohort 706
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_v4_document_result --phase nested
```

Para los sitios reales nuevos, fuera del menú (la prueba de su efecto requiere al menos diez aceptados y el E1 terminado):
```cmd
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_v4_new_sites
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_v4_new_sites --fetch 5
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_v4_new_site_effect --min-new 10 --buffer-km 5
"D:\Librerias\Python312\python.exe" -m scripts.scientific.m2_v4_document_result --phase new_sites
```

Todos los temporales, descargas, matrices, documentos de resultados y cachés quedan en el disco D. Los resultados y checkpoints de E1/E2/E3 se guardan en results/scientific_simulations/m2_rededge_v4_20261113.

## 7. Alternativa de cierre: candidato M1 ampliado + release M2 anterior

Si E1/E2/E3 no muestran una mejora útil o demostrable en M2, **no es necesario inventar un ganador**: conservar el CatBoost M2 de septiembre (training_20260920_195933_m0300) y estudiar la pareja con el candidato ampliado M1 (training_20261106_111653_m0300). Ambos artefactos están presentes en el clon.

**Estado verificable al 13/11:** el registro activo sigue apuntando a los releases de septiembre para **ambos** modelos. El M1 ampliado de noviembre existe como candidato, pero no ha sido activado ni publicado. Su mejora exploratoria en 1.004 casos emparejados no sustituye su propia validación independiente. No interpretar la existencia de ambos archivos como que el sistema ya está usando esa combinación.

Antes de adoptar esa pareja se debe probar de forma integrada que la salida de M1 permite seleccionar las tres zonas, que el M2 del release puede procesar la imagen de cada zona con su propio esquema v3 de seis bandas y que la decisión del mejor candidato no pierde calidad frente a M1 solo. **No modificar modelos activos ni crear release combinado sin aprobación explícita**. Si es viable, cerrar con un registro honesto de las limitaciones de generalización del M2.

## 8. Límites, criterios de decisión y tareas pendientes

No declarar ganador antes de obtener métricas reales. No confundir bootstrap sobre folds de desarrollo con validación externa independiente. No activar M2 v4 por subir AUC unas milésimas sin recuperar positivos y estabilidad por región. Si los seis algoritmos v4 no superan de forma consistente el protocolo previo, el release CatBoost original continúa activo y el resultado queda registrado como **experimento que no confirmó mejora**, igualmente válido científicamente.

**Pendiente:** completar 703 parches históricos nuevos, obtener diez o más sitios nuevos con QA y etiquetas provisionales declaradas, ejecutar E1 y E2, analizar E3, revisar datos externos y, especialmente, validar la aplicación real de M1→Top3→M2→Top1 antes de publicar modelo nuevo. No hubo push ni promoción de releases.
