# M1 — nueva línea aprobada: aumento de datos y adquisición dirigida

## Objetivo y límites

El usuario aprobó el plan de siete fases: auditar tratamientos, adaptarlos al esquema enriquecido, evaluar sintéticos, investigar fuentes, adquirir/EDA/armonizar, comparar reales y sintéticos, optimizar y entrenar si cumple. Esto reabre el laboratorio de datos, no la búsqueda indefinida de hiperparámetros ni el release. M2 congelado. Comparación académica: RMSE medio <5.361 ppm y R² medio >.499. Referencia exploratoria HGB combinado: 7.074911 ppm, .631231, MAE 3.892598; 3263 TEST históricos, cinco folds aleatorios previamente inspeccionados. Validación espacial separada de 3420 filas; no promover por una sola comparación. Conservar cohorte, reserva y evidencia anterior. No generar puntos de evaluación sintéticos.

## Fase 1: auditoría del código

- `experiments/generalization/resampling.py`: rare/group/rare_group_bootstrap; SMOTER conservador con mismo origen, sitio, grupo, categoría y patrón de faltantes, Q90, límite por progenitor y pareja. Recalcula cuatro derivados originales, pero considera razones adicionales como mediciones si se le pasa el esquema enriquecido. No aplicar sin adaptación. SMOTE binario es para clasificación, no M1.
- `experiments/model_1/targeted/policies.py`: KDE sobre log1p del target, ponderación inversa moderada/fuerte, pesos fuente-grupo, remuestreo cuantílico, transformaciones. No confundir el control de representación con nuevas mediciones reales.
- `expansion/academic_treatments.py`: pesos de cola Q90, imputación/nativos, transferencia atenuada; ensayos ya realizados sobre esquema anterior.
- `expansion/academic_features.py`: mediciones y razones, admisión por cobertura TRAIN >=5%, diez grupos, variación. Mantener bloques analíticos separados.
- No existe SMOGN ni WERCS completos en los módulos inspeccionados. Las variantes nuevas son inspiradas en estas técnicas, no reproducciones exactas. Referencias: Branco et al. 2017, https://proceedings.mlr.press/v74/branco17a.html ; Branco et al. 2019, https://www.sciencedirect.com/science/article/pii/S0925231219301638 .

## Fases 2–3: campaña de tratamientos enriquecidos

Nuevo módulo independiente `experiments/model_1/augmentation`; no cambia producción. Seis brazos: control HGB combinado; copias de cola (>40 ppm) dos por fila; KDE moderado existente; copias por relevancia continua entre Q90 TRAIN y 40 ppm, balance por grupo y máximo cuatro copias por padre (inspirado en WERCS, solo sobremuestreo, sin undersampling); SMOTER enriquecido; interpolación con perturbación gaussiana acotada en trazas (inspirada en SMOGN, no SMOGN original). Presupuesto máximo de filas añadidas igual a dos veces el número de filas reales >40 ppm en cada TRAIN. Los métodos sintéticos pueden producir menos o cero; si cero, reutilizar control, no relajar reglas. Comparar cantidades efectivas; no afirmar que idénticos presupuestos equivalen a dosis efectivas iguales.

Reglas sintéticas: mismo sitio exacto, grupo, fuente, categorías, patrón faltante canónico; al menos ocho predictores químicos canónicos observados, al menos 80% de solapamiento de mediciones observadas en ambos padres respecto de su unión; distancia RMS robusta log1p <=1.5, distinta de cero, máximo cinco vecinos, diferencia Li <=max(20 ppm, .4*menor Li). Padres de cola >40 ppm exclusivamente TRAIN. Máximo cuatro usos por padre contando ambos roles y dos por pareja, rotación de grupo. Interpolación convexa con lambda [.1,.9] para oxidados y objetivo; ruido gaussiano de escala 10% de diferencia de padres solo en trazas, truncado a sus envolventes. Conservar faltantes cuando falta cualquiera de los padres. No inventar sitio/categorías ni afirmar que las observaciones sintéticas cumplen un ensayo de laboratorio o QC de originales. No extrapolar. Recalcular cuatro derivados canónicos y todas las razones enriquecidas desde mediciones generadas; no interpolarlas directamente. Mantener métodos separados, no usar Li como predictor. Los umbrales son hipótesis conservadoras fijadas antes de correr la campaña, no límites de validez física certificados.

Protocolo: reutilizar cinco controles externos archivados HGB combinado, verificar fuente/piloto/recuperación/evidencia y esquema. 90 tratamientos internos (5 x 3 x 6), 30 externos (5 x 6), máximo 115 fits nuevos tras reutilizar controles exteriores, menos si no existe soporte sintético. Selección interna requiere mejor RMSE que control y R² >.499, o control; 7 series externas incluyendo política seleccionada sin fit adicional. Guardar predicciones, métricas de bandas, conteos reales/sintéticos, trazas de progenitores, semillas, particiones, decisiones y hashes. Sin cambiar hiperparámetros HGB. Los folds aleatorios conocidos, la vinculación retrospectiva asistida por Li y los métodos sin certificar impiden tratar estos resultados como validación externa independiente.

Ejecución (CMD):

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q
python -m lithiumscope.experiments.model_1.augmentation.academic --allow-unverified-metadata
```

Salida `results/generalization/model_1_enriched_augmentation/<run>`. Ocho nuevas pruebas previstas (55 total), no ejecutadas localmente. Comparar con control 7.074911, .631231; verificar linaje, rendimiento de cola y métricas agregadas. Si hay pocos pares, documentar falta de soporte en lugar de inventar observaciones. No repetir todas las campañas anteriores.

## Fase 4: fuentes reales, revisión preliminar de metadatos

| Fuente | Identificador / enlace | Evidencia pública | Decisión |
|---|---|---|---|
| Meixner et al. (2020) | https://doi.org/10.1007/s00126-019-00915-2 | Li en rocas volcánicas cenozoicas y basamento andino; página del editor enlaza ESM1 DOCX y ESM2–4 XLS | Alta prioridad: inspeccionar suplementos, coordenadas, química, permisos y deduplicación; separar litologías/materiales. No se verificó aún contenido de tablas. |
| Álvarez-Amado et al. (2022) | https://doi.org/10.1016/j.gexplo.2022.107062 | Li e isótopos de ignimbritas del norte de Chile; rocas volcánicas 20–50 ppm según resumen, además de sedimentos/aguas | Alta prioridad: obtener tabla original y comprobar predictores completos y cobertura 40–126. Excluir sedimentos/aguas de cohorte roca total salvo dominio separado. Acceso a suplementos no confirmado. |
| Fernández-Turiel et al. (2020) | https://doi.org/10.20350/DIGITALCSIC/10527 | Repositorio abierto CC BY 4.0, XLSX de roca total shoshonitas de Puna y compilación; XRF y HR-ICP-MS | Revisar solo tabla roca total, presencia de Li no confirmada; descartar si falta objetivo; no asumir independencia de la compilación. |
| Abat et al. (2025) | https://doi.org/10.60520/IEDA/113778 | EarthChem Library id 3778, XLSX 59.49 KB, CC BY 4.0; Macusani, roca volcánica/intrusiva y mena sedimentaria | Riesgo de solapamiento conocido con Ramirez-Briones et al., DOI 10.1093/petrology/egaf041, ya presente en auditoría GEOROC. No contar como independiente; revisar solo filas nuevas compatibles. |
| EarthChem/PetDB | https://earthchem.org/petdb | Sistema de consulta geoquímica, fuentes federadas pueden incluir GEOROC | Canal adicional, no dataset validado; exportar roca total Andes con Li y química, revisar duplicación por referencia e identidad. |

No descargar indiscriminadamente ni contabilizar nuevos casos desde metadatos. Esta es una investigación inicial, aún sin tablas crudas inspeccionadas ni nuevas filas aprobadas. Adquisición dirigida posterior: conservar originales y hashes, licencia y cita, tabla de método/unidad, coordenadas/CRS, litología/material, QC, fuente/ID/publicación. Comparar contra histórico, candidatos, reserva anterior y bibliografía, incluyendo deduplicación independiente de Li. EDA por fuente y banda con cobertura (especialmente 40–126 ppm), grupos/publicaciones nuevas, química y faltantes; mantener 10–40 ppm y el dominio completo en evaluación. Congelar reserva genuinamente independiente y protocolo antes de ajustar. Fase 5 no iniciada.

## Fases 6–7: diseño aprobado, pendiente evidencia

A: reales históricos. B: históricos + sintéticos TRAIN. C: históricos + nuevas filas reales elegibles. D: históricos + nuevas reales + sintéticos TRAIN. Reusar controles, seleccionar tratamientos internamente, comparar mismos TEST, errores por banda y grupos; no tratar sintéticos como observaciones independientes. Solo avanzar al fit definitivo si evidencia y contratos de inferencia lo respaldan, con validación espacial y prueba independiente; no promover por superar cifras en poblaciones distintas. Full suite posterior a cambios pendiente. M2 no se toca.


## Fase 3 validada; avanzar a adquisición dirigida
Run 20261126T180155_005d9a59, commit 8e4c9d1. 55/55 tests (3.52 s). RAR SHA256 8d8b7487fb78b5048dfb36dced7cd56177443310e587efa90a8bcfe86cc43761; manifest SHA256 d5e7f6cb8f6afce7913a8a1cb9275889525137516e7386e85029674935a215fe. 10/10 hashes, 115 fits nuevos, 120 tratamientos, 22841 predicciones externas (7 x 3263) alineadas por ID/fold/grupo/verdad; 78312 internas; 20 particiones, 7335 registros de linaje y ningún padre registrado fuera del TRAIN correspondiente. No se han verificado mediciones sintéticas directamente (no se exportaron). Métricas recalculadas.

| Política | RMSE medio | R² medio | MAE medio | RMSE global >40 ppm | RMSE global <=40 ppm |
|---|---:|---:|---:|---:|---:|
| Control |7.074911|.631231|3.892598|24.888575|5.085539|
| SMOTER |7.230849|.615639|3.958293|25.569325|5.185766|
| Selección interna |7.319484|.605633|3.982114|25.844258|5.260398|
| Gaussiano |7.348474|.602499|4.013787|25.859887|5.292169|
| Copias cola |7.392674|.594996|4.033167|25.747261|5.366148|
| Copias relevancia |7.440143|.591892|4.038865|26.135125|5.376133|
| KDE |7.717502|.562479|4.145926|26.900469|5.617366|

SMOTER +2.20% RMSE vs control. 175 filas sintéticas en cinco ajustes externos (28/41/44/26/36) de presupuesto 1080, 106 parejas contadas por fold, no únicas globalmente. Gaussiano 173. En 20 particiones SMOTER 427, gaussiano 428, copias cola 3240, copias relevancia 3240. Selección interna gaussiano/gaussiano/SMOTER/SMOTER/copias. SMOTER baja RMSE 40–60 de 14.424811 a 13.795802 pero aumenta >100 de 57.971611 a 62.649335. No supera control ni umbral OASIS; no concluir que todo SMOTER/SMOGN sea inútil, solo estas variantes y presupuestos en esta población. Conservar brazo B negativo; no repetirlo por defecto. Objetivo permanece abierto.

## Fase 4–5: adquisición e inventario implementados
Confirmados enlaces ESM1–4 de Meixner en https://link.springer.com/article/10.1007/s00126-019-00915-2 . Los binarios no pudieron descargarse en entorno de chat (fallo DNS/cache). Artículo de suscripción; no eludir autenticación. Álvarez-Amado (https://www.sciencedirect.com/science/article/pii/S0375674222001200) bloqueó lectura de suplementos; repositorio U. Mayor confirma 20–50 ppm en rocas, no confundir con sedimentos. Fernández-Turiel: https://epos-msl.uu.nl/data-publication/0aa9848d06477eaee50b61f3aa829b64 informa métodos y tablas; Li no confirmado. Macusani https://ecl.earthchem.org/view.php?id=3778 licencia CC BY 4.0, posible duplicación conocida; no asumir independencia. Del Bono et al. 2026 DOI 10.1007/s00126-026-01428-5 suplemento XLSX de ubicación/descripción y DOCX de figuras de lixiviación, no tabla química completa confirmada; no priorizar descargas.

Nuevo `augmentation.source_intake`: descarga explícita de cuatro suplementos Meixner a `data/raw/model_1/research/directed/meixner_2020`, sin reemplazar archivos. Crea directorios para tres fuentes manuales. Verifica firma/formato, límites, SHA256, inventaría hojas/tablas, dimensiones y celdas que mencionan Li/unidades sin inferir orientación ni número de muestras. DOCX tablas y párrafos; XLS requiere xlrd; PDF manual. No extrae automáticamente objetivos, métodos ni aprueba filas. Errores explícitos; permite repetir inventario sin descarga. Los originales locales y licencias requieren revisión antes de compartirlos; no subirlos al repositorio público. Cuatro pruebas nuevas pendientes. Próximo paso: revisar originales y fijar mapeo por fuente, deduplicación independiente de Li, QC y EDA real. No entrenar todavía. M2 congelado.

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q
python -m pip install "xlrd>=2.0.1,<3"
python -m lithiumscope.experiments.model_1.augmentation.source_intake --download
```
Salida `results/generalization/model_1_directed_source_intake`. Esperadas 59 pruebas. Adjuntar evidencias del inventario y, si es permitido, originales para mapear hojas. No repetir tratamientos. Cero filas nuevas aprobadas.


## Fase 5: adquisición Meixner validada, mapeo pendiente
Run 20261126T195310_b4a13662, commit e63fdb9195346547db6d07abb7c537b49163e583. 59/59 pruebas focalizadas (31.48 s); xlrd 2.0.2 instalado localmente. Manifiesto SHA256 c2bb62201978dda79f66be707268df11d9d580a3bf32241fbca9fd7f2f8ec709; tres archivos de inventario recibidos y sus hashes verificados. Cuatro originales descargados e inventariados, 0 filas aprobadas. Hashes de originales registrados por ejecución local, no recalculados aquí porque no se recibieron originales:

- ESM1 DOCX 113116 bytes, SHA256 2070c8e531c40ef2816621e3d786e626c43a4c622d69344df50d12ac4a00272e. 45 párrafos, información metodológica/figuras.
- ESM2 XLS 52224 bytes, SHA256 ffb7171170cfb383bc47ccc4f86f6aa7f1197f08c9a4bcdfed9e65a14efffbe1. Sheet1 133x14, 886 celdas no vacías. Tabla A1, composición isotópica Li/Sr/Nd, concentración Li y SiO2; fila 9 col 7 Li [µg/g] (equivalente ppm en masa), col 5 δ7Li. No asumir 133 muestras ni cobertura química completa.
- ESM3 XLS 35840 bytes, SHA256 7e172ef76fb1387aa6a264ed9bce62b9a4f5e8c25a5f725c192e3b6bb195767d. Final 54x14, 151 celdas no vacías; fila 7 cols 10/14 Li [µg/g], referencias y materiales de control posibles (Li carbonate, blanco). No interpretar estándares como roca ni combinar columnas sin revisar.
- ESM4 XLS 29184 bytes, SHA256 155c7c5a44c090a0a99726c49cc247e60df551ad907296507a861e31ae557286. Sheet1 27x12, 115 celdas no vacías; dos hojas vacías. Li [ng] en fila 9 cols 7/12 expresa masa, no concentración; no convertir a ppm sin masa de muestra y contexto experimental.

La tabla A1 puede aportar concentraciones pero el inventario no prueba suficientes predictores (>=8 mediciones químicas, QC de óxidos, coordenadas) para M1. Priorizar mapeo por identificador de muestra, referencia y química no Li contra histórico y particiones originales, rechazar ambigüedad, no asignar química por proximidad. No usar isótopos de Li como predictores. Necesarios originales ESM1–4 para revisar tablas y método; conexión remota de cuenta principal sin dispositivo disponible. Solicitar originales permitidos en el chat, no publicar en GitHub. Sin nuevo código/fit ni repetición de intake por esta fase.

Alternativas confirmadas: Fernández-Turiel (https://digital.csic.es/handle/10261/197644) descarga únicamente Shoshonites_Puna-geochemistry-0_metadata.xlsx y Shoshonites_Puna-geochemistry-1-whole_rock_this_work-compilation.xlsx, CC BY 4.0, cobertura Li por verificar; el registro EPOS https://epos-msl.uu.nl/data-publication/0aa9848d06477eaee50b61f3aa829b64 enumera elementos analizados, no confirma Li. Macusani (https://ecl.earthchem.org/view.php?id=3778) archivo 3778-1_EC_BulkSample_Analyses_TorroetalMACUSANI.xlsx, CC BY 4.0; comprobar duplicación con publicación ya incluida, no contar como fuente independiente. Álvarez-Amado DOI 10.1016/j.gexplo.2022.107062: acceso a tabla original sigue pendiente. Compilación Wörner/Mamani (https://doi.org/10.5880/digis.e.2024.005) incluye datos publicados hasta 2009 y Mamani 2010; no contabilizar automáticamente como nueva por solapamiento con Mamani09. Evitar salmueras, sedimentos y minerales aislados para este modelo de roca total. Aún no se descargaron estas alternativas. M2 congelado.
