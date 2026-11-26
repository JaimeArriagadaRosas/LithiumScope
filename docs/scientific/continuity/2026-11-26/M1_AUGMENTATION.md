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
