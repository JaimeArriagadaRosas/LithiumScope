# Continuación M1 — expansión con datos reales

## Decisión vigente
El 26/11 se reabre la investigación de M1 en esta rama. La decisión anterior de no incorporar las fuentes corresponde al cierre de las campañas previas; no constituye una renuncia a estudiarlas. M2 queda congelado; no entrenamiento, promoción, release ni merge. Se preservan todos los modelos y datos históricos.

Base: suite b5b059415bf4c9ec2f45ed12f3cac516db1dc171, run 20261126T143344Z_d13f9d7678c449f1825f87304e2c897c, 1005 tests, 1002 aprobados, 0 fallos/errores, 3 omitidos, cobertura 57.05%, 236.703 segundos, Git limpio. La suite no valida los cambios posteriores.

## Objetivo y límites
Mejorar M1 frente al RF actual y al release SVM en los mismos 3420 TEST históricos, con 244 grupos y cinco folds. Investigar el aporte de datos reales nuevos, sin repetir las campañas 45/21. Siete observaciones concentran 95.82% del SSE histórico; ninguna debe eliminarse del TEST. Las fuentes candidatas ya fueron descargadas y examinadas (2998 y 8234 filas armonizadas). OASIS es una comparación de protocolo diferente: objetivo provisional RMSE medio <=5.629 ppm y R² >=.499 bajo condiciones comparables; tolerancia 5% respecto a 5.361 ppm. No confundir con 5% de error relativo individual ni declarar equivalencia sin datos/protocolo común.

## Plan
1. EDA avanzado: recuperar metadatos disponibles de fuentes originales locales, resolver copias exactas con identidad fuerte, mantener conflictos para revisión, diagnosticar química y disponibilidad del esquema fijo de 33 predictores.
2. Preparación D0 (base), D1 (+convergentes), D2 (+altas transferencia), D3 (+ambas), D4 (+otros dominios compatibles). La preparación crea planes, no modifica la cohorte ni aprueba muestras de producción.
3. Experimento RF, A control, B balanceo 2700 equivalentes, C–F B con las cuatro extensiones. Congelar hiperparámetros, mantener TEST e impedir coincidencias con validación externa e interna. Balancear la base histórica TRAIN, añadir nuevas filas únicamente de TRAIN permitido. Establecer previamente presupuesto de ajustes internos y evaluación reservada. Reportar aportes por brazo y criterios de elegibilidad; si no hay nuevas filas aptas, no ejecutar brazos vacíos.
4. Solo si la extensión aporta, optimización de algoritmo y procedimiento; RF, CatBoost, HGB, XGBoost y otras familias según resultados. Selección dentro de TRAIN y evaluación por banda, fuente, fold, RMSE/MAE/R², sesgo e IC agrupados. Criterios cuantitativos de parada antes del fit.
5. Fit final y evaluación de elegibilidad de release M1, sin promoción automática. M2 no se modifica.

## Primera implementación (fases 1–2)
`python -m lithiumscope.experiments.model_1.expansion.runner`
Desde CMD en la raíz: `set "PYTHONPATH=%CD%\src"`. No descarga ni entrena. Usa el EDA existente y exige hashes de la cohorte y folds; vuelve a preparar de manera determinista X sin cambiar la fuente. Los originales se buscan en data/raw/model_1/research/<source_key>/<filename>; si faltan, se registran como no verificados. No deducir método de una columna Li(PPM) ni CRS de coordenadas válidas. Se inspeccionan encabezados y valores específicos de Li/CRS cuando son inequívocos.

- row_audit.csv: identidad, estado original y posterior a resolución, disponibilidad, metadatos, perfiles de dominio y partición.
- candidate_features.csv: identidad y predictores deterministas según el esquema congelado, sin ajuste aprendido.
- summary.csv, coverage.csv: conteos y cobertura; distancia química multivariada descriptiva, no elegibilidad definitiva.
- raw_metadata_inventory.json: particiones y hashes; encabezados relevantes y columnas detectadas.
- publication_review.csv y review_template.csv: evidencias que requieren revisión.
- fold_membership.csv y fold_counts.csv: candidatos autorizados para cada TRAIN exterior, por cohorte, sin modificar TEST.
- manifest.json: hashes y límites; si no hay revisión positiva, estado review_required.

El ledger opcional (`--review RUTA.csv`) exige candidate_id, raw_source_sha256, decision (compatible/experimental_transfer/reject), method_evidence, crs_evidence, reference y notes. Una decisión positiva requiere fuente local verificada, cribado favorable, coordenadas preparadas válidas y referencias explícitas. No presentar el ledger como certificación automática; contrastar la referencia y el método antes de aprobar. Las altas intraplaca requieren decisión experimental_transfer. Ninguna fila se aprueba para producción.

Se reserva por hash determinista hasta 20% (redondeado hacia abajo, mínimo uno si >=5) de grupos nuevos elegibles que no comparten publicación conocida histórica. Filas vinculadas por publicación al conjunto reservado se ponen en cuarentena. Las fuentes ya fueron exploradas; la reserva no es un holdout externo independiente. D4 requiere similitud química vecinal y fracción marginal OOD <=.25 frente al TRAIN de cada fold; son umbrales de investigación, no límites geológicos instrumentales. El control de la publicación solo es posible si el registro histórico la contiene. En fase 3 repetir restricciones y ajuste de soporte también dentro de cada fold interno.

## Estado
Código de preparación y tests sintéticos incorporados; falta ejecución local de pruebas y EDA sobre datos reales. Revisión bibliográfica todavía no efectuada. No ejecutar entrenamiento antes de evaluar salidas y completar la elegibilidad. No repetir las pruebas históricas de comparación ni regenerar PDF M2 para este avance. Las fases 3–5 no están implementadas todavía en esta nueva línea de expansión.
