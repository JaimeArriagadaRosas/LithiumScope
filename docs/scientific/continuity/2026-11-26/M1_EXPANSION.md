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


## Validación real del EDA y siguiente auditoría (26/11)
Run 20261126T145129_ee7f5431, commit a23a627. Seis pruebas focalizadas aprobadas, 12.37 s. RAR entregado con nueve hashes de salida coincidentes. Nueve particiones originales verificadas según inventario local; ninguna con columna inequívoca de método Li ni CRS, todas con CITATIONS. El manifiesto no registra publicaciones en las 3420 filas históricas bajo el campo source_publication; no afirmar separación bibliográfica demostrada.
- 319 convergentes: 264 con soporte vecinal, 249 también con OOD marginal <=.25.
- 6393 otros dominios: 3357 con soporte vecinal, 3185 también con OOD marginal <=.25 (Li 0–307 ppm). Estos conteos son descriptivos sobre la referencia completa y no determinan elegibilidad.
- 69 altas intraplaca: ninguna con soporte vecinal. Evaluar exclusivamente transferencia si se acredita procedencia; no mezclar con compatibles.
- 4230 excluidas, 221 copias exactas. 2990/2998 candidatas andinas con alerta coordenada/Li; 2988 con alerta aproximada. Las 14 andinas >696 ppm tienen ambas alertas, todas referencia [30009], nueve superan QC. No inferir duplicación confirmada contra la cohorte preparada hasta distinguir coincidencias dentro de candidatos y con base. Las 48 altas félsicas excluidas no superan QC.
- Geologycal_age, Arc y Domain faltan en todas las candidatas preparadas. No inventar categorías.
- Los métodos y CRS desconocidos no se deducen del nombre interno Li_icpms. GEOROC documenta prioridad de métodos en compilados (https://georoc.mpch-mainz.gwdg.de/georoc/CompRules.html), pero no certifica que cada Li sea ICP-MS. Referencias de originales: https://georoc.eu/georoc/cite.asp; inventario de metadata/citas: https://doi.org/10.25625/4EZ7ID.

Nuevo comando, sin descarga ni fit:
`python -m lithiumscope.experiments.model_1.expansion.provenance --run results/model_1/expansion/20261126T145129_ee7f5431`
Recupera secciones bibliográficas locales si existen; registra coincidencias exactas de coordenada/Li con base bruta y pertenencia a base preparada, concordancia química y nombres de muestra; conserva todas las coincidencias para revisión. No autoriza automáticamente ninguna muestra. Implementación pendiente de prueba real. Los resultados permitirán concentrar revisión en publicaciones y muestras nuevas, sin repetir EDA ni descargar datos. Antes de fit, decidir explícitamente si procede un brazo de sensibilidad con metadatos no certificados separado de los estrictamente verificados, sin confundir evidencia exploratoria con validación de release. M2 congelado.


## Procedencia validada y campaña de sensibilidad
ZIP M1_provenance: cuatro hashes coincidentes; 6474 registros de pie, 5334 identificadores bibliográficos únicos (sin conflictos de texto por identificador). Las 11232 candidatas vinculan todas sus referencias a su partición. 3211 candidatas tienen clave histórica exacta: Andean Arc 2990 (2750 coinciden con base preparada), Granite/Rhyolite 221 (205 con base preparada). Las 14 andinas >696 ppm coinciden con muestra, química y Li históricos; nueve preparadas y cinco excluidas por QC. Publicación 30009: Ramirez-Briones et al., 2025, Macusani, doi 10.1093/petrology/egaf041. Las 69 altas intraplaca no coinciden por clave histórica exacta y tienen 13 referencias. Los 6781 registros de estados de revisión favorables tienen referencias resueltas, QC favorable y fuente verificada, sin clave histórica exacta. Esto no equivale a aprobación estricta.

La campaña opcional de sensibilidad permite cuantificar el aporte de estos datos sin afirmar método analítico ICP-MS ni CRS certificados. Requiere consentimiento explícito mediante flag. No modifica ledger estricto, aprobación de producción, M2 ni release. Lee también Geochemistry Reference de la base histórica y compara tokens de DOI, identificadores explícitos y texto exacto cuando disponibles; independencia bibliográfica no certificada. La reserva por hash se estratifica por estado y bloquea grupos y publicaciones vinculadas. D4 filtra otros dominios por química y rango marginal dentro de cada TRAIN externo. No se usa el EDA global para seleccionar filas por similitud. Se mantienen hiperparámetros archivados, seis brazos fijos, 30 fits exteriores como máximo y ninguna selección interna en esta primera prueba de efectos; no se ajustan hiperparámetros ni se guardan modelos. B–F balancean la base TRAIN a 2700 equivalentes antes de añadir las filas. Brazo sin extensión se registra como bloqueado en ese fold. TEST históricos íntegros, resultados por banda, fuente, grupo, fold, bootstrap pareado RF, reserva de desarrollo y manifiesto con hashes. La reserva ya fue examinada descriptivamente y no es independiente. No elegir release en base a esta campaña; una etapa posterior exige revisión estricta y selección interna con evaluación nueva.

Ejecutar pruebas focalizadas y, si el usuario acepta esta limitación:
`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.sensitivity --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata`
La salida está en results/generalization/model_1_expansion_sensitivity. No repetir EDA, adquisición ni campañas anteriores. Implementación y cuatro pruebas nuevas pendientes de ejecución local.


## Resultados sensibilidad y refinamiento de transferencia
Run 20261126T145931_b132b506, commit c240a9b; 10 tests focalizados aprobados, 3.76 s. ZIP SHA256 e077795a061b5a434c8ae85ce4ca8eb10822ac3473207c64cf150748fc040849; manifest SHA256 6ce065c46143061388baf03ce92eedf47877abab63d98f353747a70f10e8e366. 14 hashes verificados, 30 fits, cero errores, 20.520 predicciones históricas, identidades pareadas y A reproduce RF archivado (diferencia máxima 1.42e-13 ppm). 6781 elegibles exploratorias: 4611 pool, 1483 reserva, 687 cuarentena. Referencias históricas identificables mediante tokens en 676 filas; independencia bibliográfica no certificada.

| Brazo | RMSE global | MAE global | R2 global | RMSE medio folds |
|---|---:|---:|---:|---:|
| A |152.302214|15.303595|.131184|82.310168|
| B |151.648396|15.300003|.138627|81.431717|
| C |152.807342|15.214643|.125411|82.221456|
| D |146.698532|16.106772|.193940|90.295179|
| E |143.027716|16.159153|.233776|90.811562|
| F |134.461658|15.618894|.322807|87.592351|

F reduce RMSE global 11.71% frente a A y 15.88% frente al SVM publicado, pero aumenta MAE .315299 ppm y RMSE medio. IC95 agrupado de diferencia F-A RMSE [-46.7653,57.3129], MAE [-3.593754,4.886718]. Fold 5 RMSE 338.14 ->272.74; fold 3 32.35 ->124.16, tres casos del grupo g_151_219 explican 93.56% del incremento neto de SSE del fold 3. No excluir ni seleccionar reglas usando ese grupo o los valores reales de TEST. Nueve fuera del soporte histórico: RMSE 2932.72 ->2364.68; restantes 3411: 23.74 ->58.08; Li <=100 ppm 13.34 ->11.81. El máximo TRAIN fold 5 aumenta de 696 a 6201.47 ppm. Reserva 1483: RMSE 351.74 ->186.61, MAE 136.27 ->63.97, R2 -3.445 ->-.251; sigue siendo exploratoria y bajo desempeño. No autorizar release ni afirmar superioridad integral.

Siguiente subetapa acotada, sin repetir A-F: G = F sin alta transferencia; H/I/J = F con peso .10/.25/.50 de las filas intraplaca de alta concentración (incluidas todas sus copias rare_group_sqrt). El resto conserva peso 1. Se mantienen RF, folds, hiperparámetros archivados, balanceo histórico y reglas de admisión. Los pesos son hipótesis predeclaradas de sensibilidad, no estimaciones de incertidumbre; selección futura exclusivamente interna. 20 fits exteriores fijos, no búsqueda de hiperparámetros, no promoción ni guardado de modelos. G prueba el aporte de expansión amplia sin las altas; H/I/J buscan reducir transferencia negativa. Reutilizar evidencias de A-F como controles. No declarar candidato hasta verificar mejora global, MAE y estabilidad de folds. Si no hay equilibrio, probar adaptación condicionada por química y/o otras familias con selección espacial interna. Mantener evaluación OASIS separada; no inferir cumplimiento de 5% a partir de RMSE espacial.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.sensitivity --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata --transfer-refinement`

Salida: results/generalization/model_1_expansion_transfer. Cinco tests adicionales pendientes de ejecución local; M2 intacto.


## Refinamiento de transferencia validado y selección interna
Run 20261126T150549_0d4c067c, commit 04eea06, 15 tests focalizados aprobados (2.64 s). ZIP SHA-256 639b088aab81c1cd6ebbf56eca97594588a2c16477502a12fab7e9916e651d45, manifest c450c1cafabec26ffe965d01cf6d3996de5685af90d7661e4f3050c55107014f. 14/14 hashes correctos, 20 fits, cero errores, 13680 predicciones históricas emparejadas; 4611 pool, 1483 reserva, 687 cuarentena.

| Brazo | RMSE global | MAE global | R² global | RMSE medio fold | RMSE fold 3 | RMSE fold 5 | RMSE reserva |
|---|---:|---:|---:|---:|---:|---:|---:|
| G |153.522892|15.363838|.117201|83.101273|32.934700|340.795228|343.261820|
| H |131.794905|14.717127|.349402|80.258887|76.002089|283.675442|169.870789|
| I |126.408032|14.755666|.401499|79.112755|87.236046|267.783005|172.139102|
| J |128.521598|15.079126|.381317|81.264395|95.194140|270.060891|175.821952|

I mejora RMSE global 17.00% frente a A, 20.92% frente a SVM. H menor MAE y mejor reserva (R² -0.036727; no positivo). H/I/J mejoran RMSE global, MAE y RMSE medio frente a A, pero conservan transferencia negativa en fold 3. RMSE banda (100,696] A=128.9188, H=236.8785, I=267.1393, J=290.2585. RMSE <=100 A~13.34, H=11.6233, I=11.4744. IC95 agrupado delta RMSE H-A [-39.9921,24.7446], I-A [-51.9937,31.7965], J-A [-49.9322,36.9746]; MAE también incluye cero. G empeora RMSE con IC95 [0.2162,2.1138]. No eliminar casos del fold 3 ni escoger reglas por sus valores de TEST.

Próxima subetapa K: B (balanceado sin extensión), H e I como expertos; 15 políticas predefinidas, incluida B; combinaciones convexas .25/.50/.75/1 del experto y límites de discrepancia .5/1/2 veces max(predicción B, Q75 del TRAIN histórico, 1 ppm). No usar verdad, ubicación o identificación del caso para enrutar. Selección por OOF interno espacial: minimizar RMSE global sujeto a no empeorar MAE, RMSE medio entre folds, RMSE <=100 y RMSE (100,696] frente a B; al menos 5 filas por banda para permitir cambio. Si no cumple, B. Criterios son guardas exploratorias, no tolerancias instrumentales. Por cada fold interno se bloquean grupos/publicaciones identificables de TEST externo y validación interna, y recalcula D4 desde TRAIN interno. Reserva congelada previa. 75 fits internos (3 x 5 x 5) y 5–10 externos (B y solo el experto seleccionado si procede). No repetir A–J completos. Hiperparámetros archivados seleccionados previamente en TRAIN externo, por tanto los resultados internos no son una nueva búsqueda de hiperparámetros independiente; TEST externo es el mismo desarrollo reutilizado. Registrar evidencia de decisiones, predicciones y trazabilidad. Sin guardado ni promoción.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.nested --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata`

Salida results/generalization/model_1_expansion_nested. Cinco pruebas nuevas pendientes de ejecución. Después de K, decidir si conservar la estrategia o cambiar de familia (CatBoost/HGB) en vez de prolongar la búsqueda de pesos RF; ejecutar evaluación académica con protocolo y población explícitos, sin equiparar RMSE espacial global con OASIS. Certificación analítica/CRS y evaluación independiente continúan pendientes. M2 congelado.


## Selección interna K validada; transición a comparación de familias
Run 20261126T151830_b320b6a8, commit 703d789. 20 tests focalizados aprobados (2.70 s). ZIP SHA-256 e09907520197260b954a44e5632d0252cfbe50bfdb878ba228f6e61f7a8ad928; manifest SHA-256 d1248db1f5fb29a694094a4c25f68d21749ce5b43665ecbbd5442e0a6c386056. 16/16 hashes correctos, 83 ajustes (75 internos, 8 externos), 205200 predicciones internas, 6840 predicciones externas pareadas, 30 particiones, 83 trazas. Estado complete; plan interno histórico verificado. K RMSE global 152.449623, MAE 15.468376, R² .129501, RMSE medio 84.319444; B 151.648396, 15.300003, .138627, 81.431717. Delta RMSE K-B .801227 IC95 [-.249450,8.805731], MAE .168372 IC95 [-.206317,.794316]. K mejora reserva frente a B (RMSE 287.559 vs 377.397, MAE 115.540 vs 149.131, R² -1.971 vs -4.117) pero no alcanza H/I ni la estabilidad histórica. Selección externa 1=B, 2=H completo, 3=H cap1, 4=I blend25, 5=B. En fold 5 interno B/H/I RMSE 30.352/24.734/22.321, MAE 9.557/8.999/8.740, RMSE <=100 9.543/10.284/10.876; RMSE (100,696] 201.622/157.566/136.680. No hay observaciones >696 en el TRAIN histórico de fold 5. Guarda de error bajo bloquea transferencia. K no se selecciona; no relajar retrospectivamente restricciones para favorecer TEST conocido.

Siguiente comparación fija: CatBoost e HistGradientBoosting, cada familia B (histórico balanceado), H (D4 peso alto .10), I (D4 peso alto .25). 30 fits exteriores (15 por familia), mismos datos, reserva y folds. Hiperparámetros y políticas archivados por familia/fold; no retuning. Se usa preprocesador canónico de la política, incluida exclusión de coordenadas para CatBoost cuando corresponde, límites de objetivo y ponderación de copias raras. Comparar vs referencia archivada de cada familia y su B, además de RF H/I y release bajo identidad OOF. Tres tests nuevos, dos de fit sintético real (CatBoost opcional) y uno de contratos. Pendientes de ejecución local. No repetir A-K. Resultados no independientes ni aprobados para producción.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.sensitivity --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata --algorithm catboost`
`python -m lithiumscope.experiments.model_1.expansion.sensitivity --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata --algorithm hist_gradient_boosting`

Salida results/generalization/model_1_expansion_families/<algorithm>/<run>. Después: comparar familias, ejecutar puente académico sobre población y protocolo de piloto OASIS congelados (no equiparar RMSE espacial global), decidir si procede optimización interna acotada o finalizar investigación. No guardar ni promover modelos; M2 congelado. Criterio académico 5.629 ppm es provisional y no equivale a 5% de error relativo individual.


## Comparación de familias finalizada y objetivo académico confirmado
Commit de ejecución 6f0ad0f. 23 pruebas focalizadas aprobadas (3.36 s). CatBoost run 20261126T152757_7ad534c0: 15 fits, 15 hashes verificados, ZIP SHA256 75b21460573cf71a7a570694f0feb946c19e83b733bbea6d600bf1f5ad885564, manifest 1dca4d376dd665f34261c49ead11ee2f9436f74425e9ebd62c8e73567e814787. HGB run 20261126T153320_8ee569e5: 15 fits, 15 hashes verificados, RAR SHA256 37301edfcf28aaf7347dcfe8bed1f0e2b79d098820de9d06004c9ffdac4a95ac, manifest 59cdd3826d6ff5060b90a3ad56e42e145b6e2b6b75adcee3ec2c5b6f7f3d7b08. Sin errores; 10260 predicciones históricas por familia. HGB identidades emparejadas y métricas recalculadas.

| Familia/brazo | RMSE global | MAE global | R² global | RMSE medio fold | RMSE reserva | R² reserva |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost B |144.328269|16.079038|.219778|80.800935|189.298338|-.287421|
| CatBoost H |111.579554|14.910078|.533679|68.506516|162.662352|.049393|
| CatBoost I |119.933031|15.534761|.461243|74.518678|162.701469|.048936|
| HGB B |150.039667|15.455950|.156805|82.476928|291.686034|-2.056737|
| HGB H |129.832405|14.554447|.368633|74.593312|168.114315|-.015398|
| HGB I |133.256904|15.364124|.334888|81.296095|165.073248|.021006|

CatBoost H lidera RMSE y R² global espacial; HGB H lidera MAE global entre brazos de expansión; HGB I obtiene MAE reserva 47.565051. HGB H fold 3 RMSE 52.298, fold 5 284.796. IC95 delta RMSE H-B CatBoost [-59.994947,8.432960], HGB [-37.473596,7.794419]; no afirmar superioridad concluyente. Mantener ambos candidatos exploratorios; no guardar ni promover.

El usuario precisa que basta superar las métricas publicadas en la memoria de referencia, no reproducir el trabajo ajeno. Consultada directamente Memoria Litio (2)(5).pdf, biblioteca, sección 9.1.1 tabla 15 (p. 112 del PDF): RF, RMSE medio 5.361 ± .595 ppm, R² medio .499 ± .086. Criterio académico vigente: RMSE medio <5.361 y R² medio >.499. La tolerancia previa de 5% no es necesaria para demostrar superación numérica. No confundir RMSE global con media de folds ni afirmar superioridad científica directa sobre datos distintos.

Puente académico implementado: conserva las 3263 filas del piloto 20261126T123613_24ef3f6c, su rango congelado (recorte previo QC 4.2439–126.12 ppm), seed 42 y KFold aleatorio 5; verifica manifiesto e identidades OOF del piloto y sus métricas. Tres familias con hiperparámetros fijos predeclarados, dos brazos (histórico completo del TRAIN académico y expansión compatible en rango, sin altas intraplaca), 30 fits. No usar hiperparámetros archivados de CV espacial, pues pueden haber visto los TEST aleatorios. No hay ajuste ni búsqueda interna en esta campaña fija. Preservar reserva y cuarentena de datos nuevos, bloqueos de grupo y referencia respecto a candidatos en TRAIN. El KFold histórico permite solapamiento espacial, documentado. No volver a recortar evaluación para favorecer resultados. Si ambos umbrales se superan, registrar cumplimiento numérico académico sobre esta población separada; todavía no es evidencia independiente ni aprobación de release de dominio completo. Si no, valorar optimización interna acotada orientada al RMSE académico. Tres pruebas nuevas pendientes de ejecución local.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.academic_bridge --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata`

Salida results/generalization/model_1_expansion_academic. No repetir campañas previas. M2 congelado.


## Puente académico validado; refinamiento HGB
Run 20261126T153931_23ea0e09, commit 142e339. 26 pruebas focalizadas aprobadas (3.12 s). SHA256 manifiesto 603d44eb4e49d948e16820f9ad7e08edf1cf1c8d00653030d57d4fdd91fc4ea4; cinco hashes de salidas verificados. 30 fits; 19578 predicciones, 3263 identidades únicas por brazo, mismos grupos/verdades/folds. Train histórico 2610–2611, expansión 1729–1788 por fold (1844 candidatas únicas en unión). Ningún ID histórico TRAIN/TEST compartido; grupos espaciales históricos sí compartidos por diseño aleatorio. Reserva de 1483 excluida.

| Familia | RMSE medio histórico | R² medio histórico | RMSE medio expandido | R² medio expandido |
|---|---:|---:|---:|---:|
| RF |7.678787|.566651|7.762007|.557885|
| CatBoost |7.521075|.585015|7.953079|.534899|
| HGB |7.291410|.607571|7.490458|.586865|

Ninguna configuración alcanza ambos umbrales de memoria (RMSE <5.361, R² >.499). Las seis superan R²; mejor RMSE HGB histórico, requiere reducción adicional 26.48% del RMSE medio. Expansión sin ponderación deteriora las tres familias en este dominio; no extrapolar esta conclusión al dominio completo, donde CatBoost H lidera RMSE. HGB histórico global RMSE 7.319892, MAE 4.054506, R² .608648. En su OOF: 135 observaciones >40 ppm (4.14%) explican 48.7970% del SSE, sesgo -16.7808 ppm; 40 >60 explican 37.1441%; 8 >100 explican 16.3815%, media real 109.11875 y predicha 50.29118 ppm. 50 errores cuadrados mayores concentran 47.4662% SSE. Mantener todos los casos; no modificar población ni filtrar por errores.

Refinamiento acotado: HGB y siete perfiles (control mediana; nativo NaN; nativo min_leaf=5; nativo min_leaf=5 y peso cola 2/4; último con peso cola 2 y transferencia compatible .10/.25). Umbral de cola Q90 del TRAIN histórico de cada ajuste; normalizar pesos históricos, atenuar nuevos datos, sin sobremuestreo. Reusar hiperparámetros del puente excepto min_samples_leaf. Selección interna KFold aleatorio 3, seed 42+fold externo: minimizar RMSE medio sujeto R² medio >.499, control si ninguno cumple. 105 fits internos y 5 finales, sin volver a ajustar controles externos. Validar hashes puente y piloto, mismos TEST históricos de 3263 filas, recorte sin cambios. En cada TRAIN interno recalcular compatibilidad y bloquear grupos/publicaciones identificables de ambas particiones excluidas; conservar reserva y cuarentena. Se reutilizan TEST externos previamente examinados; es desarrollo, no validación independiente. Sin promoción ni guardado de modelos. Cuatro pruebas nuevas pendientes. Si no alcanza el objetivo, no repetir sin límite las variaciones HGB; considerar auditoría de predictores adicionales y mayor información geoquímica útil, o reportar la brecha con la memoria y diferencias de dispersión. M2 congelado.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.academic_refinement --run results/model_1/expansion/20261126T145129_ee7f5431 --allow-unverified-metadata`

Salida results/generalization/model_1_expansion_academic_refinement.


## Refinamiento académico HGB: cierre de tratamientos y auditoría de predictores
Run 20261126T161005_a8008a65, commit 23332ad; 30 pruebas focalizadas aprobadas (23.58 s). Manifiesto SHA256 da784283574b77281c0bee1ef85b80b9a4da0843ab7d26b584497092ac814a18. Ocho hashes de ocho archivos recibidos coincidentes; falta partitions.json (hash esperado b7d9f2def5646068a4039b82564feaa4cf65cf99036216336dec2c76d20a5c1b), no afirmar verificación de los nueve. 110 ajustes: 105 internos, cinco externos; 91364 predicciones internas, 3263 externas únicas. No solapamiento de IDs de validación interna con TEST externo correspondiente. Métricas externas recalculadas; HGB seleccionado RMSE medio 7.2250622237, R² medio .6126280133, MAE medio 4.0533042578; RMSE global 7.2493640069, R² global .6161535047, MAE global 4.0531226632. HGB anterior RMSE medio 7.2914101137 y R² medio .6075714237. Reducción RMSE .91%, falta reducir 25.80% para superar 5.361 ppm; R² supera .499. Selección: fold1 mediana, fold2 transferencia .10 (1788 nuevas filas), fold3 nativo fino, fold4 nativo, fold5 nativo fino. 135 filas >40 ppm explican 48.9034% SSE; ocho >100 ppm RMSE 57.052922 vs 59.833645 anterior. No retirar filas. Promedio descriptivo de los 15 splits internos: nativo fino 7.761189; nativo 7.764491; transferencia .10 7.789880; cola x2 7.797451; mediana 7.807991; transferencia .25 7.856758; cola x4 7.964521. No repetir búsqueda de pesos con los mismos predictores.

Siguiente etapa: inventario de las 118 columnas preparadas de M1, con población académica de 3263 y los 33 predictores canónicos intactos. Se verifican fuente, piloto y predicciones del refinamiento. Lista explícita de elementos químicos distintos de Li con sufijos de medición; nombres sin unidades y edad solo revisión. No usar ID, referencias, alias de objetivo, salidas predictivas, totales ni residuos como predictores. Se inventarían mediciones adicionales y razones químicas (seis razones ya exploradas se mantienen solo como referencia, otras nuevas y A/CNK molar sin corrección de Ca por apatita), con cobertura, valores válidos y asociaciones descriptivas por fuente, banda y TRAIN externo. Las correlaciones sobre población completa/residuos son diagnóstico retrospectivo, no selección válida de predictores. Antes de un fit futuro confirmar significado, unidades, independencia de la medición, disponibilidad en predicción, armonización de fuentes y recalcular selección dentro de TRAIN interno. El inventario no revisa los campos descartados por adaptadores en los CSV nuevos ni demuestra disponibilidad de nuevos predictores en producción; no añadir automáticamente columnas, no entrenar ni descargar. No modificar M2. Tres pruebas nuevas pendientes de ejecución.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.feature_audit`

Salida results/generalization/model_1_expansion_feature_audit. Si no hay cobertura informativa, evaluar adquisición/enriquecimiento de mediciones independientes antes de nuevas búsquedas. Mantener criterio académico numérico y protocolo separado de dominio completo; no reducir población para alcanzar un RMSE.


## Inventario de predictores validado y recuperación de originales
Run 20261126T162414_9f0900a7, commit 94d74bb. 33 pruebas aprobadas (5.07 s). ZIP SHA256 2a76161ee61e7162fd65d8ab66fd2e322fc8686aa2a43d591ee07245a8ff1640; manifiesto 3c4ff8ddb1e9dc5004c951ae9cdeb1e10cad1bed11c311c47166831f04bc91d6; cinco hashes coincidentes. 118 columnas, 33 predictores canónicos, 38 columnas adicionales de medición (algunas elementos repetidos con otro método), 16 variables derivadas. Las 38 solo presentan cobertura en Mamani09 (674/3263); GEOROC (2589/3263) no presenta ninguna. La cobertura de La ICP-MS es 674/674 Mamani09; de las 135 observaciones >40 ppm solo 25 presentan La; entre las ocho >100 ppm solo una. No asumir que ausencia de columna equivale a falta de medición original. El contrato de adquisición histórica solicita únicamente LI y los elementos canónicos; el adaptador conserva un subconjunto explícito. Razones de cobertura amplia: Cs/Sr 83.05%, Rb/Sr 98.59%, A/CNK molar 99.97%; asociaciones Spearman con Li .4989/.4846/.3798 (descriptivas, no prueba de beneficio incremental). Edad aparece con doble espacio (Age  (Ma), 173 valores presentes), fuera del alias del inventario; hay también isótopos Sr/Nd/Pb con cobertura escasa, sin revisión de unidades/procedencia. No incorporar automáticamente.

Se implementa raw_recovery: primero busca particiones históricas nombradas en la base y verifica SHA contra auditoría de ingestión; luego reutiliza las tres particiones precompiladas Andean Arc, verificadas contra inventario y manifiesto de expansión anterior. Recupera solo elementos nuevos con encabezados explícitos ppm/ppb (conversión ppb/1000 y misma política de censura); no inferir ICP-MS, no fusionar con XRF/ICP-MS de Mamani. Exigir coincidencia única de nombre de muestra normalizado, coordenadas, Li y al menos cinco predictores independientes concordantes (tolerancia numérica estricta), rechazar filas ambiguas/reutilizadas y coincidencias múltiples entre particiones; no rellenar discrepancias de medición entre original y precompilado. Puede perder correspondencias válidas por diferencias de compilación; reportar en lugar de relajar automáticamente. Recuperar valores como tabla lateral por ID histórico, sin cambiar dataset, objetivos, población académica, esquema ni M2. No descarga ni fit. Cobertura descriptiva total, por concentración y TRAIN. No afirmar método analítico certificado. Tres pruebas nuevas pendientes. Si cobertura útil, diseñar comparación de predictores con selección interna y verificar disponibilidad/compatibilidad en inferencia; si no, evaluar adquisición dirigida. Objetivo académico permanece RMSE medio <5.361 y R² >.499, sin recortar TEST.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.raw_recovery`

Salida results/generalization/model_1_expansion_raw_recovery. No repetir EDA ni entrenamientos previos.


## Corrección de recuperación: reutilizar correspondencias de procedencia
Run 20261126T162820_54d9553f, commit e524bfd. ZIP SHA256 cdb089c807affbe4a59560f1b67386bea32f6ba37f34beaaa399f64c79e5fd5d, manifiesto SHA256 3911c5b0abd24a7fda6b8cf7d47cb88242b1d370525cc2d45e3795c6f8102730; seis hashes correctos. El comando de pytest quedó concatenado al SET de PYTHONPATH: no consta ejecución de las tres pruebas de raw_recovery. Recuperación completada con cero filas aceptadas. CSV histórico presente (SHA256 40ee961450b55f708909e5f90ba3bbb686255d3595a16507fb7d09b66c5d104e), hash no verificado contra auditoría; no autorizar su uso. Los cuatro CSV tienen 42 encabezados explícitos adicionales; no afirmar cobertura por encabezado. Tres particiones Andean verificadas; 8232 comprobaciones (2744 por partición) clasificadas sample_unmatched antes de contraste químico. Posible causa: normalización de nombre anterior no contempla Sample alternativo, puntuación/sufijo de referencia; la auditoría de procedencia sí lo hace.

Reanalizada evidencia histórica M1_provenance: 3296 aristas con clave exacta coordenadas/Li para 2743 filas preparadas GEOROC, 2692 aristas concordantes por nombre normalizado y química para 2689 filas; 2686 aristas uno a uno, eliminadas seis aristas ambiguas (tres filas históricas). Las 2686 tienen 12–22 predictores químicos compartidos en auditoría previa. No equivale a recuperación validada. Nuevo linked_recovery verifica hashes de expansión, procedencia, originales y fuente, reutiliza enlaces uno a uno y posiciones originales, rearmoniza las filas enlazadas, comprueba nombres con la misma normalización, claves de coordenadas/Li, concordancia estricta con vista candidata y tolerancia histórica ya documentada (máximo .001 o 2% por elemento, mínimo cinco, hierro excluido del nuevo contraste por diferencias de representación). La tabla lateral conserva IDs y valores adicionales ppm, no incluye Li como predictor. Inventaría también cobertura real en cada CSV original; no usar métodos no certificados como equivalentes a ICP-MS. Sin adquisición, entrenamiento, cambio de esquema o M2. Dos nuevas pruebas y tres anteriores sin ejecución confirmada; no repetir recuperación fallida.

`set "PYTHONPATH=%CD%\src"`
`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.linked_recovery`

Salida results/generalization/model_1_expansion_linked_recovery. Esperadas 38 pruebas. Revisar cobertura efectiva antes de nueva campaña; no repetir ciclos indefinidos de entrenamiento.


## Recuperación enlazada validada; ablación académica de predictores
Run 20261126T163242_d5f5c5fe, commit c6550da. 38 pruebas aprobadas (3.30 s). ZIP SHA256 3482eb676667d3256ddac51d1e90630169b828dba3b540e429344ad242ed8cac, manifiesto f8469364dc93133a4fe23edcd385844a426f4d19e94ccf950d6e58365ccd7ce8; seis hashes verificados. 2686 enlaces aceptados (859/1031/796 por partición), sin reutilizar candidato o posición, seis aristas ambiguas excluidas. 11–21 elementos concordantes en nueva comprobación química (excluye Fe). 3420 filas históricas intactas; 2532 enlaces académicos, 2531 con alguna medición nueva. 58708 valores recuperados sobre 3263 filas académicas, 42 columnas, 40 con datos, 22 cobertura >=50%. Y 2482; La 2432; Sc 2432; Sm 2437; Ce 2425; Yb 2419; Ni 2377. Entre 135 casos >40 ppm Y cubre 104, La/Ce/Nd/Sm/Gd/Yb 96, Sc 94, Ni 92. No afirmar método analítico certificado. Enlaces usaron concentración Li para confirmar identidad retrospectiva, no es predictor ni procedimiento disponible en inferencia; cualquier mejora es exploratoria, puede reflejar selección por disponibilidad de enlace. Validar recuperación con identificadores independientes de Li antes de producción.

Nueva ablación academic_features: HGB native_fine fijo, cuatro brazos (canónico, razones, mediciones, ambos), 60 fits internos y 20 externos. Además una quinta serie de predicciones externas seleccionada por RMSE medio interno sujeto a R² >.499, sin fit adicional. Cobertura mínima 5% y diez grupos con observaciones, variación no constante calculadas solo en TRAIN de cada ajuste. Mantener mediciones Mamani ICP-MS/XRF separadas de GEOROC ppm sin método certificado. Razones de química conocida, A/CNK molar y razones REE dentro del mismo bloque de medición. No utilizar Li, IDs, referencias, residual ni etiquetas de procedencia como nuevos predictores. Verificar fuente, manifiesto y seis salidas de recuperación, hashes de las tres particiones originales, membresía, IDs del piloto y predicciones previas. No añadir filas ni modificar TEST, esquema productivo, M2 o release. Evaluación aleatoria reutilizada; diferencias de métodos y población frente a OASIS permanecen. No elegir un ganador por comparación externa retrospectiva: los brazos fijos son diagnóstico, la política seleccionada internamente es evaluación del procedimiento. Cuatro pruebas nuevas pendientes de ejecución. Meta RMSE medio <5.361 ppm y R² medio >.499. Si la expansión de predictores no aporta una reducción significativa, no continuar indefinidamente con el mismo enfoque.

`python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.academic_features --allow-unverified-metadata`

Salida results/generalization/model_1_expansion_academic_features.


## Ablación de predictores validada y campaña de cierre académico acotada
Run 20261126T163711_3f9dd35a, commit e5c322f. 42 pruebas aprobadas (3.41 s); 80 ajustes, 16315 predicciones, seis hashes verificados. Manifiesto SHA256 ecb7314e6a4b7ad370d715af7e326c3bd557c06e3284655ec19f3c9f022a044f.

| Brazo | Predictores externos | RMSE medio | MAE medio | R² medio |
|---|---:|---:|---:|---:|
| Canónico |33|7.239729|4.025695|.611629|
| Razones |56|7.250854|4.025934|.611965|
| Mediciones |107|7.206212|4.002265|.617685|
| Combinado |130|7.074911|3.892598|.631231|
| Selección interna |variable|7.085133|3.915786|.628802|

La ampliación combinada reduce RMSE medio 2.28% frente al control; 135 filas >40 ppm concentran 50.83% de SSE del combinado. Diferencia RMSE global combinado-canónico IC95 por 2000 remuestreos de grupos: [-.436,.110] ppm; no concluyente. Varianza objetivo de población académica 136.91 ppm²; RMSE global 5.361 exigiría aproximadamente R² global .7901 (no equivalencia exacta con media de folds). No reducir población, no usar Li como predictor ni confundir métricas académicas con espaciales. Todavía no alcanza RMSE medio <5.361; sí supera R² >.499. Mantener cautela por recuperación retrospectiva que empleó Li para corroborar identidad y por métodos no certificados.

Última campaña acotada propuesta para este conjunto y esta evaluación: academic_final. Contrastar HGB combinado con CatBoost combinado (perfil académico fijo, NaN numéricos nativos), y un estimador local de analogías basado solo en química canónica (log1p, escala robusta TRAIN, cobertura >=50%, mínimo ocho variables compartidas), distancia geográfica haversine con escala fija 50 km y cinco vecinos ponderados. Excluir vecinos de identidad coincidente y coincidencias casi idénticas a <=100 m según química, sin inspeccionar Li de evaluación; si menos de tres vecinos válidos, usar HGB. No garantiza eliminar todas las duplicaciones ni demostrar generalización espacial; el KFold aleatorio histórico comparte grupos. Registrar vecinos y exclusiones. Cuatro mezclas convexas fijas (tres pares al 50%, mezcla de tres a tercios). Siete políticas y una seleccionada exclusivamente por validación interna de tres folds, minimizando RMSE sujeto a R² >.499 (fallback HGB). Reutilizar cinco predicciones externas HGB combinadas archivadas tras verificar hashes y columnas; 35 ajustes nuevos de árboles y 20 construcciones del estimador local. Sin ampliar filas, sin modelo definitivo, sin promoción, sin M2. Cinco nuevas pruebas pendientes de ejecución local.

`git pull --ff-only && set "PYTHONPATH=%CD%\\src" && python -m pytest tests/scientific/m1_expansion -q`
`python -m lithiumscope.experiments.model_1.expansion.academic_final --allow-unverified-metadata`

Salida results/generalization/model_1_expansion_academic_final. Si no se alcanza el objetivo, cerrar esta secuencia de refinamientos sobre la población histórica y documentar brecha; no prolongar indefinidamente la optimización sobre folds ya examinados. Antes de promover M1 enriquecido, verificar correspondencia sin Li, métodos y disponibilidad de predictores en inferencia, evaluar protocolo espacial completo y conjunto independiente. Decisión posterior de release separada del cumplimiento numérico de la comparación OASIS.
