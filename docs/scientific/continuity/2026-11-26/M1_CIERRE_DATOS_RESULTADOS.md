# M1 — cierre experimental de composición y auditoría de comparabilidad

Fecha: 26/11/2026. Rama feat/m1-targeted-data-refinement. Run 20261126T213014_fc01460c, commit ff4a4a5a7adff25b11d7895903eed1d91b26297f.
85/85 pruebas focalizadas en 27.13 s, 15 fits nuevos, cinco políticas completas. No equivale a suite completa posterior a expansión.
ZIP Evidencias(9).zip SHA256 92136668fa1a1599b4d09366f6dd814d1568acfe360019f307b9db53c063ea1a; manifiesto SHA256 929732600e49a8da72ba506729f16a2b7a48195e0fa9189695d2474079762e46, nueve hashes de salidas recalculados y coincidentes. Ruta results/generalization/model_1_data_closure/20261126T213014_fc01460c. Conservar los originales privados.

## Resultados verificados desde predicciones

| Política | RMSE medio (ppm) | R² medio | MAE medio (ppm) | RMSE global | RMSE >40 | Sesgo >40 |
|---|---:|---:|---:|---:|---:|---:|
| A histórico | 7.074911 | .631231 | 3.892598 | 7.100761 | 24.888575 | -16.247821 |
| B SMOTER archivado | 7.230849 | .615639 | 3.958293 | 7.268343 | 25.569325 | -17.093513 |
| C1 solo cola | 7.122205 | .626578 | 3.938920 | 7.148209 | 25.126888 | -16.552906 |
| C2 transferencia ponderada | 7.053262 | .633404 | 3.864404 | 7.076868 | 25.093647 | -16.729418 |
| D transferencia + SMOTER | 7.212146 | .617283 | 3.957060 | 7.242107 | 25.437573 | -16.760593 |

16315 OOF (5 x 3263), claves/verdad/fold/grupo iguales entre políticas, TRAIN y TEST disjuntos por identidad y cubren cohorte congelada. C2 gana descriptivamente RMSE medio (-.021649 ppm, ~.31%), R² (+.002173) y MAE (-.028194), mejora 4/5 folds. Desviación estándar muestral entre folds C2 RMSE .649776, R² .021075; A .680944/.024471. RMSE <=40 A 5.085539, C2 5.006692; >40 C2 empeora .205072 ppm (~.82%). Sesgo de cola más negativo, -16.729418. El 4.14% de muestras de cola representa 52.019% del SSE de C2 (A 50.828%). No cumple guarda de cola ni RMSE objetivo. IC descriptivo bootstrap pareado por 242 grupos, 5000 remuestreos, seed42, delta RMSE global C2-A [-.145338,.102405], incluye cero; no prueba independiente ni significancia concluyente.

| Banda ppm | TEST n | RMSE C2 |
|---|---:|---:|
| <=5 | 81 | 6.436208 |
| (5,10] | 688 | 5.215797 |
| (10,20] | 1565 | 4.132656 |
| (20,40] | 794 | 6.106624 |
| (40,60] | 95 | 14.223006 |
| (60,80] | 21 | 30.391237 |
| (80,100] | 11 | 42.731085 |
| >100 | 8 | 57.346322 |

## Auditoría de composición

15399 membresías nuevas (7692 C2, 7692 D, 15 C1), no duplicados por fold/política/ID, cero grupos candidatos en TEST del fold. 1588 IDs distintos en C2 y D, cuatro en C1. Pesos individuales respetan .10/.25; suma añadida por banda <=10% masa histórica; conteos de cobertura y pesos agregados coherentes. Masa total C2 por fold 123.10/127.00/124.95/127.35/126.65 frente a 2610–2611 históricas (aprox. 4.7–4.9%); masa de cola .5/1/.75/.75/.75. No son 7692 mediciones nuevas independientes. 175 sintéticos y 175 registros de linaje con padres dentro de TRAIN, 28/41/44/26/36; objetivo coincide con interpolación de las etiquetas históricas a precisión numérica, valores 40.341615–116.830032 ppm. Bandas sintéticas 145/20/4/6 en >40–60/60–80/80–100/>100; suma de los 10 óxidos en 175 entre 94.682312 y 99.788130. No se verificó interpolación de todos los predictores contra los originales de los padres en chat ni se certificó método/CRS. Ninguna fila nueva aprobada; cero modelos guardados.

## Auditoría de métrica y comparabilidad

Revisión directa de Memoria Litio (2)(5).pdf, sección 7.3.3, 7.7.1, 7.7.2 y tabla 15 (p.111–112 PDF). OASIS: 2635 brutas, 787 Li válido, 681 tras recorte percentil central ~95% y SUM(no water) 94–102. Li en ppm (mg/kg de roca), no porcentaje de acierto. Antes de filtros media 19.006, SD 10.823, min 3.361, max 100.574. No tomar esa SD inicial como dispersión final. RF VCA 5x5, 100 trials por fold exterior, RMSE media 5.361 ± .595 y R² .499 ± .086 publicados. Nuestro código core/scientific/regression_metrics.py usa sqrt(mean_squared_error) y r2_score, ambos en escala original de Li. Promedio de cinco RMSE, no RMSE global; cálculo coincide con definición de memoria. HGB cierre con parámetros fijos, folds aleatorios históricos reutilizados, 3263 TEST, 242 grupos, rango real 4.27–123.8 ppm (límites congelados 4.2439–126.12); no VCA independiente del proceso de desarrollo ni misma población que OASIS.

En cualquier fold con varianza no nula y métricas estándar, RMSE = SD poblacional(y) * sqrt(1-R²). A partir de los pares publicados (RMSE/R²) 4.327/.605, 5.500/.543, 5.341/.346, 6.136/.500, 5.503/.500, se deduce SD poblacional aproximada por fold 6.885, 8.136, 6.604, 8.678, 7.782 ppm (redondeo de tabla). Nuestro SD poblacional global 11.700950; por fold aprox 12.621,10.492,10.821,12.928,11.424 ppm (SD muestral 12.630,10.500,10.829,12.938,11.433). Esto confirma diferencia de dispersión, no descompone causalmente la brecha ni certifica comparabilidad. RMSE/SD por fold de OASIS equivale a sqrt(1-R²), ~.628,.676,.809,.707,.707; media ~.706. C2 media ~.6053; global .60481. No es evidencia nueva independiente de R². Como ilustración, para RMSE global 5.361 con nuestra SD global se requeriría R² global ~.79008; NO es umbral de R² medio ni reemplaza la meta del RMSE medio. C2 global R² .634203. Brecha RMSE medio 1.692262 ppm; reducción adicional ~23.99%. No confundir RMSE en ppm con error relativo porcentual.

Comparación numérica descriptiva: C2 supera R² publicado pero no RMSE; las dos distribuciones y protocolos son distintos, no declarar superioridad integral. El contraste válido requiere el Excel original OASIS y, si disponibles, folds/OOF originales y reglas de limpieza, exclusión de coincidencias con entrenamiento propio. Si se consigue, realizar benchmark separado con población y particiones comunes; si no, reportar las cifras como referencias descriptivas. No eliminar extremos ni recortar retrospectivamente la cohorte por error para obtener un número menor. RMSE/SD complementario solo descriptivo.

## Decisión final

Fases 5 y 6 cerradas operativamente: no repetir búsqueda masiva, balanceos ni campañas de ajuste. Objetivo numérico completo NO alcanzado. C2 conserva el mejor resultado descriptivo académico, A es control conservado y preferible bajo la guarda de cola y trazabilidad. No concatenar candidatos al dataset productivo; 0 aprobados. Fase 7 no ejecutada: falta evidencia favorable según guardas, equivalencia/validación independiente y autorización; no promover ni alterar características productivas. Prioridad si se abre una nueva etapa: obtener cohorte/folds OASIS comparables o un conjunto independiente representativo, no prolongar hiperparámetros con los mismos TEST. El resultado de dominio espacial completo permanece separado; M2 y main intactos. No hay nuevo comando necesario para este cierre.
