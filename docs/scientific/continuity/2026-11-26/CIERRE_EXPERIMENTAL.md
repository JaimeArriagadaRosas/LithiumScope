# Cierre experimental aprobado — 26/11/2026

## Decisión
Conservar los dos Random Forest guardados el 25/11: M1 `training_20261125_115932_m0300_826123c1` y M2 `training_20261125_144535_m0300_6fa5d269`. El usuario aprobó no ejecutar las fases 3 y 4: no incorporar candidatos GEOROC ni realizar un fit final adicional. No es una aprobación de release, promoción ni merge. La decisión no certifica superioridad estadística ni dominio externo.

## Evidencia local recuperada
- Restauración: 10 archivos copiados y 4 ya verificados mediante el restaurador; tres archivos adicionales (OOD, auditoría física, inventario STAC) recuperados desde Scientific-Lab y aceptados por posprocesamiento. No se reentrenó.
- Comparación: `20261126T141852_5f6cc5d4`, estado global complete. SHA-256 del manifiesto entregado: `7cf6a5920a0874a9eb8bc2763d00f233028a179665da22c3997fb4e3f993d7a1`.
- M1: 3420/3420 identidades y folds coincidentes, 244 grupos; RMSE 159.8463 -> 152.3022; MAE 16.3204 -> 15.3036; R² .0430 -> .1312. Ganancias con IC95 agrupado (1000 remuestreos): RMSE 7.5441 [-.5274,12.3886], MAE 1.0168 [-.00085,2.3375], R² .0882 [-.0549,.2612].
- M2: 706/706, corte fijo 21.6 ppm, 66 grupos. Referencia reconstruida (no artefacto publicado): AUC .6274 -> .6570; AP .3700 -> .3969; Brier .1859 -> .1842; BA .5400 -> .5669. Los IC95 agrupados de AUC/AP/Brier (5000 remuestreos) incluyen cero. Métricas nativas publicadas/candidato: AUC .5617/.6660, AP .2765/.4004, Brier .2144/.1822; protocolos propios, comparación descriptiva.
- Diagnóstico M2 completo: 706 históricos, 0 provisionales; QA espectral 690 aprobados y 16 retenidos en TEST pese a no aprobar QA; 566 proxy óptico SCL aceptados, 140 abstenciones, 25 sitios con etiquetas contradictorias, 9 OOD >25%, 64 escenas de la cohorte y 111 en inventario original de 783. No certifica exposición de roca de campo.

## Fase 2: auditoría reproducida
Run `20261126T142237_76e00f95`; ZIP privado `LithiumScope_candidate_audit.zip` SHA-256 `dc8439fbd12b7aab2899b517f72cccc8456096aa0a37ad38e5792817c25c54b3`. Los tres CSV coinciden con su manifiesto:
- row_screening.csv: 2a42ffcaa6b5971eb0a5668315e0a31499e99ecc5c19548c17e43d63730d5ee8
- summary.csv: d787144d903e5e52f38fcf7c57fd6739be43283c48c74ebd882566cbf1b3fb64
- domain_shift.csv: 31ba70a4d8feb3b5e50d9fd63fe586688b5ba2b421492cde89702e4265fcf687

| Estado | Filas |
|---|---:|
| Margen convergente | 319 |
| Transferencia alta | 69 |
| Otros dominios | 6393 |
| Excluidas en cribado | 4451 |
| Total | 11232 |
| Aprobadas | 0 |

Margen convergente: Li 1.534–517 ppm, mediana 25; 73 grupos nuevos. No aporta ejemplos por encima de 696 ppm. Transferencia alta: 69 muestras intraplaca, 14 grupos nuevos, 53 >1000 ppm, mediana 1512.362 y máximo 6201.467 ppm. Las medianas de SiO2/MgO/CaO de referencia bruta son 60.10/2.67/5.36; convergente 74.0346/.2864/1.451; transferencia alta 72.52/.05/.37. Se observan desplazamientos de composición, no únicamente del objetivo. No se verificaron publicaciones originales, método analítico ni CRS; no se aprueba ninguna fila. Cierre de la fase 2 como decisión de no incorporación, no como certificación de las fuentes. Futuro: evaluación de transferencia y procedencia independiente.

## Software y reporte
Suite local en commit `1cd365920c9f3f2fc7604de7dea1a783256e47db`: 1004 totales, 1001 aprobadas, 0 fallos, 0 errores, 3 omitidas, cobertura 57.02%, código 0, 2:09, Git limpio; run `20261126T135021Z_9fef752595a24d18817227676b5cf16d`.
Revisión estática de omitidas: las dos pruebas históricas de compare_vnext_trained_runs se refieren a ejecuciones de 05–06/11 (M1 20261105_204029 / 20261106_111653; M2 20261105_223618 / 20261106_135029), distintas de la comparación actual. No sustituirlas con artefactos de 25/11. La tercera exige manifiesto local vnext_expanded y matriz v4 783x170, extensión no usada por el candidato. Mantener skips condicionales legítimos; no afirmar ejecución de esas tres. No repetir entrenamientos para eliminarlos.

PDF previo a recuperación, run `demonstration_20261126_105426_m0300_c04bd5968e22`, 37 páginas, SHA-256 `6533e379d25587836096131fa9f43624302a9214f6feb4ab717649deb29edcae`. Página 2 con ambas etapas y tres RGB, prioridad exploratoria A1015-29; mayor Li observado A1015-22. M1 RMSE 1.9461, MAE 1.4994, R² .9301; M2 AUC .25, AP .1964, BA .5, F1 0. No usar demo de diez casos para seleccionar modelos; independencia no certificada. Falta render posterior a recuperación y reconstrucción real.

## Próximos pasos (sin fit)
1. Actualizar rama con git pull --ff-only. No ejecutar otra comparación si el manifiesto complete permanece disponible.
2. Menú 1 > 2: generar PDF integrado con ambos RF y evidencias restauradas. Guardar PDF y prediction_manifest.json; comprobar página 2, trazabilidad de los dos modelos, anexo de OOF emparejada y referencia reconstruida, piloto OASIS, diagnósticos históricos y gráficas. Si hay omisión, corregir el enlace a evidencias, no los datos.
3. Verificar reconstrucción desde artefactos congelados y registrar resultado; ejecutar pruebas focalizadas si hay cambios funcionales, suite final si se modifica código.
4. Consolidar defensa: mejora descriptiva espacial respecto a release, límites de IC95, OASIS (R² .5614 vs .499, RMSE 7.7237 vs 5.361, muestras y protocolos distintos), dominios y colas de M1, señal débil de M2 en demo, y decisión de no ampliar. No publicar datos brutos ni subir ZIP al repo.
5. Fase 6 se cierra con evidencias de operación y reconstrucción. Release es decisión posterior separada.

Para ejecutar la auditoría desde CMD sin usar una instalación anterior: `set "PYTHONPATH=%CD%\src"`, luego `python -m lithiumscope.experiments.data_lab.candidate_audit`. Ya fue ejecutada; no repetir sin motivo.
