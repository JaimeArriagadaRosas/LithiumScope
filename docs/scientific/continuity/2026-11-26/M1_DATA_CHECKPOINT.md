# M1 — punto de continuidad del laboratorio de datos

Fecha: 26 de noviembre de 2026. Rama: `feat/m1-targeted-data-refinement`.
Alcance exclusivo M1; M2, modelos activos, dataset productivo y main congelados.
Estado vigente: fase 5 cerrada operativamente con cobertura de cola insuficiente; comparación final de fase 6 implementada, pendiente de ejecución. Ver M1_FUENTES_Y_CIERRE.md.

## Referencias congeladas y criterio

Cohorte espacial: 3420 filas, 244 grupos. Cohorte académica: 3263 filas,
135 superiores a 40 ppm; intervalo 4.2439–126.12 ppm. Cinco folds históricos
del piloto, aleatorios y con solapamiento espacial. Mejor HGB enriquecido:
RMSE medio 7.074911 ppm, R² medio .631231, MAE medio 3.892598 ppm.
OASIS RF: RMSE 5.361, R² .499, otra población y protocolo; objetivo descriptivo,
no superioridad científica demostrada. Reducción RMSE necesaria ~24.23%.
La evaluación académica recortada no sustituye la espacial de dominio completo.

## Fases

| Fase | Estado |
| --- | --- |
| 1. Auditoría de tratamientos | Completa |
| 2. Adaptación | Validada |
| 3. Aumento artificial | Cerrada sin mejora |
| 4. Investigación dirigida | Realizada inicialmente; Álvarez-Amado pendiente |
| 5. Adquisición, EDA y armonización | Preflight validado; adquisición dirigida pendiente |
| 6. Comparación real/sintética | Pendiente, condicionada a elegibilidad |
| 7. Entrenamiento definitivo | Pendiente y condicionado; sin autorización de release |

## Aumento artificial

Run 20261126T180155_005d9a59, 115 fits nuevos, 120 tratamientos,
10/10 hashes verificados, TEST exclusivamente real. No repetir por defecto.

| Política | RMSE medio | R² medio |
| --- | ---: | ---: |
| Control | 7.074911 | .631231 |
| SMOTER | 7.230849 | .615639 |
| Selección interna | 7.319484 | .605633 |
| Gaussiano | 7.348474 | .602499 |
| Copias raras | 7.392674 | .594996 |
| Copias por relevancia | 7.440143 | .591892 |
| KDE | 7.717502 | .562479 |

Linaje de padres comprobado, no se exportaron todos los valores sintéticos
para auditoría directa. Recuperación histórica de trazas retrospectiva,
asistida por objetivo: no equivale a identidad certificada para producción.

## Fuentes adquiridas

| Fuente | Registros Li | Decisión |
| --- | ---: | --- |
| Meixner 2020 | 80 | Química incompleta; 5 coincidencias por nombre, 0 sitio+SiO2. No expansión directa. |
| Fernández-Turiel 2020 | 6 | QC y rango académico; 0 duplicados detectados, 19.96–29.64 ppm, ninguna >40. Revisión. |
| Macusani 2025 | 22/29 | 310–4160 ppm; 8 QC, 20 posibles duplicados, 29 bloqueadas por publicación. Cuarentena. |
| Chen 2020 | 3680 | 3676 QC, 2303 candidatas preliminares, 45 cola. Preflight pendiente. |
| Álvarez-Amado 2022 | desconocido | Original no adquirido. |

Cero filas nuevas aprobadas. No sumar fuentes como muestras únicas.
Originales en data/raw/model_1/research/directed/; no subir datos de terceros
al repositorio público. Los hashes y rutas de ejecuciones están en
[M1_AUGMENTATION.md](M1_AUGMENTATION.md) y [state.json](state.json).

Chen: 50145 filas de hoja, 66 columnas, incluye bibliografía; 3461 en rango
académico y 95 en cola antes de deduplicar. FEOT convertido a Fe2O3 equivalente
(factor 1.111348), conservando base original. 826 posibles duplicados y 1122
solapamientos bibliográficos con expansión; filtros no disjuntos. 2303
candidatas, 330 grupos; 45 cola, solo 6 andinas (2 cola). Cita [17558]
concentra 32/45 cola y todo el lote andino candidato: cuarentena provisional.
115 límites invertidos, 252 amplitud >.1 grados; no intercambiar extremos
sin evidencia. 2044 pasan filtro de límites ordenados y amplitud <=.1.
No hay IDs de muestra ni metadatos analíticos certificados en Chen.
La ausencia de arista no demuestra ausencia de duplicación.

## Ejecución y fallo del preflight (histórico)

Commit ejecutado 588c313247979ce38e7ca19711d4b1a874fb53d2: 81/81 pruebas
focalizadas, 4.59 s. El preflight alcanzó el resumen posterior a los cinco
folds, por lo que los controles previos del código (incluidos hashes locales
del whole-rock y originales) no lanzaron excepción. No se emitió manifiesto
ni se conocen los recuentos finales; no certificarlos en chat.
Fallo: acceso `plans.tail` resolvía el método de pandas, no la columna.
Corrección 60f0ec9: `summarize_plans` con acceso explícito por clave.
Prueba de regresión c98ecec cubre dos folds, frontera 40 ppm y elegibilidad.
82/82 pruebas aprobadas en 4.58 s tras corrección. Ver actualización final.

```cmd
git pull --ff-only && set "PYTHONPATH=%CD%\src" && python -m pytest tests/scientific/m1_expansion -q && python -m lithiumscope.experiments.model_1.augmentation.transfer_preflight
```

Salida: results/generalization/model_1_directed_transfer_preflight/.
Adjuntar manifest.json, static_review.csv, fold_review.csv, summary.csv,
cross_source_edges.csv. No repetir descargas, auditorías anteriores ni fits.

## Decisiones pendientes del checkpoint anterior (histórico)

1. Validar los cuatro hashes y recuentos por fold; comprobar cuántos candidatos
   únicos y cuántos de cola sobreviven, por fuente/publicación y grupo.
2. Resolver método/material/CRS y procedencia bibliográfica. No convertir
   candidatos de revisión en datos aprobados automáticamente. Si no aportan
   cobertura útil, cerrar la expansión o buscar una fuente mejor.
3. Congelar una reserva realmente independiente antes de cualquier ajuste
   definitivo. Una reserva de Chen explorado no cumple esa condición.
4. Si hay datos aptos, predefinir comparación acotada A histórico, B histórico
   + sintético (resultado negativo archivado), C histórico + real, D ambos;
   mantener TEST histórico y filtros exclusivamente TRAIN en cada partición
   externa e interna, excluir grupos/publicaciones/identidades de evaluación.
   No reutilizar membresías de outer TRAIN dentro de inner TRAIN sin recalcular.
   Fijar presupuesto, selección y criterios antes de ejecutar; no lanzar
   campañas si faltan metadatos, soporte o volumen. Reportar RMSE, R², MAE,
   sesgo, error de cola y estabilidad, conservar resultados desfavorables.
5. Solo considerar fit definitivo con evidencia favorable y validación espacial
   e independiente. No modificar producción, guardar/promover modelos,
   fusionar main ni tocar M2 sin autorización correspondiente.

Última suite completa: 1002 aprobadas, 3 omitidas, cobertura 57.05% antes
de expansión. No confundir con las 81 focalizadas recientes.


## Actualización vigente: preflight completado

82/82 focalizadas en 4.58 s; 4/4 archivos verificados. Manifiesto
21b112c5faaee0c410a5045ab295b57a41c875eb6dda78f268c8b4c0cfb0d99b.
Run 20261126T211301_3a7cd8be, 3686 registros, 18430 membresías.
1974 Chen pasan gate estático, 1588 distintos en algún fold, 1469 en los
cinco; únicamente 4 distintos >40 ppm (41.4–46.19); ningún >60.
Puna 0/6 por falta de soporte multivariado en todos los folds.
La cola de candidatos representa 0.252% frente a 4.14% histórica.
Conservar control HGB 7.074911/.631231, no ajustar de nuevo sin evidencia útil.
Cribado cerrado para fuentes actuales, fase 5 abierta para fuente dirigida
original. Macdonald 1992, USGS PP1523 y Álvarez-Amado 2022: revisar tablas
primarias, no aprobar por resumen. No se adquirieron nuevas tablas en este
turno. Fase 6 no ejecutada, fase 7 bloqueada. Cero nuevas aprobadas.
No hay comando pendiente de preflight. M2, producción y main congelados.
Detalle y hashes en M1_AUGMENTATION.md y state.json.


## Decisión de cierre de adquisición y comparación acotada — 26/11
Revisión bibliográfica y catálogo completo de las fuentes registradas en [M1_FUENTES_Y_CIERRE.md](M1_FUENTES_Y_CIERRE.md). Se detiene la búsqueda externa en esta iteración, conservando candidatos y cuarentenas. Fase 5 cerrada operativamente con cobertura de cola insuficiente, no certificación de nuevas filas. Se levanta únicamente el bloqueo de comparación exploratoria de fase 6: sensibilidad explícita con metadatos no certificados, sin cambiar predictores, algoritmo, hiperparámetros, TEST ni producción. 15 fits HGB nuevos máximo, A y B archivados, C1 solo cola Chen, C2 transferencia Chen ponderada por banda, D C2 + SMOTER histórico TRAIN. Peso individual <=.10 hasta 40 ppm y <=.25 por encima, peso total candidato por banda <=10% del histórico TRAIN; cero si banda sin histórico. No se eligen políticas con TEST para inferencia definitiva; reporte descriptivo de umbral OASIS y guardas. Resultados sintéticos exportados además de linaje. 85 pruebas esperadas, ejecución pendiente. No se inicia otra búsqueda o fit por defecto si no alcanza objetivos. Fase 7 sigue bloqueada por evidencia favorable, validación espacial/independiente y autorización. M2 congelado.
