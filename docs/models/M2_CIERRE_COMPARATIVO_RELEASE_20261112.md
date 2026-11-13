# LithiumScope Modelo 2 — cierre comparativo frente al release publicado

**Fecha:** 12 de noviembre de 2026. **Decisión:** cierre de la investigación interna sobre este protocolo; **NO sustituir el release activo**, **NO continuar ciclos de hiperparámetros con estos folds**. Se conserva íntegro el Modelo 1.

## 1. Cuál es el release anterior

El registro local `models/active_models.json` identifica como **Modelo 2 activo publicado** el artefacto `training_20260920_195933_m0300`, dentro del release de GitHub `lithiumscope-training_20260925_111245_m0300`. Esa es la referencia para la comparación, no CatBoost de una competencia posterior.

Se recuperaron sus predicciones out-of-fold históricas sobre 706 observaciones (`legacy_archived`). Se cruzaron por índice, muestra, fold, grupo geográfico y etiqueta con el candidato M2 exclusivamente espectral (RF fijo regularizado, 126 características) ya ejecutado, sobre **las mismas 706 observaciones**. En ambas series hay 177 positivos y 66 grupos. Todas las coincidencias se verificaron; no se usó el PDF ni los diez casos de demostración.

**Advertencia:** las particiones históricas ya han sido usadas en el desarrollo. La comparación es objetiva en cuanto a población y etiquetas, pero **NO es un nuevo test externo**, y los candidatos fueron desarrollados observando tendencias de estos folds.

## 2. Resultado numérico: mismo test de 706 casos

| Métrica | Release publicado (OOF archivado) | M2 corregido sin calibración | M2 corregido con sigmoide entrenada solo en train |
|---|---:|---:|---:|
| ROC-AUC (↑) | 0,620711 | **0,658721** | 0,647261 |
| Average Precision (↑) | 0,382668 | **0,393443** | 0,366600 |
| Brier (↓) | 0,184115 | 0,219656 | **0,180070** |

La ganancia puntual del candidato sin calibración sobre el release anterior es **+0,038010 ROC-AUC** y **+0,010775 AP**, pero **+0,035541 Brier** (empeoramiento). La sigmoide entrenada con OOF de tres grupos internos dentro de cada fold mejora el Brier, pero disminuye AP por debajo del release. No existe una variante que supere las **tres métricas simultáneamente** en este cierre.

### Incertidumbre de la comparación pareada

Bootstrap geográfico pareado de 320 repeticiones, con 66 grupos; diferencias candidato menos release:

| Variante | Δ ROC-AUC, IC95 % | Δ AP, IC95 % | Δ Brier, IC95 % |
|---|---|---|---|
| M2 corregido original | +0,038010 [−0,015427; +0,103884] | +0,010775 [−0,025740; +0,086214] | +0,035541 [+0,006336; +0,054367] |
| M2 corregido calibrado | +0,026550 [−0,020561; +0,097879] | −0,016068 [−0,053301; +0,068043] | −0,004045 [−0,017073; +0,004008] |

Todos los intervalos de mejora para AUC/AP incluyen cero; por ello **la superioridad de discriminación no está demostrada**. En el brazo no calibrado el Brier empeora incluso dentro del intervalo reportado. La calibración no autoriza declarar ganador: el AP empeora puntualmente y su intervalo incluye cero. Estos intervalos no representan confirmación independiente, ni corrección por las múltiples variantes probadas durante el proyecto.

## 3. Decisión sobre reflectancia Sentinel-2

La corrección radiométrica **ya se había ejecutado**, por lo que no corresponde abrir otra campaña idéntica. Los resultados históricos comparables (706 observaciones, 66 grupos) en `results/scientific_simulations/sentinel_methodology_20261112/experiments/` fueron:

| Extracción | ROC-AUC | AP | Brier |
|---|---:|---:|---:|
| Release archivado | 0,620711 | **0,382668** | **0,184115** |
| DN válidos sin relleno | 0,632138 | 0,368460 | 0,185724 |
| Reflectancia corregida, solo píxeles válidos | 0,602234 | 0,327413 | 0,196837 |
| Reflectancia corregida, relleno histórico | **0,640618** | 0,367808 | 0,185595 |

Ninguna de esas tres extracciones mejoró simultáneamente AUC, AP y Brier frente al release. Por tanto, **no se reentrena ni promueve** una variante radiométrica de forma automática.

## 4. Alcance real de la mejora

Las correcciones metodológicas —umbrales de etiqueta definidos sobre entrenamiento, folds espaciales sin superposición de grupos, QA aplicado solo a entrenamiento, regularización de RF, selección y calibración internas— son necesarias y más confiables que la metodología anterior. **Sin embargo, corregir el método no garantiza aumentar toda métrica predictiva**. El RF corregido ofrece mejor AUC/AP puntuales pero peor calibración. No sería honesto venderlo como superioridad científica probada.

La conclusión defensible para la memoria es: *«La metodología corregida reduce riesgos de fuga y permite una evaluación espacial reproducible. El candidato regularizado presenta una mejora puntual de ROC-AUC y AP frente al release anterior sobre los mismos 706 casos, aunque la incertidumbre espacial no confirma superioridad y su calibración requiere cautela.»*

## 5. Cierre técnico

Se implementó `src/lithiumscope/model_2/methodology/release_closure_comparison.py` y la prueba `tests/scientific/test_m2_release_closure_comparison.py`. Resultados producidos en `results/scientific_simulations/m2_release_closeout_20261112/`: `paired_release_706.csv` y `closure_decision.json`. Ningún modelo, release activo, menú, imagen ni dato del Modelo 1 fue modificado. Los experimentos anteriores de radiometría y sobreajuste permanecen disponibles para auditoría, sin repetir entrenamientos costosos.

**Decisión final:** mantener el release anterior activo hasta contar con un conjunto externo realmente independiente. **La fase de corrección y comparación queda cerrada; no seguir ajustando estos cinco folds en busca de un resultado forzado.** Si se exige un nuevo release científicamente superior en las tres métricas, su aprobación continúa bloqueada por la evidencia, no por falta de refactorizaciones o datasets indiscriminados.
