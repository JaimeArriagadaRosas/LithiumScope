# Modelo 2 — Diagnóstico, experimentos controlados y criterios de liberación

**Estado:** investigación de desarrollo; no activa candidatos, no crea versiones de release y no certifica validación externa. Fecha: 12/11/2026.

## 1. Objetivo y restricciones

Determinar si el Random Forest experimental incurre en sobreajuste o subajuste; evaluar regularización apropiada para **126 variables derivadas de seis bandas Sentinel-2 y 783 muestras**, 767 elegibles según QA95; contrastar datos provisionales e inconsistentes sin borrar ni reescribir fuentes; revisar calibración, umbrales y selección; exigir evidencia independiente antes de autorizar un release. Mantener los 706 casos históricos y sus cinco folds espaciales congelados. El menú interactivo y el código del PDF no forman parte del alcance y deben conservarse. Trabajar en `main` con commits locales y usar disco D para temporales.

## 2. Diagnóstico reproducible

**Artefacto en evaluación:** `models/model_2/trained/training_20261112_123451_m0300/model.joblib`, Random Forest con 354 árboles, profundidad máxima 16, mínimo tres muestras por hoja y class_weight balanced. Este modelo **continúa pendiente**, no activo.

Con `evaluation/generalization.py` se comparó el ajuste aparente final contra predicciones OOF espaciales de las mismas 767 filas elegibles:

| Métrica | Entrenamiento aparente | OOF espacial |
|---|---:|---:|
| ROC-AUC | 0,990865 | 0,644973 |
| AP | 0,970613 | 0,384500 |
| Brier | 0,070213 | 0,211121 |
| Balanced Accuracy @ 0,5 | 0,957445 | 0,595001 |

**Diagnóstico:** señal fuerte de sobreajuste / generalización limitada, no de subajuste. **Limitación:** las predicciones in-sample provienen del estimador final entrenado con todos los ejemplos QA95, mientras las OOF proceden de entrenamientos por folds, y los cortes de litio OOF originales dependen del fold. Esta brecha inicial NO es una comparación pareada con el mismo estimador ni una prueba de significancia. La verificación más rigurosa es el estudio de cinco folds que puntúa entrenamiento y prueba en el mismo ciclo.

## 3. Independencia de la demostración

La antigua auditoría de `prediction/demo.py` solo consultaba la fuente histórica Mamani09 del Modelo 1. **Ahora también contrasta todas las 783 filas del manifiesto expandido M2**, utilizando distancia geodésica y valores de litio. Se registraron como **posibles duplicados, pendientes de investigación documental**:

- `A1015-22`: aproximadamente 1,91 m de una muestra GEOROC con Li 28,191 ppm coincidente.
- `A1015-25`: aproximadamente 3,79 m de una muestra GEOROC con Li 14,079 ppm coincidente.
- `A1015-21`: 175 m de una muestra GEOROC con Li diferente; cercanía espacial, **no duplicado confirmado**.

Los diez casos de demostración están a latitudes 43°–44°S; 637 de 783 muestras se concentran entre 10°–20°S. La nueva auditoría no declara independiente un conjunto por el simple hecho de no compartir identificadores ni coordenadas redondeadas. **El informe de diez casos no sirve como holdout independiente para certificar el release**. Las métricas deben presentarse como exploratorias. El chequeo adicional cambia únicamente la exactitud de la advertencia del reporte, no su diseño ni su renderizador.

## 4. Técnica elegida y experimento controlado

**Regularización de árboles Random Forest**: profundidad máxima 8 y mínimo 10 observaciones por hoja. Se prefiere frente a crear espectros artificiales o añadir atributos en un conjunto pequeño de muestras, con 126 características espectrales potencialmente correlacionadas y múltiples etiquetas del mismo parche.

El experimento `model_2/methodology/regularization_study.py` mantiene los mismos 783 test OOF, las mismas cinco particiones espaciales y el filtro QA95 únicamente en entrenamiento. Para ambas configuraciones, cada fold externo entrena modelos de sus tres subfolds internos para obtener predicciones OOF internas, calibrar con una sigmoide tipo Platt y seleccionar el umbral Balanced Accuracy. El fold externo **no participa** en elegir calibrador ni umbral. No se utilizan los diez casos demostrativos.

| Versión | ROC-AUC OOF | AP OOF | Brier OOF | Brecha aparente media AUC por fold |
|---|---:|---:|---:|---:|
| RF sin cambio, sin calibrar | 0,649555 | 0,381100 | 0,203806 | 0,366406 |
| RF regularizado, sin calibrar | **0,663126** | **0,396195** | 0,216693 | **0,307031** |
| RF sin cambio, calibración sigmoidal interna | 0,643546 | 0,362461 | 0,179843 | 0,366406 |
| RF regularizado, calibración sigmoidal interna | 0,653855 | 0,374560 | **0,178853** | **0,307031** |

Estas son estimaciones **exploratorias**, sujetas a intervalos por grupos geográficos y selección posterior. El mejor ranking sin calibración no implica mejor Brier. La calibración puede mejorar Brier a costa del ranking. **No desplegar un calibrador o umbral sin validación externa**.

## 5. Experimento de calidad de datos

`model_2/methodology/data_ablation_study.py` compara, con los mismos cinco test espaciales intactos y sin ajustar parámetros al conjunto demostrativo:

| Política de entrenamiento (solo TRAIN) | Ejemplos elegibles | ROC-AUC OOF agrupado | AP OOF |
|---|---:|---:|---:|
| QA95 completo | 767 | 0,644785 | 0,378575 |
| Excluir 77 GEOROC provisionales | 690 | 0,639168 | 0,377020 |
| Excluir sitios con conflicto de etiquetas | 693 | 0,642543 | 0,375070 |

No está justificado borrar GEOROC ni todos los sitios ambiguos indiscriminadamente: ninguno de los dos recortes mejoró la discriminación global en este experimento. Sigue siendo necesario investigar calidad por muestra, publicación, coordenadas y representatividad geográfica. **Eliminar una muestra de entrenamiento no equivale a eliminarla del conjunto de test; los 783 casos siguen evaluándose.**

## 6. Criterios de liberación científica

`evaluation/release_gate.py` define una decisión **sin promover ni instalar modelos**. Condiciones preliminares para un futuro conjunto externo previamente congelado, no consultado durante el desarrollo:

- Identidades, procedencia y etiquetas geoquímicas revisadas; sin posibles duplicados del entrenamiento.
- Al menos 100 muestras, 20 positivas, 20 negativas y tres regiones geográficas independientes. Son requisitos mínimos operativos preliminares, **no garantía estadística universal**; una validación de potencia puede exigir más.
- Comparación pareada con el CatBoost anterior sobre las mismas muestras, con intervalos por grupos espaciales; métricas de ROC-AUC/AP, Brier y clasificación con umbral elegido exclusivamente en entrenamiento.
- Umbral de revisión: ROC-AUC absoluta al menos 0,60, ninguna regresión AUC frente al anterior y deterioros de AP/Brier dentro de tolerancias predeclaradas; incertidumbre pareada también informada.
- Si no se cumplen: mantener candidato **pending**, conservar release previo. El cumplimiento numérico tampoco activa un release sin aprobación explícita.

## 7. Arquitectura y pruebas

Se auditaron 383 archivos de código versionados, con 23 superiores a 300 líneas al inicio. La arquitectura requiere separar responsabilidades, preservar interfaces y agregar controles automáticos de tamaño, sin modificar la semántica de CV, reportes o menús.

Se empieza por componentes de menor riesgo con interfaces ya cubiertas por pruebas: interpretación por caso y resumen; posprocesamiento histórico y diagnóstico científico. Las demás divisiones deben validarse antes de su commit. El cambio de arquitectura no debe mezclarse con ninguna conclusión de mejora científica.

## 8. Procedimiento de reproducción (CMD)

Desde la raíz del repositorio:

```cmd
set "PYTHONPATH=%CD%\src"
"D:\Librerias\Python312\python.exe" -m lithiumscope.model_2.methodology.regularization_study --output "results\scientific_simulations\m2_generalization_20261112\repeat_regularization"
"D:\Librerias\Python312\python.exe" -m lithiumscope.model_2.methodology.data_ablation_study --output "results\scientific_simulations\m2_generalization_20261112\repeat_ablation"
"D:\Librerias\Python312\python.exe" -m pytest tests -q
"D:\Librerias\Python312\python.exe" -m ruff check src tests main.py
```

No crear una nueva versión del modelo sin evaluar resultados, reproducibilidad, incertidumbre, criterios de independencia y aprobación humana. Las salidas de los experimentos se guardan en `results/scientific_simulations/m2_generalization_20261112` del disco D.



## 9. Estado de modularizaci?n y deuda pendiente

Tras conservar las interfaces p?blicas, se separaron las responsabilidades
de interpretaci?n, posprocesamiento cient?fico, experimentos, provisi?n,
preboot, distribuci?n y predicci?n integrada en m?dulos especializados.
Los archivos originales refactorizados quedaron por debajo de 300 l?neas
y las nuevas unidades tambi?n. Las pruebas de regresi?n garantizan
el funcionamiento observable, pero no sustituyen una evaluaci?n humana de
cohesi?n y estructura.

**Restan ocho archivos extensos: cuatro funciones de orquestaci?n/training y sus m?dulos de gran
tama?o:** Modelo 2 competition.py (760), Modelo 2 cv_runner.py (541),
Modelo 1 competition.py (513) y Modelo 1 cv_runner.py (433). Son las
funciones cient?ficas m?s cr?ticas; refactorizarlas por fases con
interfaces y snapshots de salida es preferible a partirlas mec?nicamente.
Tambi?n permanecen extensos model_2/data/expansion_vnext.py (536), model_2/data/builder.py (464), model_2/data/sentinel2.py (368) y results/release.py (305). Las dos excepciones de tests corresponden a la versi?n remota previa a
su divisi?n en archivos por escenarios. Ninguna excepci?n puede crecer;
su lista se registra en config/code_size_legacy_exceptions.json.

El control python scripts/check_code_size.py --check queda incorporado
al workflow existente de GitHub para impedir archivos nuevos mayores de
300 l?neas y aumentos en los m?dulos hist?ricos pendientes.

**Aprobaci?n arquitect?nica final pendiente:** reducir los ocho archivos restantes, incluidos los cuatro grandes
orquestadores y sus funciones a responsabilidades cohesionadas sin
alterar resultados, checkpoints ni interfaz interactiva.
