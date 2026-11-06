# LithiumScope ? precisiones acordadas y ejecuci?n controlada de M1/M2

Fecha: 2026-11-06. Rama: feat/scientific-simulation-lab.
Estado: trabajo experimental. Los modelos del 05/11 siguen siendo evidencia de desarrollo, no candidatos aprobados para release.

## Respuestas a las consultas del usuario

### ?Por qu? GEOROC podr?a ayudar al Modelo 2 si cumple una funci?n diferente?

El Modelo 1 usa directamente elementos y variables geoqu?micas para predecir concentraci?n de litio. GEOROC mejora potencialmente tanto la entrada geoqu?mica como la cobertura y las etiquetas de M1.

El Modelo 2 aprende relaciones entre **caracter?sticas espectrales de im?genes Sentinel-2** y una referencia de litio alto/no alto. En su entrenamiento existente, cada muestra de Li tiene coordenadas y se asocia a un **parche satelital**; el Li se utiliza **solo como etiqueta de supervisi?n** (q75 calculado con el entrenamiento de cada fold), no como caracter?stica predictora. GEOROC podr?a aportar nuevos pares (parche Sentinel-2, etiqueta de Li), no sustituir los espectros por columnas geoqu?micas ni crear un tercer modelo.

**Advertencia:** la etiqueta est? asociada a la muestra y su parche, no indica que todos los p?xeles contengan esa concentraci?n, ni representa una segmentaci?n pixel a pixel. La validez depende de que la ubicaci?n corresponda al parche, que la muestra represente el material superficial observable y que su litio sea comparable al de entrenamiento.

### ?Aumenta el peso o cambia el funcionamiento del Modelo 2?

Si se aceptan nuevos pares, crecer?n las descargas, los parches almacenados, el c?lculo de caracter?sticas y el tiempo de entrenamiento. El costo de preparaci?n puede crecer aproximadamente con el n?mero de im?genes, aunque no necesariamente de forma exactamente lineal por la disponibilidad de escenas/cach?. Entrenar con m?s filas puede aumentar tiempo y memoria. **El n?mero y significado de las variables de entrada se mantienen** (seis bandas Sentinel-2 procesadas al mismo extractor), por lo que el tama?o del modelo final no crece necesariamente en la misma proporci?n y el flujo de predicci?n sigue siendo: imagen -> caracter?sticas espectrales -> puntuaci?n de prospectividad. La mayor cobertura s? puede cambiar los par?metros aprendidos, la calibraci?n y el umbral de clasificaci?n.

Una nueva fuente de etiquetas no garantiza mayor precisi?n: coordenadas inexactas, litolog?as o anal?ticas no comparables y dominio espacial diferente pueden perjudicar el resultado. Una prueba sin litio independiente mide estabilidad o viabilidad t?cnica, **no exactitud geol?gica externa**.

## Evidencia previa que motiva esta fase

- M1: 1.120 registros A a 3.420 registros B luego de reconstruir Fe2O3 equivalente a partir de FeOT y auditar cierre 94?102; 2.744 muestras GEOROC B. Mejora exploratoria RMSE medio de 20,78 a 19,40 ppm (?6,65 %), pero bootstrap de mejora RMSE incluye cero. No promover B sin validaci?n de procedencia y test independiente.
- M2: 706 pares hist?ricos, 66 grupos (uno con 156 muestras). La curva de aprendizaje con distintas fracciones de grupos mantiene ROC-AUC aproximadamente 0,56?0,58, aunque Brier mejora. M?s ejemplos de las mismas regiones no garantizan mejor discriminaci?n.
- El lector hist?rico de muestras M2 privilegia Longitude (X), Latitude (Y) y Sample por columnas completas, descartando GEOROC cuyos valores residen en Longitude, Latitude y source_sample. El dataset B ofrece 178 grupos geogr?ficos potencialmente adicionales, sujetos a verificaci?n.
- 2.566 registros GEOROC B tienen l?mites de coordenadas con extensi?n nula, lo que NO demuestra precisi?n de campo suficiente para parches de 640 m. Requieren validaci?n por fuente/lote.
- El release publicado y los modelos experimentales locales deben permanecer sin cambios.

## Plan de ejecuci?n aprobado (fase actual)

1. **Documentar primero** este acuerdo, preservar cambios en commits locales y nunca hacer push sin autorizaci?n. Mantener inalterados men? principal, entrenamiento productivo y modelos/release activos.
2. **M1:** continuar verificaci?n de FeOT/Fe2O3 total, cierre de ?xidos, referencias bibliogr?ficas y duplicados, y documentar limitaciones de las 2.300 muestras recuperadas. Los resultados de desarrollo son hip?tesis, no evidencia externa definitiva. No imponer autom?ticamente B en el pipeline original.
3. **M2:** preparar un camino experimental *opt-in* para formar pares supervisados desde GEOROC B, recuperando coordenadas e identificadores por fila. Seleccionar un piloto peque?o de ubicaciones y grupos diferentes, documentar origen, Li y incertidumbre geogr?fica. La configuraci?n original y sus 706 muestras contin?an intactas.
4. **Piloto Sentinel-2:** reutilizar el adaptador de Earth Search/Sentinel-2, seis bandas y el extractor existente, guardando todo fuera de los datos can?nicos. Limitar expl?citamente n?mero de solicitudes, registrar por muestra escena/fecha/nubes/bandas/fallos y abstenerse ante im?genes inservibles. La obtenci?n de un parche NO certifica la ubicaci?n geoqu?mica ni su validez como etiqueta.
5. **Tests y evaluaci?n:** pruebas sint?ticas de selecci?n de columnas por fila, identidad/procedencia, aislamiento respecto del flujo original, l?mites de adquisici?n y firma de caracter?sticas; controles de fuga por grupos espaciales. Solo con suficientes muestras verificadas e im?genes aptas considerar luego un experimento de entrenamiento separado y comparable; no entrenar el modelo definitivo ni alterar los activos sin evidencia.
6. **Informe y commit:** dejar tabla de elegibilidad, resultados del piloto y decisi?n de suficiencia, incluyendo costo de almacenamiento/preparaci?n, pasos restantes e incertidumbres. Commits locales diferenciados, sin push.

## L?mites expl?citos

- No incorporar elementos qu?micos de GEOROC como nuevas variables predictoras de M2.
- No interpretar una muestra puntual como anotaci?n de todos los p?xeles.
- No usar una etiqueta calculada con datos del test para seleccionar algoritmos o calibraciones.
- No tocar CSV originales, parches hist?ricos, cach?s can?nicas, datos productivos, men? ni configuraciones de modelos existentes.
- No descargar miles de escenas por activar accidentalmente el lector del entrenamiento principal.
- No promover ni publicar los modelos del 05/11 ni los futuros del piloto sin un conjunto independiente com?n.
- No asumir que las 2.744 muestras son inmediatamente 2.744 pares v?lidos de entrenamiento.
- Datos externos nuevos EarthChem/PetDB o SERNAGEOMIN: fuera de esta fase salvo evaluaci?n documental; priorizar primero la fuente GEOROC existente.

## Criterios de aceptaci?n de un par M2

- Li num?rico positivo o f?sicamente admisible, procedencia/papel geol?gico documentado, material WHOLE ROCK de GEOROC B.
- Coordenadas dentro del ?rea objetivo y consistencia espacial verificable; si la precisi?n no est? documentada, marcar como **no certificado para entrenamiento final**.
- Identidad estable y libre de colisiones, que incorpore procedencia para nombres gen?ricos.
- Parche Sentinel-2 con seis bandas, CRS y fechas de adquisici?n registrados; imagen utilizable seg?n las reglas existentes.
- Grupo espacial conocido; las regiones de test deben quedar fuera del entrenamiento. Las etiquetas de litio se derivan de train solamente.
- Registro completo de p?rdidas, errores, y de la distinci?n entre **viabilidad t?cnica** y **validaci?n cient?fica**.

## Criterio de salida

Se entregar? evidencia reproducible y un piloto, y una recomendaci?n expl?cita de si vale la pena ampliar M2. La ?ltima decisi?n sobre aprobar nuevos datos y ejecutar el entrenamiento final debe basarse en resultados comparables, no en el aumento bruto de archivos.

## Documentaci?n relacionada

- docs/models/INTEGRAL_TRAINING_VNEXT_2026.md
- docs/models/DATA_ADEQUACY_VERIFICATION_2026.md
- results/scientific_simulations/vnext_20261106/stage4_evidence/
