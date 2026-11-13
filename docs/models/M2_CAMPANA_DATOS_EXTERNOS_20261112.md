# Modelo 2 — Campaña de adquisición externa y control de sobreajuste

**Fecha:** 12-11-2026. **Estado:** datos descargados, auditoría inicial ejecutada; **ninguna etiqueta externa aprobada**. No se cambió ningún modelo, partición, escena, predicción, menú ni registro de release.

## 1. Propósito y línea de base

El ensayo canónico del 12-11-2026 comparó Random Forest clásico y regularizado en 783 registros y cinco folds espaciales. La regularización disminuyó la brecha aparente de entrenamiento a test, pero el intervalo bootstrap de su mejora ROC-AUC incluyó cero. No corresponde declarar resuelto el sobreajuste ni promocionar modelos. Ver `docs/models/M2_PROTOCOLO_CANONICO_20261112.md`.

La prioridad de esta campaña es encontrar **mediciones de litio verificables, sitios independientes y una mejor distribución geográfica**, no agregar indiscriminadamente bandas, imágenes repetidas o muestras de materiales incompatibles.

## 2. Material adquirido (exclusivamente disco D)

Raíz: `D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Datasets-Externos\2026-11-12`.

| Fuente y archivo | Casos | Material | Estado |
|---|---:|---|---|
| Figshare La Negra `samples_rock.csv` | 7 | Roca de afloramiento, 5 sitios | Revisión documental y compatibilidad de método pendientes |
| Figshare La Negra `samples_soil.csv` | 101 | Suelo superficial | No mezclar con roca M2 |
| Figshare La Negra `samples_soil_duplicates.csv` | 15 | Duplicados de QC de suelo | Nunca contarlos como sitios nuevos |
| SERNAGEOMIN `sernageomin_salmueras.json` | 243 | Agua/salmuera | No mezclar con roca M2; unidad/método por verificar |
| SERNAGEOMIN `sernageomin_costras.json` | 369 | Costra salina | No mezclar con roca M2; unidad/método por verificar |
| SERNAGEOMIN `sernageomin_limites.json` | 19 polígonos | Geometría de salares | Contexto cartográfico, no etiquetas |

**Total:** 735 registros geoquímicos, todos con litio numérico y coordenadas válidas; **0 nuevas etiquetas admitidas al entrenamiento**. Los 19 polígonos no se incluyen en los 735. Los JSON descargados se verificaron contra el número de registros anunciado por el servicio al momento de la adquisición.

La fuente Figshare corresponde a Lam-Esquenazi y colaboradores, 2026, DOI `10.6084/m9.figshare.32181333` (CC-BY-4.0), con 62 analitos por muestra y Li en ppm. El método es ICP-MS tras digestión **parcial** con agua regia; no es directamente equivalente a litio total de una roca. En su análisis de duplicados de suelo, Li presenta R² ≈ 0,78 (criterio «aceptable», no «alto»).

SERNAGEOMIN: `https://services.arcgis.com/r7t1P5pnkoOLRdhr/arcgis/rest/services/Salares_Norte_de_Chile/FeatureServer`. Las capas consultadas son 1 (agua), 3 (costras) y 14 (límites). No se descargó la capa 5 de mineralogía porque reutiliza la población de la capa 3, evitando multiplicar artificialmente los registros. Comprobar condiciones de reutilización antes de redistribuir los archivos.

### 2.1. Fuente adicional cribada y descartada para etiquetas

Se descargaron además los cuatro archivos Excel de Ronemus et al. (Zenodo 18423631; DOI 10.5281/zenodo.18423631; CC-BY-4.0), sobre geoquímica del arco andino cerca de 35°S. Se revisaron las columnas de elementos traza y roca total (S2, S3 y S4): **no contienen Li analítico**, por lo que no permiten etiquetar M2. S1 proporciona metadatos y geocronología, tampoco etiquetas de Li. Los archivos permanecen en la carpeta externa de fuentes descartadas para reutilizarse como contexto geológico si fuese pertinente. Ver `zenodo_18423631/cribado_fuente.json` para hashes y motivo de exclusión.

## 3. Verificación geográfica de roca de afloramiento

Se compararon las **cinco ubicaciones distintas** de roca Figshare contra las 783 filas del manifiesto actual (`vnext_expanded/training_manifest.csv`) mediante distancia Haversine al vecino de entrenamiento más cercano:

| Sitio | Réplicas | Li promedio, ppm (AR-ICP-MS) | Vecino M2 más cercano |
|---|---:|---:|---:|
| GT-TO-03-01 | 2 | 13,30 | 42,135 km |
| GT-TO-04-01 | 1 | 17,10 | 46,357 km |
| JL-TA-03-01 | 1 | 19,50 | 116,967 km |
| PD-AN-01-03 | 2 | 17,25 | 147,281 km |
| PI-AN-02-13 | 1 | 8,20 | 21,399 km |

Los cinco sitios superan una distancia de 5 km, pero esto **no prueba** que se hayan obtenido mediante una publicación o campaña independiente de todas las fuentes históricas, ni que las mediciones tengan el mismo significado químico. Las réplicas no constituyen muestras espacialmente independientes. Los cinco valores de Li se sitúan por debajo de los umbrales q75 históricos (~21–22 ppm), de modo que **no aportan positivos** bajo esos umbrales; tampoco sirven por sí solos como test AUC (se necesitan ambas clases).

## 4. Archivos de la auditoría reproducible

Script en el repositorio: `scripts/external_lithium_evidence_audit.py`.

Salidas externas:
- `audit/external_candidates_separated.csv`: casos, coordenadas, material, Li bruto, unidad, procedencia y bloqueo explícito de admisión.
- `audit/rock_site_review.csv`: agregación por sitio y distancia al entrenamiento.
- `audit/source_integrity.json`: hashes SHA-256 del material y manifiesto de referencia, cifras, licencia, límites y procedencia.

El script falla si hay fuentes vacías, faltantes, reportes de ArcGIS truncados, errores API o cifras de población distintas de las verificadas; no modifica fuentes ni artefactos de entrenamiento. Ejecutar desde la raíz del clon:

    & 'D:\Librerias\Python312\python.exe' scripts\external_lithium_evidence_audit.py --dataset-root 'D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Datasets-Externos\2026-11-12' --training-manifest 'data\processed\model_2\vnext_expanded\training_manifest.csv'

## 5. Decisión y acciones siguientes

**No iniciar entrenamiento sobre estas 735 observaciones.** Sería engañoso sumar 243 salmueras o 369 costras a 783 parches de roca y declarar que el sobreajuste bajó: las distribuciones químicas, las unidades y la relación con la reflectancia satelital son distintas.

Para lograr un nuevo test externo defendible: obtener suficientes **sitios de roca** independientes con Li alto y bajo y analítica homogénea; comprobar publicación, ID de muestra, superficie expuesta, método, unidad y coordenada original; revisar duplicados bibliográficos respecto de Mamani09/GEOROC; completar parches Sentinel-2 con QA de nubes; y entonces ejecutar una evaluación espacial con embargo real de 5 km, definiendo reglas **antes** de abrir el test. Conservar intacto el modelo y las métricas históricas para comparación pareada. El resultado final puede mostrar límites reales de predictibilidad y no se debe forzar una mejora.

La fuente de uso más inmediato es el nuevo conjunto Figshare para **piloto de independencia y procedimiento**, no para demostrar mejora de AUC. Datos disponibles no equivalen automáticamente a nuevas etiquetas admisibles.
