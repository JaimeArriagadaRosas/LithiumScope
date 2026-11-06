# Campaña de revisión por publicaciones — GEOROC

## Alcance de esta fase
Se revisó el universo de **59 referencias bibliográficas** vinculadas con las **1.450 muestras** que habían pasado los controles iniciales de distancia, tipo de roca, duplicados, perforación y referencia. Se integró la auditoría adicional contra las **1.788 filas del corpus completo Mamani09**, no solamente las 706 muestras entrenadas en M2.

Se elaboró el catálogo `publication_first_review_register.csv` de las 59 referencias, priorizando las publicaciones con más candidatas alejadas del corpus completo y la cobertura de Chile. El archivo `publication_priority_top10.csv` identifica las diez primeras publicaciones para revisión documental: agrupan **904 candidatas**, todas separadas al menos 100 km del corpus Mamani09 en este filtro geográfico. No se ha demostrado su independencia bibliográfica ni representatividad superficial.

## Recuperación de publicaciones
Se consultó el catálogo OpenAlex por DOI para las diez referencias prioritarias: **5** figuran con alguna ruta de acceso abierto, aunque no todas cuentan con PDF descargable. Registro: `top10_open_access_locations.csv` y `top10_source_lookup_manifest.json`.

Se descargaron y verificaron PDF de las referencias **17913** (63 páginas) y **18365** (73 páginas) desde el repositorio académico GEOMAR. El artículo 17913 describe procedimientos de ICP-MS para elementos traza, con referencia al método de Garbe-Schönberg (1993). El artículo 18365 remite a Jacques et al. (2013) y Hoernle et al. para detalles de ICP-MS. Estas menciones **no equivalen a la acreditación de Li por muestra** y no autorizan métricas de evaluación externas.

Los PDF, enlaces, hashes y fragmentos que indican métodos están almacenados en `publication_sources/` y `paper_method_scan.json`. Las fuentes originales descargadas previamente permanecen en `primary_sources/`.

## Criterios definitivos de aceptación
Una muestra solo se incorporará como etiqueta externa de M2 cuando la fuente primaria permita comprobar: (1) valor y técnica de medición de Li, con unidades y posibles interferencias; (2) identificador y coordenadas con precisión suficiente; (3) muestra superficial/afloramiento efectivamente representado por el parche Sentinel-2; (4) ausencia de contaminación por muestras, publicaciones y compilaciones empleadas en Mamani09 y entrenamiento; (5) clasificación fija derivada exclusivamente del entrenamiento, sin optimizar umbrales en los datos externos.

**Resultado de esta fase:** catálogo y evidencias recuperadas; **0 etiquetas aprobadas**; métricas externas de ROC-AUC, AP o BA no calculables con rigor. La etapa de revisión documental automatizable queda organizada y repetible, pero la aceptación científica requiere evidencia primaria por muestra.

## Códigos y resultados
- `src/lithiumscope/experiments/georoc/publication_first_review.py`
- `src/lithiumscope/experiments/georoc/publication_source_lookup.py`
- `src/lithiumscope/experiments/georoc/publication_pdf_screen.py`
- `results/scientific_simulations/external_geochemistry_audit/publication_first_review_register.csv`
- `results/scientific_simulations/external_geochemistry_audit/publication_priority_top10.csv`
- `results/scientific_simulations/external_geochemistry_audit/top10_open_access_locations.csv`
- `results/scientific_simulations/external_geochemistry_audit/publication_sources/paper_method_scan.json`

No se modificó `main`, el artefacto entrenado de M2 ni los informes académicos.
