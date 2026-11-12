# LithiumScope — Fase D: revisión física de etiquetas y parches

**Fecha:** 12 de noviembre de 2026. **Rama:** feat/scientific-simulation-lab. **Alcance:** revisión de integridad y procedencia, sin ajuste, entrenamiento o activación de modelos.

## Dictamen

**La unidad de supervisión muestra–parche requiere corrección antes de un nuevo entrenamiento científico definitivo.** El principal hallazgo es que 25 ubicaciones históricas contienen 77 muestras con etiquetas alto/bajo de Li discordantes, **pero todos los registros de cada ubicación comparten exactamente el mismo contenido binario de imagen satelital**. Un clasificador determinista no puede asignar dos clases diferentes a la misma entrada sin una política explícita por sitio o por litología.

Esto NO prueba que el análisis de laboratorio sea erróneo: varias muestras podrían tener composiciones distintas dentro del mismo parche, o el punto original podría ser aproximado. La solución no es borrar las observaciones, sino revisar la representatividad geológica y definir la unidad correcta de supervisión.

## Trazabilidad restituida

| Evidencia | Resultado |
|---|---:|
| Muestras históricas enlazadas a filas originales Mamani09 por ID normalizado + Li + coordenadas | 706 / 706 |
| Enlaces inequívocos a una única fila original | 705 / 706 |
| Identidades con más de una fila coincidente | 1 |
| Referencias geoquímicas recuperadas de Mamani09 | 706 / 706 |
| Descripciones litológicas históricas disponibles | 704 / 706 |
| GEOROC vinculado al cribado y bibliografía original | 77 / 77 |
| GEOROC sin descripción de tipo de roca | 12 |
| GEOROC sin comentario de muestra | 26 |
| GEOROC con 3 o menos decimales nominales en al menos una coordenada | 25 |
| Sitios de Li discordante | 25 sitios / 77 muestras |
| Sitios discordantes que comparten idéntica imagen (SHA-256) | 25 / 25 |
| Sitios discordantes que incluyen varios tipos de roca | 13 / 25 |
| Parches con metadatos y cobertura de píxeles satisfactoria | 767 / 783 |
| Parches con menos de 95 % de píxeles finitos | 16 históricos / 0 GEOROC |
| Parches espectrales distintos, según hash del archivo | 593 |
| Emparejamientos muestra–afloramiento–parche validados con documentación de campo | **0** |

### Corrección de una interpretación anterior

Los 706 registros históricos no guardan la cita bibliográfica en el manifiesto de entrenamiento, **pero sí la conservan en el fichero fuente Mamani09_Table_DR2.csv**, junto con tipo de muestra, litología y ubicación. Esta auditoría recuperó la relación entre la fuente y el entrenamiento mediante un cruce de identificador normalizado, concentración de Li y ambas coordenadas. Una muestra tiene dos filas originales coincidentes y no se certificó automáticamente la unicidad.

La recuperación bibliográfica **no equivale** a una demostración de precisión GPS, exposición superficial o correspondencia con una celda Sentinel-2.

### Las 16 incidencias de imagen no son un problema de metadatos

Para estas 16 imágenes históricas los identificadores, escenas, coordenadas, orden de seis bandas y dimensiones se corresponden con los archivos auxiliares. El problema comprobado es la **cobertura insuficiente de valores finitos**, entre 50 % y aproximadamente 92,1 %. Tres de esos registros también pertenecen a sitios con etiquetas discordantes. El umbral de 95 % reproduce el criterio de integridad técnica utilizado por la adquisición GEOROC.

La futura decisión de readquirir o aislar esas ventanas necesita una nueva evaluación controlada. **No se sobrescribieron las imágenes originales ni las 126 características ya utilizadas.**

## Veinticinco sitios: evidencia que debe revisar un especialista

El CSV conflicting_25_sites_dossier.csv detalla los 25 sitios con nombres originales de muestra, Li mínimo y máximo, referencias geoquímicas, tipologías, código geográfico y huella de la imagen.

Algunos casos prioritarios:

| Coordenadas aproximadas | Muestras | Li mínimo–máximo (ppm) | Litologías mencionadas |
|---|---:|---:|---|
| −18,483 / −70,067 | 4 | 7,8–36,3 | Rhyolite |
| −15,807 / −72,982 | 3 | 6,8–32,1 | Andesite, rhyolite |
| −16,248 / −71,348 | 3 | 6,7–30,4 | Rhyolite |
| −25,329 / −67,857 | 2 | 13,0–36,0 | Rhyolite |
| −18,198 / −69,139 | 2 | 11,4–33,3 | Andesite, rhyolite |
| −18,197 / −69,272 | 4 | 9,2–30,6 | Andesite, rhyolite |

Una imagen idéntica no contiene información sobre la composición química diferenciada de cada espécimen que comparte sus coordenadas. **Al utilizar el corte fijo descriptivo de 21,6 ppm, al menos 28 etiquetas de las 77 muestras discordantes serían imposibles de acertar simultáneamente para cualquier clasificador determinista que solo reciba esa misma imagen** (suma de las clases minoritarias de cada sitio). Es una cota descriptiva sobre el conjunto de entrenamiento, no una tasa de error OOF ni una estimación de generalización; los umbrales propios de cada fold difieren. Podrían resultar necesarias etiquetas agregadas por sitio, una definición más precisa de afloramiento o una función objetivo que admita heterogeneidad; **ninguna opción queda aprobada antes de revisar las fuentes y ejecutar comparaciones controladas**.

## Las 77 nuevas etiquetas GEOROC

Las 77 coincidieron con su muestra original, concentración de Li, latitud, longitud, referencia, auditoría bibliográfica y cola de selección. Los 77 parches aprobaron integridad técnica, pero todos mantienen categoría **provisional sin validación de campo**.

- **36** requieren prioridad P1 por al menos un indicador nominal de geolocalización imprecisa o falta de tipo de roca.
- Las otras **41** también requieren revisión de campo/superficie y correspondencia con parche (P2).
- **Ninguna** puede llamarse certificada por tener un DOI válido, precisión decimal aparente, una escena satelital o baja nubosidad.

La fracción SCL de nube y nieve que conserva el manifiesto no demuestra exposición de roca en el punto muestreado. Las fracciones NDVI del informe son **proxies de vegetación**, no certificaciones de afloramiento. No se recuperó una máscara SCL histórica completa por píxel desde estos artefactos.

## Contrato de revisión antes de entrenar

El archivo manual_review_evidence_template.csv deja sin rellenar, para cada caso prioritario, las columnas del revisor, documento fuente y mapa de campo, incertidumbre GPS, evidencia de roca superficial, correspondencia con parche, decisión y observaciones. Este formulario es un expediente de revisión científica, no un mecanismo para inventar etiquetas verificadas.

**Secuencia propuesta:**

1. **P0:** revisar los 25 sitios con etiquetas discordantes e imagen idéntica. Recuperar la posición real y las descripciones de roca en la publicación; no resolver por votación automática.
2. **P0/P1:** investigar las 16 imágenes incompletas, preservando el archivo original y registrando, antes de cualquier readquisición, qué píxeles y bandas faltan.
3. **P1:** resolver la identidad histórica con dos filas originales y los 36 GEOROC con datos de localización o litología insuficientes.
4. **P2:** obtener evidencia de campo o mapas de muestra y afloramiento para los otros 41 GEOROC.
5. **Fase siguiente:** preparar una matriz experimental previa que compare variantes de etiqueta por sitio/parche y QA espectral, con los mismos folds históricos y un test verdaderamente independiente; no usar la demostración conocida de diez casos como criterio de selección.

## Gate científico

**NO APROBADO.** El gate estricto permanece cerrado mientras existan etiquetas discordantes que comparten imagen, parches incompletos, identidad bibliográfica ambigua, sitios GEOROC de precisión geográfica cuestionable o ausencia de verificación física de la correspondencia muestra–parche.

No se declara inválido el conjunto completo; se prohíbe la certificación automática y se separan los niveles de evidencia.

## Artefactos y reproducción

El módulo de auditoría está en src/lithiumscope/model_2/methodology/physical_label_audit.py. Se ejecuta con:

    python -m lithiumscope.model_2.methodology.physical_label_audit --output results/scientific_simulations/m2_phase_d_20261112/physical_label_audit --strict

El modo estricto produce evidencia completa y termina con código 2 por motivos científicos pendientes, sin editar los datos, modelos ni menú. Sus archivos incluyen:

- scientific_quality_summary.json y scientific_readiness_gate.json
- historical_706_source_reconciliation.csv
- georoc_77_scientific_review.csv
- conflicting_25_sites_dossier.csv
- patch_image_integrity_783.csv
- full_sample_review_queue.csv y manual_review_evidence_template.csv

Se guardan las huellas SHA-256 de cinco archivos originales. El experimento de Fase C y el Modelo 2 activo permanecen intactos. No se hizo entrenamiento, selección de parámetros o calibración adicional, ni se ejecutó push.
