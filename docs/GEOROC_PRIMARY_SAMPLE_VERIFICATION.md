# Verificación de publicaciones y muestras: resultados contrastados

## Fuente principal 20576: evidencia directamente recuperada

Se recuperó el suplemento oficial del artículo *Parental arc magma compositions dominantly controlled by mantle-wedge thermal structure* (DOI `10.1038/ngeo2788`), publicado en Nature Geoscience. Descargas verificadas:

- `publication_sources/20576_supplement.pdf`, 434.833 bytes, SHA-256 `3b302359ff1f468d1bad2251f19f79623c67150f5c96b9fa177a00bf5763a5c3`.
- `publication_sources/20576_supplement.xlsx`, 280.239 bytes, SHA-256 `547bb7c6a39bf8615c790b5cc43bfce09c4fe4da11c47274f0d779ca15c448a4`.

El Excel contiene las pestañas `Data for new samples` y `Samples with new ICPMS data`, con una columna explícita `Li` y `Sample Name`. La unión por nombre normalizado y comparación exacta de Li (tolerancia 0,02 ppm) permite rastrear **210 de los 219 registros candidatos GEOROC** de la referencia [20576] a un valor en el suplemento original, sin inventar correspondencia para los nueve restantes.

De esos **210**, las tablas originales marcan **24 con `Passes Filters? = 1`** y **186 con `Passes Filters? = 0`**. **87** muestran además coincidencia geográfica a tolerancia ±0,01 grados en el suplemento; el resto no puede marcarse como coincidencia de coordenadas si el campo no está presente. Los números y las pruebas están disponibles en `georoc_20576_primary_li_crosswalk.csv` y `.json`.

### Restricción decisiva

El indicador `Passes Filters?` corresponde a los filtros del estudio original, no a un control de elegibilidad para prospectividad satelital de LithiumScope. Confirmar el valor Li en la tabla original **no demuestra exposición del material en la superficie del píxel**, método de ensayo individual suficientemente especificado, independencia total de fuentes ni existencia de una etiqueta válida para M2. No se deben convertir los 210 ni los 24 en muestras externas aprobadas.

## Otras fuentes consultadas

Se descargaron dos artículos adicionales de GEOMAR de las referencias [17913] (63 páginas) y [18365] (73 páginas). Ambos documentan análisis de elementos traza con procedimientos de ICP-MS o referencias metodológicas. La mención de ICP-MS sin un cruce por ID de muestra y columna Li primaria **no autoriza aprobación**.

## Estado de cierre

- Descarga de GEOROC, filtrado, coincidencias con el corpus completo de Mamani09, inspección de referencias, DOI y recuperación de suplementos accesibles: **realizados**.
- Comprobación de 210 valores de Li en tabla primaria para la referencia 20576: **realizada**.
- Independencia bibliográfica y geológica, verificación de la exposición del afloramiento para cada imagen y confirmación del método de Li por espécimen: **no acreditadas**.
- Etiquetas externas aprobadas: **0**. Exactitud externa ROC-AUC/AP/BA: **no calculada**.

El trabajo restante requiere acceder a evidencia geológica/muestreo primario adicional, no ejecutar más modelos ni extrapolar datos de una tabla. El modelo original y `main` no fueron modificados.
