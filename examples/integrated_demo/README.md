# Demostración integrada de LithiumScope

Este directorio contiene un **snapshot curado y versionado** para la opción
`4. Demostración integrada automática`.

## Procedencia

Los registros provienen de:

- repositorio: `siwill22/andes_paleoelevation`;
- archivo: `datafiles/geoch_Castro++2021.csv`;
- commit fijado: `cc00b161ace1be8bd9ed46d306a8ac3b2dc39ecc`;
- licencia del repositorio de origen: MIT;
- contexto: geoquímica ígnea de los Andes del sur de Chile.

Se conservaron diez casos con concentración de Li, coordenadas y un subconjunto
amplio de variables compatibles con el esquema de LithiumScope. Los nombres se
normalizaron al contrato del Modelo 1. La fuente informa hierro total como
`feo_tot`; LithiumScope **no lo convierte silenciosamente a Fe2O3**, por lo que
`Fe2O3` se deja vacío y esa ausencia queda reflejada en los diagnósticos de
aplicabilidad.

Al preparar esta demostración, los identificadores seleccionados se contrastaron
con el mirror bootstrap público actualmente utilizado por LithiumScope y no se
encontraron coincidencias exactas de `Sample`. Esto no demuestra independencia
respecto de toda fuente futura u oficial. La ejecución vuelve a auditar
solapamientos cuando el dataset local de entrenamiento está disponible.

## Qué demuestra

La demostración ejecuta los dos modelos sobre los mismos casos:

1. el Modelo 1 estima `Li_icpms`;
2. LithiumScope obtiene automáticamente parches Sentinel-2 para las coordenadas;
3. el Modelo 2 calcula prioridad exploratoria;
4. se calculan métricas externas, concordancias y correlaciones matemáticamente
   válidas;
5. se genera una interpretación científica determinista y reproducible.

El score del Modelo 2 sigue siendo una **prioridad relativa**, no una
concentración de litio ni una probabilidad de yacimiento.
