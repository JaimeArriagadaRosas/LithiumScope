# Procedimiento de laboratorio: ampliación GEOROC y diagnóstico

## Objetivo

Este procedimiento existe para responder, antes de entrenar, cuántas muestras reales quedan disponibles para el Modelo 1 y cuántos candidatos georreferenciados quedan disponibles para ampliar el Modelo 2.

El laboratorio es independiente del flujo principal de entrenamiento. No modifica `main.py` ni el preboot normal de la aplicación.

## Regla de preboot del laboratorio

Todo ingreso a `lithiumscope-inspect` pasa por un preboot de laboratorio.

El preboot de laboratorio comprueba:

1. Python, configuración, permisos y dependencias mediante el preboot compartido.
2. Disponibilidad del dataset base Mamani09.
3. Disponibilidad de una extracción GEOROC apta para LithiumScope.

El preboot de laboratorio solo garantiza los datos brutos necesarios. No considera como requisito que ya existan los archivos concatenados, deduplicados o limpiados; esos artefactos son precisamente el resultado que el laboratorio debe reconstruir y auditar.

## GEOROC: qué se necesita de los precompilados Andean Arc

Los tres precompilados Andean Arc oficiales suman aproximadamente 22.3 GiB. LithiumScope no necesita descargar ese conjunto completo.

La extracción buscada se restringe a:

- fuente: GEOROC;
- ámbito geológico: Andean Arc / margen convergente andino;
- material: whole rock (la interfaz histórica de GEOROC puede agrupar whole rock + volcanic glass; el adaptador local conserva solo whole rock);
- requisito químico principal: Li medido;
- coordenadas válidas;
- metadatos de muestra y procedencia;
- predictores de Modelo 1:
  - SiO2, TiO2, Al2O3, Fe2O3/Fe total compatible, MnO, MgO, CaO, Na2O, K2O, P2O5;
  - Th, U, Rb, Cs, Nb, Ta, Pb, Ba, Sr, Zr, V, Hf.

No se solicitan de forma intencional isótopos, gases nobles, tierras raras no usadas por M1 ni otras familias analíticas ajenas al contrato actual.

## Adquisición

La adquisición del laboratorio usa la interfaz pública de consulta de GEOROC/EarthChem y solicita una exportación filtrada. Nunca debe caer automáticamente al plan de descargar los tres precompilados completos.

Si la interfaz remota cambia o no puede producir una exportación filtrada, el laboratorio debe fallar con un mensaje explícito y conservar evidencia diagnóstica. En ese caso no se inicia una descarga masiva como fallback.

Fuentes técnicas de referencia:

- GEOROC Query by Chemistry: https://georoc.eu/georoc/Chemistry.asp
- GEOROC precompiled Andean Arc DOI: https://doi.org/10.25625/PVFZCE
- GEOROC 2.0 API: https://github.com/digis-georoc/database-api
- EarthChem Portal: https://portal.earthchem.org/
- EarthChem tutorial: https://earthchem.org/resources/support

La API GEOROC 2.0 publica rutas de consulta, pero actualmente exige un access key para las rutas `/api/v1/queries/`. Por ello el laboratorio no depende de una credencial privada y usa la interfaz pública de consulta/exportación.

## Flujo de la opción 7

```text
preboot laboratorio
  -> entorno
  -> Mamani09 bruto
  -> GEOROC filtrado bruto
  -> step 1: inspección de fuentes
  -> step 2: armonización GEOROC
  -> step 3: concatenación Mamani09 + GEOROC
  -> step 4: deduplicación
  -> step 5: limpieza real de M1
  -> step 6: candidatos M2 (Li + coordenadas)
  -> resumen final sin entrenamiento
```

El resultado principal es el número final de muestras de M1 antes de entrenar.

## Ampliación de Modelo 2

Las nuevas muestras GEOROC con Li y coordenadas también amplían el conjunto candidato de M2: cada muestra elegible puede solicitar un parche Sentinel-2 y posteriormente producir features espectrales.

El laboratorio reporta el número de candidatos antes de Sentinel-2. Ese número no equivale al N final de M2 porque algunas muestras pueden no conseguir una escena válida.

Como trabajo posterior, se evaluarán fuentes adicionales no duplicadas con GEOROC (por ejemplo, otras fuentes federadas de EarthChem y datasets específicos de provincias de litio). Estas fuentes no se mezclan automáticamente hasta revisar compatibilidad geológica y duplicados.

## Diagnóstico GPU

El mismo ejecutable de laboratorio ofrece `--gpu`. No se crea otro comando.

La prueba informa:

- detección NVIDIA mediante `nvidia-smi`;
- nombre y VRAM reportada;
- disponibilidad CUDA en PyTorch;
- prueba mínima real de XGBoost sobre CUDA cuando está instalado;
- prueba mínima real de CatBoost en GPU cuando está instalado.

La prueba GPU no descarga datasets.
