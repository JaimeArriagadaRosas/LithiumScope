# Modelos publicados

LithiumScope no utiliza tags de versión del producto como `v1`, `v2` o `v3`.

Los tags `lithiumscope-training_...` identifican entrenamientos concretos por fecha/hora y sirven como ancla de trazabilidad para los modelos publicados.

## GitHub Release

Un tag de Git siempre apunta a código fuente. Por eso GitHub muestra automáticamente:

- `Source code (zip)`;
- `Source code (tar.gz)`.

LithiumScope **no descarga esos archivos**.

Para distribuir un entrenamiento se crea un GitHub Release sobre su tag y se adjunta un asset cuyo nombre termina en:

`-artifacts.zip`

Ese ZIP contiene únicamente los artefactos necesarios de Modelo 1 y Modelo 2, sus metadata y el release manifest.

## Opción 4

`4. Cargar modelo versionado` consulta GitHub Releases y muestra solamente aquellos que tienen un asset compatible.

La instalación:

1. descarga el ZIP a una carpeta temporal;
2. verifica el SHA-256 publicado por GitHub cuando está disponible;
3. bloquea rutas ZIP inseguras y bundles anormalmente grandes;
4. valida el release manifest;
5. verifica el SHA-256 de cada `model.joblib` contra su `metadata.json`;
6. comprueba que ambos modelos pueden deserializarse con el entorno actual;
7. instala ambos artefactos;
8. actualiza `models/active_models.json` solo después de completar la instalación.

Las predicciones prefieren el modelo activo. Un entrenamiento local nuevo pasa a ser activo automáticamente para su familia.

## Clon nuevo

El preboot inicial ya no descarga datasets ni construye Sentinel-2 antes de mostrar el menú.

Los datasets se provisionan cuando el usuario elige entrenar. Por eso un usuario que solo quiere probar un modelo publicado puede:

```text
clonar repositorio
→ instalar dependencias
→ ejecutar LithiumScope
→ 4. Cargar modelo versionado
→ 2. Realizar predicción
```

sin ejecutar la competencia de entrenamiento.
