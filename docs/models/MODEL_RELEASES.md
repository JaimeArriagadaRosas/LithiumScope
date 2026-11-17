# Modelos publicados

**Estado público verificado a 17-11-2026:** el release más reciente disponible en GitHub es [lithiumscope-training_20261106_111653_m0300](https://github.com/JaimeArriagadaRosas/LithiumScope/releases/tag/lithiumscope-training_20261106_111653_m0300), publicado el 14-11-2026. Contiene M1 SVM ampliado (`training_20261106_111653_m0300`), M2 CatBoost histórico (`training_20260920_195933_m0300`), un ZIP de modelos y un PDF de demostración como segundo asset. **La procedencia de ambos modelos es distinta y debe conservarse como tal.** M2 espectral v4 de 170 features sigue siendo experimental, no el modelo publicado. La selección activa en un equipo local puede seguir siendo otra; publicar un release no activa automáticamente pesos locales.

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
