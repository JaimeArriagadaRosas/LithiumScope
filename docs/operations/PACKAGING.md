# Distribución y empaquetado de LithiumScope

## Tres artefactos diferentes

**1. Código de la aplicación.** Configurado por `pyproject.toml`, con `setuptools`, código Python en `src/lithiumscope/`, entradas `lithiumscope` y `lithiumscope-inspect`. `main.py` es el arranque del checkout y no constituye por sí solo un instalador independiente.

**2. Bundles de modelos.** Los GitHub Releases aportan assets terminados en `-artifacts.zip`. Cada paquete contiene manifest y modelos con sus metadatos; la opción 4 descarga y valida el ZIP, verifica hashes, deserializa ambos modelos y actualiza el registro local únicamente después de instalar correctamente.

**3. Demostración científica.** El generador integrado produce PDF, Excel, métricas y manifiesto por `run_id`. No equivale a un modelo entrenado ni debe fabricarse un PDF manual para presentarlo como salida del menú.

## Estado publicado

El GitHub Release `lithiumscope-training_20261106_111653_m0300` fue publicado el **14 de noviembre de 2026**, con:

- M1: SVM ampliado, entrenamiento `training_20261106_111653_m0300`.
- M2: CatBoost histórico, entrenamiento `training_20260920_195933_m0300`.
- Archivo `LithiumScope-Demostracion-Integrada-20261113.pdf` como asset separado.

Referencia: [release público](https://github.com/JaimeArriagadaRosas/LithiumScope/releases/tag/lithiumscope-training_20261106_111653_m0300). **La activación local es independiente del estado publicado.** M2 espectral v4 (170 features) no es el modelo M2 que integra ese paquete.

## Comprobaciones de empaquetado

Antes de declarar instalable un artefacto:

1. Construir sdist/wheel en un directorio de pruebas; revisar contenido, licencia y metadatos.
2. Instalar el wheel en un entorno virtual **fuera de la raíz del repositorio**.
3. Comprobar los scripts de consola, la resolución de `config/*.yaml` y rutas de datos y salidas.
4. Confirmar dependencias extras, compatibilidad de Python y ausencia de acceso accidental a datos locales.
5. Verificar en entorno limpio la instalación de bundle y predicción M1 y M2 sin reentrenar.
6. Registrar hashes, `run_id`, procedencia y limitaciones; no atribuir a ambos modelos un mismo commit si provienen de entrenamientos diferentes.
7. Ejecutar tests y revisar los resultados antes de promover un release.

**Distribución independiente del checkout:** el paquete sigue el layout `src/`. El hook acotado de `setup.py` incorpora durante la construcción una copia de los archivos YAML/JSON del `config/` canónico en `lithiumscope/_defaults/`, sin mantener una segunda fuente editable de configuración. Una instalación desde wheel utiliza `~/.lithiumscope/` como espacio local de datos y configuración (o `LITHIUMSCOPE_HOME` si se establece). La primera carga de configuración copia los valores predeterminados que falten sin sobrescribir modificaciones del usuario. En el checkout fuente se conservan las rutas originales. Se verificó la construcción e instalación de un wheel en un directorio temporal aislado, incluyendo el acceso a las seis configuraciones y al manifiesto espacial JSON; la ejecución completa de entrenamiento científico requiere una validación independiente con los datasets correspondientes.

## Exclusiones

No incluir automáticamente `data/raw`, imágenes satelitales masivas, resultados de simulaciones, modelos experimentales no seleccionados, secretos, cachés o logs privados en un wheel o bundle. Mantener la evidencia científica local sin eliminarla.

## Publicación

El flujo de publicación debe ser explícito: selección científica, manifests, verificación de inferencia, generación del paquete, revisión y solo después tag/assets autorizados. No mover ni sobrescribir tags o releases históricos para corregir documentación.

## Lecturas

- [Modelos publicados](../models/MODEL_RELEASES.md)
- [Versionado](../models/VERSIONING.md)
- [Predicción integrada](../models/PREDICTION_WORKFLOWS.md)
- [Reproducibilidad](REPRODUCIBILITY.md)
