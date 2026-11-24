# Protocolo prospectivo de evaluación externa

**Estado:** implementación funcional comprobada únicamente con datos sintéticos. No hay cohorte externa certificada registrada. No hay autorización de promoción.

La cohorte histórica de 706, la ampliación de 783, el conjunto M1 de 1120 y el de 3420, las muestras conocidas de demostración y todos los sitios/publicaciones consultados durante EDA, ajustes y análisis son desarrollo/exposición. No pueden convertirse retrospectivamente en un holdout ciego.

## 1. Registro antes de conocer etiquetas

El custodio conserva las concentraciones de laboratorio por separado y entrega solamente el SHA-256 del CSV de etiquetas. El responsable de datos documenta el inventario completo de exposiciones y la independencia de fuente/sitio. Identificadores canónicos deben usar el mismo sistema en todas las tablas.

- sitios.csv: sample_id, source_id (publicación o fuente original identificada), site_id (identidad física), spatial_group, latitude, longitude, region. M1 añade las características geoquímicas de entrada, nunca Li ni derivados de la etiqueta. M2 añade image_path, image_sha256, scene_id, sidecar_sha256; v4 requiere rededge_patch_sha256 y scene_band_hash previamente verificados. Los cubos red-edge y sus descriptores deben existir en la caché local de inferencia; no se adquieren durante score.
- exposiciones.csv (uno o varios): sample_id, source_id, site_id, spatial_group, latitude, longitude; M2 también scene_id, image_sha256. Debe incluir **todas** las fuentes consultadas, no solamente los conjuntos ajustados. Identidades desconocidas bloquean el registro. Un CSV incompleto no se convierte en certificación por el hecho de que el código acepte las filas suministradas.
- revision.json: reviewer, label_custodian, documentary_basis (referencia verificable a los documentos de campo, laboratorio, identidad y custodia), complete_exposure_inventory, independent_source_and_site, labels_unconsulted, site_identity_verified, coordinates_wgs84_verified. Los cinco campos finales deben ser verdaderos tras revisión humana, no completados por inferencia.
- Modelos locales de confianza: candidato y referencia, congelados por SHA-256. Para M2 se exige que ambos tengan el mismo endpoint Li en ppm. Los umbrales operativos se conservan de cada bundle, no se ajustan usando el holdout.

Ejemplo de interfaz (rutas ilustrativas; no ejecutar con archivos inexistentes):

\`\`\`powershell
.venv\Scripts\python.exe -m lithiumscope.prediction.candidates.holdout register results/diagnostics/holdouts/externo01 --model-group model_1 --sites-path sitios.csv --exposure-paths exposiciones_m1.csv exposiciones_m2.csv exposiciones_demo.csv --reference referencia.joblib --candidate candidato.joblib --review-path revision.json --label-commitment-sha256 SHA256_DEL_CUSTODIO --minimum-distance-m 1000
\`\`\`

Para M2 se añade \`--label-cutoff-ppm\` con el endpoint común predefinido. El ejemplo de buffer es ilustrativo; su elección debe justificarse y registrarse antes de evaluar. Política provisional por defecto: 100 muestras, 3 regiones, 5 grupos; M2 20 positivos y 20 negativos. Bootstrap agrupado emparejado de 500 iteraciones con semilla 42. Estos límites no son prueba de potencia estadística; requieren aprobación metodológica.

## 2. Inferencia sin etiquetas

\`\`\`powershell
.venv\Scripts\python.exe -m lithiumscope.prediction.candidates.holdout score results/diagnostics/holdouts/externo01 --trusted-local-bundles
\`\`\`

La opción declara que los archivos joblib son artefactos locales confiables; nunca deserializar archivos de terceros desconocidos. Se verifica registro, modelos, fuentes e imágenes; se conservan las dos predicciones por ID y sus advertencias. No se lee el archivo de etiquetas ni se modifica el activo.

## 3. Revelación y evaluación

El custodio entrega etiquetas.csv con sample_id, source_id, li_ppm, unit=ppm, analytical_method. Debe corresponder exactamente a la huella comprometida. No se permiten etiquetas no finitas, negativas ni métodos desconocidos.

\`\`\`powershell
.venv\Scripts\python.exe -m lithiumscope.prediction.candidates.holdout evaluate results/diagnostics/holdouts/externo01 --labels-path etiquetas.csv
\`\`\`

Se sella la revelación antes de interpretar el CSV, incluso si termina siendo inválido; no se permiten nuevas puntuaciones ni una segunda evaluación en ese registro. Se comparan pares completos, sin descartar abstenciones para obtener un resultado favorable. M1 calcula regresión y diferencias RMSE/MAE mediante bootstrap por grupos; M2 conserva métricas a 0.5 y a los umbrales congelados y estima incertidumbre emparejada. La salida solo queda lista para revisión independiente; la aprobación y activación automáticas son siempre falsas.

## Limitaciones

Los hashes detectan cambios de archivos, no demuestran que las etiquetas nunca se consultaron, ni prueban cronología o autoría frente a un actor con acceso al filesystem. El inventario de exposición y la cadena de custodia necesitan evidencia documental revisada independientemente. Bloquear reutilización local no equivale a una custodia externa criptográficamente autenticada. Una evaluación de M2 v4 requiere preparación offline previa de imágenes y red-edge. Los tests ejercitan M1 y M2 v3 sintéticos; no se evaluaron modelos reales ni se creó una cohorte externa. La promoción permanece bloqueada.
