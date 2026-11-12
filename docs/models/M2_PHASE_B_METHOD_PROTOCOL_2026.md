# LithiumScope — Fase B: controles metodológicos de M2

Fecha: 2026-11-12. Rama: `feat/scientific-simulation-lab`. Alcance: **validación e infraestructura experimental**, sin entrenamiento, sin activar modelos ni modificar el menú interactivo.

## Motivación y antecedente

La auditoría de Fase A (`results/scientific_simulations/m2_phase_a_20261111/`) estudió 783 muestras y encontró 593 sitios de coordenadas (redondeo a 5 decimales), 25 sitios con clases de Li mezcladas al corte descriptivo 21,6 ppm, 77 etiquetas GEOROC provisionales sin verificación de campo y 635 observaciones OOF con un identificador de escena Sentinel-2 presente también en su train fold. Ninguna de esas cifras demuestra por sí sola una fuga de *target* o un error de código.

En especial, la validez física de un parche Sentinel-2 de 640 m como supervisor de una muestra geoquímica puntual **no está certificada** por el éxito de los tests, la nubosidad/SCL ni por poseer una publicación bibliográfica.

## Implementado (sin cambiar el entrenamiento habitual)

El paquete `src/lithiumscope/model_2/methodology/` incorpora:

1. `contracts.py`: valida identidad de muestra, ruta del parche, Li y coordenadas finitas, grupo espacial y procedencia. Agrega identificadores *de cribado* sitio/parche/escena/publicación y marca contradicciones alto/bajo. Distingue `FIELD_VERIFIED_SURFACE`, `PROVISIONAL_SURFACE`, `HISTORICAL_UNVERIFIED` y `EXCLUDE_SUBSURFACE_OR_INVALID`. **La ausencia de metadatos históricos no implica que sus 706 etiquetas sean falsas:** `HISTORICAL_UNVERIFIED` significa que este manifiesto no acredita verificación de campo. Tampoco basta escribir `FIELD_VERIFIED_SURFACE` sin expediente científico real.
2. `spatial.py`: audita directamente los cinco folds, excluye solapamientos de grupo o sitio, calcula distancias geodésicas y dependencias de escena/publicación. Simula `buffered_train_preview`, retirando **solo filas de train**, jamás alterando índices de test. La exclusión de escenas completas es opcional y se informa por separado, no se aplica automáticamente.
3. `domain.py`: calcula límites percentiles y huella de 126 características con **datos del train del fold**, después evalúa su test sin recalcular los límites sobre él. Es un control descriptivo de dominio, no una calibración de probabilidad de Li ni garantía de detección de anomalías.
4. `decision.py`: permite congelar un umbral de score definido **solo con predicciones OOF internas del train exterior**, con controles de pertenencia de filas, grupos y folds. El procedimiento de entrenamiento que produzca esas predicciones deberá cumplir además la independencia real de los *fits*: el validador de metadatos no puede garantizarlo por sí solo. No ajusta un calibrador ni se integra al menú.
5. `audit.py`: CLI de solo lectura que genera CSV de trazabilidad y aislamiento, perfiles OOD por fold, vista previa de los índices de train tras buffer, resumen JSON y motivo de rechazo de una validación estricta. `--strict` devuelve código 2 después de guardar la evidencia cuando persisten problemas metodológicos.

Los componentes son **opt-in** y están preparados para la Fase C. No están conectados a la competencia principal, para no invalidar silenciosamente el anclaje histórico, la firma científica ni los checkpoints.

## Ejecución de preflight del entrenamiento experimental real

Desde la raíz del repositorio:

```powershell
python -m lithiumscope.model_2.methodology.audit `
  --manifest data/processed/model_2/vnext_expanded/training_manifest.csv `
  --features data/processed/model_2/vnext_expanded/spectral_features.csv `
  --folds results/model_2/runs/training_20261111_170832_m0300/manifests/scientific_folds.json `
  --output results/scientific_simulations/m2_phase_b_20261112/real_preflight `
  --buffer-m 5000
```

Si se agrega `--strict`, el comando no construye nuevas particiones: registra problemas y termina con código de salida 2 si alguno sigue presente.

### Resultados reales del preflight 5 km

| Indicador | Resultado |
|---|---:|
| Muestras / sitios de cribado | 783 / 593 |
| Sitios y filas con etiquetas discordantes | 25 sitios / 77 filas |
| Metadatos históricos sin verificación explícita | 706 |
| Referencias bibliográficas no identificadas en el manifiesto histórico | 706 |
| GEOROC con estatus provisional | 77 |
| Metadatos explícitamente verificados en terreno | 0 |
| Test con otro train a menos de 5 km | 50 |
| Test que comparte ID de escena con train | 635 |
| Test que comparte referencia bibliográfica con train | 63 |
| Train mínimo al quitar muestras dentro de buffer | 595 |
| Train mínimo al quitar además escenas compartidas | 234 |
| Alertas OOD usando los límites del train de cada fold | 24 |
| Resultado del gate estricto | NO PASA, por motivos explícitos |

**Interpretación:** cero verificados en este manifiesto no implica cero sitios verificables en origen. Muchas de las dependencias detectadas corresponden a reutilización de un mismo identificador de escena y pueden abarcar píxeles separados; se requiere inspección de independencia real, no equipararlas a una fuga confirmada. Los 50 contactos a menos de 5 km son observaciones de test con un entrenamiento cercano, no 50 errores de programación. El buffer es una hipótesis metodológica para un experimento nuevo; la distancia apropiada todavía debe justificarse científicamente.

Los índices históricos originales y las etiquetas OOF siguen intactos. Es incorrecto comparar directamente una futura evaluación con buffer contra los resultados anteriores como si fuera el mismo protocolo, y también es incorrecto usar los diez casos para seleccionar un buffer, una transformación o un umbral.

## Pruebas automatizadas

`tests/scientific/test_m2_methodology_phase_b.py` verifica identidad, estatus de verificación, etiquetas contradictorias, rechazo de folds con grupos/sitios compartidos, escenas vacías, buffer que solo recorta train, conservación de test, perfiles espectrales congelados y exclusión de observaciones externas al elegir umbrales. Se ejecuta también la suite existente, sin alterar tests previos.

## Fase C propuesta (no ejecutada)

1. Formalizar un criterio de pertenencia geológica por sitio/parche: publicación, GPS, roca expuesta, SCL y compatibilidad de la escala de imagen. Priorizar revisión manual de los 25 sitios discordantes y de los 77 nuevos GEOROC.
2. Realizar **ablaciones pareadas** con protocolos congelados: base 706, +77 provisional, reducción de pesos por sitios repetidos, separación con buffers predefinidos (p.ej. 1/5/10 km) y filtros que exijan evidencias de campo. Cambia solo un componente experimental por comparación.
3. Evaluar discriminación, calibración, recall de Li alto, error por región/escena/publicación y abstención OOD. Hacer ajustes de hiperparámetros, calibradores y cortes solo en CV anidada del train; no aprovechar los diez casos en la selección.
4. Generar reportes con incertidumbre por sitio/grupo y advertencia de dependencia entre escenas, y conservar un test externo realmente nuevo para confirmación. **Ninguna salida de esta Fase B demuestra aún mejora predictiva.**

No se toca M1, las descargas, el menú, el registro de modelos activos ni resultados anteriores; no hay push.
