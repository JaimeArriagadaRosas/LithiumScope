# LithiumScope — Cierre de CI y continuidad para el siguiente empaquetado

**Trabajo (hora de Chile):** 13 de noviembre de 2026. **Alcance:** reparar los siete jobs de GitHub Actions y documentar su causa, sin alterar pesos, datos, releases activos, ZIP, PDF ni modelos.

## 1. Diagnóstico confirmado del fallo inicial

Fuente: [GitHub CI fallida #43538138870](https://github.com/JaimeArriagadaRosas/LithiumScope/actions/runs/43538138870), commit 35348288a945f5bc1a989915939655627a675e9e.

| Check | Fallo en los logs | Corrección |
|---|---|---|
| lint (Ubuntu 3.12) | Ruff OK, pero tamaño de scripts/scientific/m1_dual_release_diagnostics.py = 329 líneas (máximo 300) | Separar funciones métricas puras, sin ampliar límites |
| test Ubuntu 3.11 | ModuleNotFoundError: No module named scripts | Añadir la raíz al pythonpath de pytest |
| test Ubuntu 3.12 | Mismo fallo en dos archivos de pruebas | Igual |
| test Ubuntu 3.13 | Mismo fallo de importación | Igual |
| test Windows 3.11 | Mismo fallo de importación | Igual |
| test Windows 3.12 | Mismo fallo de importación | Igual |
| test Windows 3.13 | Mismo fallo de importación | Igual |

Las seis variantes no llegaron a ejecutar sus pruebas; fallaron durante la colección. No se obtuvo evidencia de un defecto del modelo CatBoost, del SVM ni del ZIP.

## 2. Cambios hechos, con alcance controlado

- pyproject.toml: pytest incluye tanto src como la raíz del repositorio; los extras de desarrollo incorporan xgboost y optuna, necesarios para probar la competencia de seis familias y su optimizador en GitHub.
- scripts/scientific/m1_dual_release_diagnostics.py: funciones met y four movidas a src/lithiumscope/model_1/methodology/dual_release_metrics.py y reimportadas con la misma interfaz. El script queda en 290 líneas.
- tests/scientific/test_m2_rededge_v4.py: se conservan los tests existentes. Los controles adicionales antes no versionados pasan al archivo tests/scientific/test_m2_v4_auxiliary.py, ambos bajo 300 líneas.
- Cinco módulos experimentales previamente sin seguimiento se incorporan selectivamente a Git para que las pruebas no dependan del contenido privado del PC: m2_v4_bundle_candidate.py, m2_v4_inference.py, m2_v4_nested_overfit_audit.py, m2_v4_paired_77_ablation.py y m2_v4_spectral_identity_audit.py.
- Los tests de modelos entrenados y matrices espectrales no versionados se marcan de manera específica como integraciones opcionales de datos locales. Se ejecutan si los archivos existen; en CI se omiten con motivos explícitos. Los controles sintéticos de selección de modelos, esquema y renderizado siguen activos.
- No se eliminaron jobs ni plataformas del workflow; no se deshabilitó Ruff, pytest o el control de tamaño.

- El checkout limpio detect? adem?s un falso fallo num?rico en la prueba de aislamiento de TEST del RF: dos refits variaban aproximadamente 1e-16 en la ?ltima cifra de sus scores. La aserci?n de igualdad exacta se sustituy? por np.allclose con atol=1e-12 y rtol=0, manteniendo la verificaci?n de que las etiquetas TEST no modifican TRAIN ni predicciones.

## 3. Simulación de CI limpia y verificaciones

Se construyó un worktree temporal exclusivamente de QA en D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-CI-Validation-20261113. Solo tiene archivos versionados y los cambios seleccionados, sin datasets ni los .joblib que Git excluye.

| Evidencia | Resultado |
|---|---|
| Ruff en código fuente y tests | OK |
| Contrato de tamaño (máximo 300 líneas) | OK, ningún módulo nuevo excede |
| Suite en Windows, Python 3.12, checkout limpio inicial | 439 aprobadas, 10 omisiones justificadas, 0 fallos |
| Suite original local con los artefactos de usuario | Pruebas dirigidas 27/27 aprobadas antes de los cambios finales |
| Matriz final de siete jobs en GitHub | Pendiente de confirmar tras push |

Los skips por datos locales evitan comparar el checkout público con los resultados privados y no sustituyen las pruebas unitarias. Es una limitación explícita, no una validación científica independiente.

## 4. Conservación y procedencia del release vigente

| Componente | Entrenamiento | Situación |
|---|---|---|
| M1 SVM ampliado | training_20261106_111653_m0300 | Seleccionado, 3.420 muestras y mejora exploratoria sobre 1.004 casos físicos comunes |
| M2 CatBoost histórico | training_20260920_195933_m0300 | Seleccionado, 706 observaciones, 126 variables, inferencia operativa |
| M2 Random Forest v4 | training_20261113_200558_m0300 | Experimental: 170 variables; no publicado porque su inferencia todavía requiere las cuatro bandas adicionales y su superioridad no quedó demostrada |
| PDF original | demonstration_20261113_212844_m0300 | 28 páginas, creado por el generador integrado de LithiumScope con los dos modelos seleccionados |

Release publicado: [lithiumscope-training_20261106_111653_m0300](https://github.com/JaimeArriagadaRosas/LithiumScope/releases/tag/lithiumscope-training_20261106_111653_m0300).

- Tag original asociado al commit 3534828; la nueva corrección CI no moverá el tag.
- ZIP: lithiumscope-training_20261106_111653_m0300-artifacts.zip; SHA-256 606200b71d61a096f79031c6f63fa7a2eefcafbbfe48758efa1fe6cb63aa2400.
- PDF: LithiumScope-Demostracion-Integrada-20261113.pdf; SHA-256 a83eec698d2c52111826a102575dcc85fcf7085305919be77bcfe684f25ed7ac.
- No se entrenaron modelos, no se cambió models/active_models.json ni se sobrescribieron assets del release.

## 5. Trabajo local no versionado, conservado expresamente

El clon principal contiene resultados científicos grandes bajo results/scientific_simulations, modificaciones previas de laboratorio, scripts aún experimentales y archivos .gitkeep borrados en su working tree. No se usó reset --hard, clean, stash ni inclusión indiscriminada en git add. Solo se prepararon archivos relacionados con CI y este cierre.

El worktree temporal se usa únicamente para pruebas de un checkout limpio; no reemplaza el clon científico. Para mañana conviene inventariar y respaldar por separado los artefactos científicos esenciales que Git ignore o no versiona.

## 6. Pendientes para retomar mañana

| Prioridad | Tarea | Condición de cierre |
|---|---|---|
| Alta | Confirmar los siete jobs del nuevo push (Ubuntu y Windows, Python 3.11/3.12/3.13, lint) | CI totalmente verde |
| Alta | Revisión visual/manual del PDF integrado original | Corregir redacción, tablas, mapas y observaciones científicas sin extrapolar conclusiones |
| Alta | Revisar el empaquetado e instalación desde el menú | Instalar ZIP en entorno de prueba y comprobar inferencia M1/M2 y hashes |
| Media | Revisar limitaciones del SVM ampliado M1 | Distinguir 1.004 casos pareados, 610 cambios de fold y capacidad ante Li extremo |
| Media | Continuar M2 espectral v4 | Implementar inferencia auténtica de 170 variables; validar generalización con muestras independientes antes de otro release |
| Media | Inventariar resultados privados del laboratorio | Separar fuentes, datos, caches y reportes necesarios para la tesis |
| Baja | Considerar siguiente release con correcciones | Usar nuevo tag con autorización, jamás mover o reemplazar la publicación existente |

**Regla de continuidad:** mantener seis algoritmos de M2; conservar versiones históricas y trazabilidad; no hacer entrenamientos ni nuevos releases por corregir CI. Este documento registra causas, límites, resultados y pasos reproducibles.
