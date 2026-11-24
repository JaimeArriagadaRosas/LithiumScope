# Progreso de implementacion

- Base: e12b877699fe97f301e2c87d07f4833084e1306f.
- Commit documental: 0cdbed9.
- Infraestructura inicial: scripts/run_tests.py separa logs y evidencia JUnit, asigna IDs unicos, usa directorios temporales y propaga el codigo de salida. No reemplaza aun el logger de la aplicacion.
- Verificacion: py_compile aprobado; tres pruebas unitarias con unittest y subprocess simulado aprobadas (XML ausente o invalido, ejecuciones distintas y codigo de salida, fallo de lanzamiento). Falta ejecucion pytest real y pruebas persistentes de integracion.
- Baseline de pytest: instalacion iniciada con el codigo original; revisar results/tests/baseline/install.log, pytest.log, junit.xml y exit-code.txt. No declarar baseline aprobado sin resultados.
- Pendiente: lectura completa de los diagnosticos prioritarios, completar trazabilidad y contrastar con implementacion real; no mover paquetes ni alterar protocolos cientificos antes de ello.
- No se modificaron modelos, releases ni el clon de defensa. Sin push ni merge.

## Baseline y primer bloque CLI
- Baseline original: 442 passed, 11 skipped, 8 warnings, 56.70 s, exit 0. results/tests/baseline/junit.xml.
- Consola: menu principal 1/2/5, suborden de prediccion, pausa de preboot, cancelacion de entrenamiento y confirmacion de instalacion. Entry point instalado unificado con ciclo de vida.
- Test de contrato: 8 fallos reproducidos antes; regresion posterior 18 passed, 5 skipped. Run 20261124T200155Z_113122b142a14e7e8992d4ffdb17208b.
- Pendientes: consulta de resultados, laboratorio, presentacion detallada, progresos, EDA, reorganizacion y correcciones cientificas.
- Se revisaron los 57 parrafos de diagnostico y la especificacion de consola; lectura de antecedentes y complementos pendiente.

## Consulta de evidencias
- Resultados de solo lectura: PDF y XLSX existentes, modelos activos y candidatos, manifiestos EDA, tests y simulaciones. PDF: comprobacion estructural preliminar; no certifica toda su integridad. Pendiente validar hashes declarados y compatibilidad EDA efectiva.
- Acceso separado conservado: python -m lithiumscope.cli.candidate_admin bundle | evaluate | promote --confirm. La promocion exige opcion explicita; no se ejecuto ninguno de estos comandos. Esta via CLI es una decision de implementacion para no agregar opciones permanentes.
- Regresion: 13 passed, 0 skipped, 0 failed. Run 20261124T200348Z_9049c7f6d1f44a45bf820685b5f77848.

## Contratos científicos iniciales
- M1: concentraciones físicas, política cero explícita, edad como esquema fijo antes de CV. M2: filtro físico independiente.
- Tests antes: 4 fallos. Después: 36 aprobadas sin fallos (20261124T200524Z_10aa90efcf90439b91fe4522f4c40b61).
- Commit 73ddaee. No se cambian modelos históricos; falta validar datos reales y semántica por fuente.

## EDA descriptivo M1/M2
- Ambos reutilizan sus preparadores/lectores de entrenamiento y no entrenan estimadores.
- M1: perfiles antes/después por fuente, etapas, filas excluidas, categorías, estadísticas, correlaciones y gráficos.
- M2: características configuradas, procedencia de manifiesto, perfiles y calidad de escena (no calidad local del parche).
- Firmas de datos/código/configuración y hashes por artefacto. Reutilización de evidencia descriptiva intacta; reconstrucción si se altera.
- Pruebas sintéticas: 10 aprobadas (20261124T200728Z_38175446fb484585af1d10bf87446c7c).
- Aún no es EDA de desarrollo independiente; no se utiliza automáticamente para selección o ajuste.
- No se adquirieron datasets ni se ejecutó EDA real. M2 v4 está condicionado a los manifiestos y matrices congelados que no están versionados.
