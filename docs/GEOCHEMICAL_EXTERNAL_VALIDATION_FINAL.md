# Estado final de la auditoría externa GEOROC para LithiumScope

**Alcance:** revisión reproducible de los 3.220 candidatos de roca total con Li en la compilación oficial del arco andino.

- Candidatos totales: **3220**.
- Superaron filtros automáticos preliminares: **1450**.
- Publicaciones/códigos de referencia distintos en ese subconjunto: **59**.
- Referencias con DOI registrado: **58**.
- Muestras admisibles para evaluar exactitud externa: **0**.

## Comprobaciones efectuadas
Cruce con registros primarios, Li numérico, material, coordenadas, duplicación de IDs, profundidad de perforación, litología, distancia a 706 puntos de entrenamiento y tabla bibliográfica oficial.
Se conservan en el ledger todos los registros, el estado individual y el motivo de exclusión o revisión.

## Limitación científica
La tabla de referencias contiene datos bibliográficos, **no demuestra el método específico de Li ni la independencia de las publicaciones**. Los datos de roca y el score de una escena de 640 m miden objetos distintos; requieren verificación geológica de correspondencia espacial.
Una ubicación a más de 100 km del entrenamiento es solo un filtro de proximidad, no un criterio suficiente de independencia.
No se calculan AUC, AP ni BA externas hasta cumplir las condiciones de evidencia. Cuatro inferencias piloto no son una prueba de generalización.

## Archivos
- `external_validation_decision_ledger.csv`: decisión individual de 3.220 registros.
- `all_reviewable_citations.csv`: referencias distintas de los candidatos con metadatos disponibles.
- `final_evidence_status.json`: resumen, bloqueos y SHA-256.
- Los CSV originales GEOROC, fuentes PDF descargadas, el catálogo Mamani09 y simulaciones M2 se conservan sin alteraciones.

## Trabajo externo no certificable automáticamente
Consulta de artículos completos y suplementos analíticos, identificación de la técnica específica para Li, revisión de equivalencia de afloramiento con los píxeles y auditoría exhaustiva de solapamiento bibliográfico. Estos controles requieren evidencia primaria, no sustitutos o valores inferidos.

## Corrección esencial: comparación con el corpus Mamani09 completo
La auditoría inicial comparó GEOROC solamente con las **706 filas seleccionadas para M2**. Esta versión también utiliza las **1.788 filas del archivo original Mamani09_Table_DR2**, evitando declarar independientes muestras que no llegaron al subconjunto de entrenamiento.
- 426 registros GEOROC dentro de 250 m de alguna coordenada Mamani09; **17** también coinciden en Li dentro de tolerancia (señal de posible muestra compartida, no identidad probada).
- 1.346 dentro de 50 km de Mamani09 completo y 1.518 dentro de 100 km; frente a los 1.190 previamente próximos al subconjunto M2, el control del corpus total descubre riesgos adicionales.
- De los 1.450 candidatos preliminares en revisión, **1.307** quedan a 100 km o más de Mamani09 completo. **Ninguno está todavía aprobado**.
- Trazabilidad completa en `results/scientific_simulations/external_geochemistry_audit/full_mamani_contamination_audit.csv` y `full_mamani_contamination_audit.json`; script `src/lithiumscope/experiments/georoc_full_mamani_audit.py`.
**Estado científico:** robustez computacional y simulaciones terminadas; exactitud predictiva M2 sobre etiquetas externas independientes NO demostrada. No sustituir la revisión geológica y metodológica por el criterio de distancia o por el simple año de publicación.
