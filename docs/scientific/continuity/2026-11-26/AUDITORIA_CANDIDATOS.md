# Auditoría adicional M1 — datos entregados

Recalculada sobre las dos vistas armonizadas del EDA (2998 y 8234 filas),
la fuente combinada bruta congelada y los 244 grupos de referencia.
No se descargó de nuevo ni se modificó la fuente productiva.

| Fuente | Estado de cribado | Filas |
| --- | --- | ---: |
| Andean Arc | convergent_review | 3 |
| Andean Arc | excluded_screening | 2995 |
| Granitos/riolitas | convergent_review | 316 |
| Granitos/riolitas | excluded_screening | 1456 |
| Granitos/riolitas | high_transfer_review | 69 |
| Granitos/riolitas | other_domain_review | 6393 |

Total margen convergente: 319, 76 grupos espaciales, 73 grupos nuevos,
60 publicaciones, Li 1.534–517 ppm. Ninguna >696 ppm.
Transferencia alta: 69, 14 grupos nuevos, 13 publicaciones, Li
696.990225–6201.466682 ppm, 53 >1000 ppm. Todas intraplaca.
En ambos subconjuntos la totalidad tiene método y CRS consignados como
desconocidos en la vista disponible. Esto no demuestra que las mediciones
sean inválidas; requiere recuperar la procedencia antes de incorporarlas.

Alertas solapadas en toda la vista combinada: ID 442; publicación/muestra
632; coordenada/Li exactos 3792; proximidad/identidad geoquímica 3310.
Una alerta no equivale a prueba de duplicado; el cribado conservador excluye
ambos miembros de coincidencias candidatas y los deja auditables.
Reglas aproximadas: distancia máxima por eje .001°, Li dentro de 1% o .05 ppm,
nombre de muestra normalizado coincidente o al menos 5 mediciones comunes
todas dentro de 2% o .001 en sus unidades originales. Estas tolerancias son
reglas de cribado de investigación, no incertidumbres instrumentales.

El script versionado reproduce las reglas sobre vistas EDA verificadas:
```powershell
python -m lithiumscope.experiments.data_lab.candidate_audit
```
Genera auditoría por fila, resumen y perfiles de desplazamiento de dominio.
Los 22 elementos/óxidos observados constituyen una comprobación inicial de
cobertura, no la totalidad del esquema de 33 predictores preparado.
El informe no aprueba automáticamente ninguna muestra. Fase 3 permanece
condicionada a revisión documental de procedencia, compatibilidad y plan
de separación espacial/bibliográfica; no ajustar ni seleccionar con TEST.
Si no se verifica, conservar RF actual o evaluar el balanceo ya documentado
sin repetir la campaña completa. No se modifican M2 ni modelos activos.
