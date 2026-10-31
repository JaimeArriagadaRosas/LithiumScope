"""Close the original scientific simulation scope; do not enlarge GEOROC research."""
from pathlib import Path
import pandas as pd,json,hashlib
base=Path("results/scientific_simulations")
m1=pd.read_csv(base/"batch_m1/batch_m1.csv")
m2_files=[
 ("2023-2025",base/"batch_m2/batch_m2.csv"),
 ("2025 seasonal/shifted",base/"campaign_2025/batch/batch_m2.csv"),
 ("2024 seasonal",base/"seasonal_2024/batch/batch_m2.csv"),
 ("2024-2025 multiscene",base/"temporal_multiscene/batch/batch_m2.csv"),
 ("same-scene spatial",base/"controlled_spatial/all_684_scenario_metrics.csv"),
]
rows=[];hashes={}
for name,path in m2_files:
 x=pd.read_csv(path)
 rows.append(dict(campaign=name,total=len(x),scored=int(x.status.eq("SCORED").sum()),
                  expected_abstained=int(x.status.eq("ABSTAIN_ALL_BAND_NODATA").sum()),
                  unexpected=int((~x.status.isin(["SCORED","ABSTAIN_ALL_BAND_NODATA"])).sum())))
 hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
report=pd.DataFrame(rows)
out=base/"final_scope_closure";out.mkdir(parents=True,exist_ok=True)
report.to_csv(out/"m2_campaign_inventory.csv",index=False)
leak=json.loads((base/"leakage_safe_m2/experiment_manifest.json").read_text())
spatial=json.loads((base/"controlled_spatial/spatial_robustness.json").read_text())
old=json.loads((base/"academic_evidence/recovered_training_validation.json").read_text())
counts={"m1_scenarios":len(m1),"m2_scenarios":int(report.total.sum()),
"m2_scored":int(report.scored.sum()),"m2_expected_abstained":int(report.expected_abstained.sum()),
"m2_unexpected":int(report.unexpected.sum()),"spatial":spatial,"corrected_cv":leak,
"historical":{k:old[k] for k in ["dataset_hash_matches","label_match","split_fold_sizes_match",
"split_group_overlaps","metrics_recomputed","fold_means","baseline_fold_means"]},
"external_accuracy":"NOT_EVALUATED","files_sha256":hashes}
(out/"closure_manifest.json").write_text(json.dumps(counts,indent=2,ensure_ascii=False),encoding="utf8")
lines=["# LithiumScope — Cierre de la etapa original de simulaciones científicas",
"","> **Alcance cerrado:** ejecución, auditoría y síntesis de robustez de M1/M2. Se detiene aquí la expansión de investigación sobre GEOROC.",
"","## Inventario verificado", "", "| Campaña M2 | Total | Puntuadas | Abstenciones previstas | Inesperadas |",
"|---|---:|---:|---:|---:|"]
for item in rows:lines.append("| {campaign} | {total} | {scored} | {expected_abstained} | {unexpected} |".format(**item))
lines+=["",f"**M2 total: {counts['m2_scenarios']} escenarios**; {counts['m2_scored']} puntuados, {counts['m2_expected_abstained']} abstenciones previstas y {counts['m2_unexpected']} fallos inesperados. **M1: {len(m1)} escenarios**.",
"","## Robustez espacial en la misma escena","",
"36 parches, cuatro sectores, distancias 400 y 1000 m, cuatro direcciones; 684 evaluaciones M2 en la campaña espacial. Cambios absolutos medios del score basal: 0,081121 a 400 m y 0,093536 a 1000 m, con grandes variaciones locales. La ventana de 640 m hace dependientes algunos desplazamientos. No mide exactitud en litio.",
"","## Robustez temporal y perturbaciones",
"Las campañas 2023–2025 y multiescena 2024–2025 comprobaron variación de score entre años, estaciones y sensores; los efectos atmosféricos, temporales y del satélite no están aislados causalmente. La sensibilidad pareada detectó efectos relevantes de brillo, ruido y pérdida de píxeles. No existe una relación monótona universal del score con distancia, año o estación.",
"","## Validación cruzada del Modelo 2",
"El histórico de CatBoost corresponde a un entrenamiento distinto (umbral q75 global, selección de umbral OOF): media de cinco folds ROC-AUC 0,573398; AP 0,308807; BA a umbral 0,5 0,498309. Las particiones se pudieron reconstruir por grupos y tamaños, no se recuperaron los índices originales exactos.",
"El experimento aislado corregido (CatBoost 240 iteraciones, profundidad 5, sin búsqueda) obtuvo cinco folds agrupados, 66 grupos, cero solapamientos: media ROC-AUC 0,546935; AP 0,306160; BA 0,513007. q75 por entrenamiento de fold y umbral de operación aprendido solo con predicciones OOF internas. No comparar sus cifras como prueba del efecto de una única corrección, ya que cambió la configuración.",
"**Interpretación:** evidencia de capacidad discriminativa interna débil/modesta, no demostración de predicción fiable de litio ni precisión prospectiva externa.",
"","## Investigaciones separadas y no requeridas para cerrar este alcance",
"GEOROC: 3.220 candidatos descargados y auditados, 1.450 tras filtros iniciales, 1.307 alejados al menos 100 km del corpus Mamani09 completo. La comprobación de 210 valores contra suplemento primario no certifica representatividad espectral de superficie. **Cero etiquetas geoquímicas externas aprobadas**, por lo que no se calcularon métricas externas. Se conservan sus archivos y documentos como línea de investigación futura, sin impedir el cierre de la batería original.",
"","## Conclusión",
"**La etapa de simulaciones científicas y análisis de robustez queda cerrada como experimento reproducible, no como validación externa de exactitud**. No se harán más descargas, verificaciones de publicaciones ni campañas adicionales dentro de este alcance. La rama original y los modelos entrenados quedan intactos.",
"","## Reproducibilidad",
"`final_scope_closure/m2_campaign_inventory.csv` y `closure_manifest.json` incluyen conteos y hashes por origen. Los detalles por experimento siguen en sus carpetas originales bajo `results/scientific_simulations/`."]
Path("docs/SCIENTIFIC_SIMULATION_FINAL_CLOSURE.md").write_text("\n".join(lines)+"\n",encoding="utf8")
print("CLOSURE",json.dumps({k:v for k,v in counts.items() if k.startswith("m1_") or k.startswith("m2_")},indent=2))
assert counts["m2_unexpected"]==0
assert spatial["scored_patches"]==36
assert leak["rows"]==706
