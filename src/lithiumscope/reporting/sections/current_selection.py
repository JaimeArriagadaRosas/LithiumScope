"""Render the already snapshotted current-model evidence, never read training files."""
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak

from lithiumscope.reporting.primitives import paragraph, table


def _number(value):
    if value is None:
        return "N/D"
    try:
        return f"{float(value):.4f}"
    except (TypeError,ValueError):
        return str(value)


def build_current_selection(evidence,styles):
    out=[PageBreak(),paragraph("Anexo científico — modelos utilizados",styles["LS_H1"]),
         paragraph("Candidatos de desarrollo; no se certifica un release ni una validación externa independiente.",styles["LS_Body"])]
    for group,model in evidence["models"].items():
        out.append(paragraph(f"{group.upper()} — {model.get('algorithm')} | {model.get('run_id')}",styles["LS_H2"]))
        eda=model.get("eda")
        if eda:
            out.append(paragraph(f"EDA ligado al entrenamiento: {eda['run_id']}; {eda['rows_prepared']} filas, {eda['features']} variables. Integridad verificada.",styles["LS_Small"]))
        else:
            out.append(paragraph("EDA ligado a esta versión: no disponible.",styles["LS_Small"]))
        metrics=model.get("metrics",{})
        if metrics:
            out.append(table([["Métrica archivada","Valor"]]+[[k,_number(v)] for k,v in metrics.items() if isinstance(v,(int,float))],
                             widths=[11*cm,5*cm],font_size=7.5))
        ranking=model.get("ranking",[])
        keys=("rmse_mean","mae_mean","r2_mean") if group=="model_1" else ("roc_auc_mean","average_precision_mean","brier_score_mean")
        if ranking:
            out.append(table([["Familia","Estado",*keys]]+[[r.get("label",r.get("algorithm")),r.get("status"),*[_number(r.get(k)) for k in keys]] for r in ranking],widths=[6*cm,2.2*cm,2.6*cm,2.6*cm,2.6*cm],font_size=7))
        treatment=model.get("winner_treatment")
        if treatment:
            data=treatment["data"]
            out.append(paragraph(f"Tratamiento ganador registrado: {treatment['file']}.",styles["LS_Small"]))
            if isinstance(data,dict):
                values=[[k,str(v)] for k,v in data.items() if isinstance(v,(str,int,float,bool))]
                if values:
                    out.append(table([["Parámetro","Valor"],*values],widths=[7*cm,9*cm],font_size=7))
        paired=evidence["paired"][group]
        if paired["state"] in {"paired_development_only","reconstructed_development_only"}:
            data=paired["data"]
            reconstructed=paired["state"]=="reconstructed_development_only"
            reference=data["reference_metrics"] if reconstructed else data["release_metrics"]
            keys=sorted(set(reference) & set(data["candidate_metrics"]))
            title=("Referencia M2 reconstruida — mismos casos, corte Li 21,6 ppm" if reconstructed
                   else "Comparación OOF emparejada exploratoria contra el release")
            out.append(paragraph(title,styles["LS_H2"]))
            out.append(table([["Métrica","Referencia" if reconstructed else "Release","Candidato"]]+[[k,_number(reference[k]),_number(data["candidate_metrics"][k])] for k in keys],widths=[8*cm,4*cm,4*cm],font_size=7))
            if reconstructed:
                old,new=data["published_descriptive_metrics"],data["candidate_native_metrics"]
                out.append(paragraph("Release publicado y candidato — protocolos propios, comparación descriptiva.",styles["LS_Small"]))
                out.append(table([["Métrica","Release publicado","Candidato"]]+[[k,_number(old[k]),_number(new[k])] for k in ("roc_auc","average_precision","brier")],widths=[8*cm,4*cm,4*cm],font_size=7))
            if data.get("gains"):
                out.append(table([["Ganancia positiva","Valor","IC95 inferior","IC95 superior"]]+[[r["metric"],_number(r["observed_gain"]),_number(r["ci95_group_lower"]),_number(r["ci95_group_upper"])] for r in data["gains"]],widths=[4*cm,4*cm,4*cm,4*cm],font_size=7))
        else:
            out.append(paragraph("Comparación OOF emparejada: no disponible para estos artefactos; no se sustituye por métricas ajenas.",styles["LS_Small"]))
    oasis=evidence.get("oasis_m1")
    if oasis:
        out.append(paragraph("Referencia académica OASIS (M1)",styles["LS_H2"]))
        out.append(table([["Muestras","RMSE medio","R² medio"],[str(oasis["rows"]),_number(oasis["rmse_mean"]),_number(oasis["r2_mean"])]],widths=[4*cm,6*cm,6*cm],font_size=7.5))
        out.append(paragraph(oasis["source"]+" "+oasis["protocol"]+" Comparación descriptiva: no son los mismos datos ni folds.",styles["LS_Small"]))
    academic=evidence.get("m1_academic")
    if academic:
        out.append(paragraph("Piloto académico propio (M1)",styles["LS_H2"]))
        out.append(table([["Muestras","Trials/fold","RMSE medio","R² medio"],[str(academic["rows"]),str(academic["trials_per_outer_fold"]),_number(academic["rmse_mean"]),_number(academic["r2_mean"])]],widths=[4*cm,4*cm,4*cm,4*cm],font_size=7))
        out.append(paragraph("Cohorte recortada propia; validación aleatoria anidada, no espacial.",styles["LS_Small"]))
    for key,title in (("m1_targeted","Refinamiento focalizado M1"),("m1_balanced","Balanceo y expertos M1")):
        result=evidence.get(key)
        if not result:
            continue
        out.append(paragraph(title+" — investigación de desarrollo",styles["LS_H2"]))
        out.append(paragraph(f"{result['configurations']} configuraciones; {result['attempted_evaluations']} evaluaciones y {result['calibration_fits']} ajustes de calibración y {result.get('extra_regime_fits',0)} ajustes adicionales por régimen. No se guardaron modelos nuevos.",styles["LS_Body"]))
        out.append(table([["Familia","Política","RMSE global","MAE global","R² global"]]+[[r["algorithm"],r["policy"],_number(r.get("global_rmse")),_number(r.get("global_mae")),_number(r.get("global_r2"))] for r in result["summary"]],widths=[4.3*cm,4.3*cm,2.5*cm,2.5*cm,2.4*cm],font_size=7))
    research=evidence.get("research_sources",[])
    if research:
        out.append(paragraph("Fuentes de investigación — no incorporadas al entrenamiento",styles["LS_H2"]))
        out.append(table([["Fuente","Filas candidatas","QC sin alerta de identidad","Li >696 ppm","Aprobadas"]]+[
            [row["source"],str(row["summary"].get("candidate_rows")),str(row["summary"].get("qc_no_identity_flag_rows")),
             str(row["summary"].get("above_696_qc_no_identity_flag")),str(row["summary"].get("approved_rows"))] for row in research],
             widths=[5*cm,2.5*cm,3.5*cm,2.5*cm,2.5*cm],font_size=7))
        out.append(paragraph("El cribado descriptivo no prueba independencia ni autoriza incorporación de muestras.",styles["LS_Small"]))
    return out
