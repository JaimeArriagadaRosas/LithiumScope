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
        if paired["state"]=="paired_development_only":
            data=paired["data"]
            keys=sorted(set(data["release_metrics"]) & set(data["candidate_metrics"]))
            out.append(paragraph("Comparación OOF emparejada exploratoria contra el release.",styles["LS_H2"]))
            out.append(table([["Métrica","Release","Candidato"]]+[[k,_number(data["release_metrics"][k]),_number(data["candidate_metrics"][k])] for k in keys],widths=[8*cm,4*cm,4*cm],font_size=7))
            if data.get("gains"):
                out.append(table([["Ganancia positiva","Valor","IC95 inferior","IC95 superior"]]+[[r["metric"],_number(r["observed_gain"]),_number(r["ci95_group_lower"]),_number(r["ci95_group_upper"])] for r in data["gains"]],widths=[4*cm,4*cm,4*cm,4*cm],font_size=7))
        else:
            out.append(paragraph("Comparación OOF emparejada: no disponible para estos artefactos; no se sustituye por métricas ajenas.",styles["LS_Small"]))
    result=evidence.get("m1_targeted")
    if result:
        out.append(paragraph("Refinamiento focalizado M1 — investigación de desarrollo",styles["LS_H2"]))
        out.append(paragraph(f"{result['configurations']} configuraciones; {result['attempted_evaluations']} evaluaciones y {result['calibration_fits']} ajustes de calibración. No se guardaron modelos nuevos.",styles["LS_Body"]))
        out.append(table([["Familia","Política","RMSE global","MAE global","R² global"]]+[[r["algorithm"],r["policy"],_number(r.get("global_rmse")),_number(r.get("global_mae")),_number(r.get("global_r2"))] for r in result["summary"]],widths=[4.3*cm,4.3*cm,2.5*cm,2.5*cm,2.4*cm],font_size=7))
    return out
