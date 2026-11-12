"""Three durable, retryable reports from the full M2 competition's saved OOF.

Never trains, re-labels, publishes, modifies a checkpoint or loads candidates.
Each report has an explicit success/error status; recover after any interruption
with the CLI without rerunning a potentially hour-long competition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
)
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.methodology.ablations import bootstrap_group_delta
from lithiumscope.model_2.methodology.three_zone_ranking import ranking_evaluation
from lithiumscope.model_2.training.competition_ranking import save_three_zone_competition

OLD_OOF=("results/model_2/runs/training_20261112_112450_m0300/"
         "all_reference_paired_oof.csv")
OLD_META=("models/model_2/trained/training_20260920_195933_m0300/"
          "metadata.json")
SCL=("results/scientific_simulations/sentinel_methodology_20261112/"
     "local_scl/per_sample_scl_quality_783.csv")
OOD=("results/scientific_simulations/m2_phase_a_20261111/"
     "training_ood_by_sample.csv")
PHYSICAL=("results/scientific_simulations/m2_phase_d_20261112/"
          "physical_label_audit/scientific_quality_summary.json")
STAC=("results/scientific_simulations/sentinel_methodology_20261112/"
      "inventory/inventory_summary.json")
PREVIOUS_ARM="original_release_protocol_retrained"
EXPECTED_OLD_REFERENCE_SHA256="e71c72196cf56c345ab70b864c66d24b7a5e4b7d82bc2b8f3385366848dcb4dd"
EXPECTED_OLD_METADATA_SHA256="0ada24faf0b33cbc8ebbf721671f6e60bcf476db801d0184ca2d758b045530fe"



def get_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),
                         encoding="utf-8")
    temporary.replace(path)


def metrics(labels,probabilities):
    y=np.asarray(labels,int)
    score=np.asarray(probabilities,float)
    if len(y)==0 or len(y)!=len(score) or not np.isfinite(score).all():
        raise ValueError("Incomplete or nonfinite model predictions")
    if not np.isin(y,[0,1]).all():
        raise ValueError("Invalid binary labels")
    predicted=(score>=0.5).astype(int)
    return {
        "n":int(len(y)),"positives":int(y.sum()),
        "roc_auc":float(roc_auc_score(y,score)) if len(set(y))==2 else None,
        "average_precision":float(average_precision_score(y,score)) if y.sum() else None,
        "brier":float(brier_score_loss(y,score)),
        "balanced_accuracy_at_0p5":float(balanced_accuracy_score(y,predicted))
            if len(set(y))==2 else None,
        "precision_at_0p5":float(precision_score(y,predicted,zero_division=0)),
        "recall_at_0p5":float(recall_score(y,predicted,zero_division=0)),
        "f1_at_0p5":float(f1_score(y,predicted,zero_division=0)),
    }


def load_predictions(manifest,tables):
    if len(manifest)!=783:
        raise ValueError("Expected 783 original Model 2 records")
    scores={}
    common_fold=None
    common_y=None
    for path in sorted(Path(tables).glob("oof_predictions_*.csv")):
        name=path.stem.removeprefix("oof_predictions_")
        df=pd.read_csv(path)
        required={"y_true","li_ppm","outer_fold","prospectivity_score"}
        if not required.issubset(df) or len(df)!=783:
            raise ValueError("Incomplete OOF for "+name)
        if not np.allclose(df.li_ppm,manifest.Li_icpms):
            raise ValueError("OOF geochemistry differs for "+name)
        folds=df.outer_fold.to_numpy(int)
        y=df.y_true.to_numpy(int)
        probabilities=df.prospectivity_score.to_numpy(float)
        if not (np.isin(folds,[1,2,3,4,5]).all() and
                np.isin(y,[0,1]).all() and np.isfinite(probabilities).all()):
            raise ValueError("Invalid OOF for "+name)
        if common_y is not None and (
            not np.array_equal(y,common_y) or
            not np.array_equal(folds,common_fold)):
            raise ValueError("Algorithms disagree on held-out fold labels or indices")
        common_y=y
        common_fold=folds
        scores[name]=probabilities
    if not scores:
        raise ValueError("No complete OOF predictions saved by competition")
    return scores,common_fold,common_y


def fixed_old_release_comparison(root,manifest,folds,scores,output):
    if file_sha256(root/OLD_OOF)!=EXPECTED_OLD_REFERENCE_SHA256:
        raise ValueError("Saved prior-release-protocol prediction source has changed")
    if file_sha256(root/OLD_META)!=EXPECTED_OLD_METADATA_SHA256:
        raise ValueError("Published release metadata has changed")
    meta=get_json(root/OLD_META)
    if meta.get("algorithm")!="catboost" or meta.get("samples")!=706:
        raise ValueError("September release reference has changed")
    cutoff=float(meta["threshold_ppm"])
    old=pd.read_csv(root/OLD_OOF)
    old=old[old.arm==PREVIOUS_ARM].sort_values("test_index")
    if len(old)!=706 or old.test_index.tolist()!=list(range(706)):
        raise ValueError("Old retrained release protocol lacks 706 heldout predictions")
    if not np.array_equal(
        old.sample_id.astype(str),manifest.sample_id.iloc[:706].astype(str)):
        raise ValueError("Old release protocol sample IDs differ from new competition")
    if not np.allclose(old.li_ppm,manifest.Li_icpms.iloc[:706]):
        raise ValueError("Original 706 Li values differ from reference")
    if not np.array_equal(old.fold.to_numpy(int),folds[:706]):
        raise ValueError("Original 706 spatial heldout folds no longer match")
    label=(manifest.Li_icpms.iloc[:706].to_numpy(float)>=cutoff).astype(int)
    common={
        "test_index":np.arange(706),"fold":folds[:706],
        "sample_id":manifest.sample_id.iloc[:706].astype(str).to_numpy(),
        "site_screening_id":(
            manifest.latitude.iloc[:706].round(5).astype(str)+"|"+
            manifest.longitude.iloc[:706].round(5).astype(str)
        ).to_numpy(),
        "spatial_group":manifest.spatial_group.iloc[:706].astype(str).to_numpy(),
        "li_ppm":manifest.Li_icpms.iloc[:706].to_numpy(float),
        "y_true":label,
    }
    rows=[{"algorithm":PREVIOUS_ARM,"method":"release_protocol_retrained_old_706",
           **metrics(label,old.score)}]
    paired=[pd.DataFrame({**common,"arm":PREVIOUS_ARM,"score":old.score.to_numpy(float)})]
    for name,value in sorted(scores.items()):
        rows.append({"algorithm":name,"method":"new_competition_783",
                     **metrics(label,value[:706])})
        paired.append(pd.DataFrame({**common,"arm":name,"score":value[:706]}))
    pd.DataFrame(rows).to_csv(output/"old_release_protocol_comparison_706.csv",index=False)
    common_preds=pd.concat(paired,ignore_index=True)
    common_preds.to_csv(output/"old_release_protocol_paired_706.csv",index=False)
    interval=[]
    for name in sorted(scores):
        sample=common_preds[common_preds.arm.isin([PREVIOUS_ARM,name])]
        interval.append({"algorithm":name,
                         **bootstrap_group_delta(
                             sample,PREVIOUS_ARM,name,repeats=320,seed=20261112)})
    pd.DataFrame(interval).to_csv(output/"old_release_protocol_intervals_706.csv",
                                  index=False)
    triples,cases=ranking_evaluation(common_preds,triples_per_fold=250,seed=42)
    save_json(output/"old_release_protocol_three_zone.json",triples)
    cases.to_csv(output/"old_release_protocol_three_zone_cases.csv",index=False)
    result={
        "state":"complete","reference":PREVIOUS_ARM,
        "prior_fitted_release_artifact_used_as_test_predictor":False,
        "prior_release_protocol_retrained":True,
        "fixed_old_release_Li_cutoff_ppm":cutoff,
        "historical_heldout":706,"same_test_cases":True,"same_folds":True,
        "old_training_706_historical":True,
        "new_training_includes_77_provisional_georoc":True,
        "train_sets_and_fold_thresholds_are_not_identical":True,
        "exploratory_reused_development_data":True,
        "not_independent_external_validation":True,
        "algorithms":sorted(scores),
        "published_release_metrics_not_paired_or_directly_comparable":meta.get("metrics",{}),
        "rank_three_zones":{k:v["top1_accuracy"] for k,v in triples["arms"].items()},
    }
    save_json(output/"old_release_protocol_comparison_summary.json",result)
    display=pd.DataFrame(rows)
    pretty=["# Comparacion con protocolo del release M2 anterior","",
            "**Referencia reconstruida**, no evaluacion del mismo artefacto publicado.",
            "Los 706 casos y folds historicos son identicos, con corte Li fijo de "+
            str(cutoff)+" ppm. Los entrenamientos NO usan igual numero de muestras:",
            "el antiguo usa 706 historicas y el nuevo incluye 77 GEOROC provisionales.",
            "Los datos se reutilizaron para desarrollo; NO son test externo.","",
            "| Algoritmo | AUC | AP | Brier |",
            "|---|---:|---:|---:|"]
    for row in display.itertuples(index=False):
        pretty.append(f"| {row.algorithm} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.brier:.4f} |")
    pretty+=["","Consultar old_release_protocol_intervals_706.csv y old_release_protocol_three_zone.json.",
             "Las metricas publicadas en septiembre estan en la metadata, pero no son comparables directamente."]
    (output/"OLD_RELEASE_PROTOCOL_COMPARISON.md").write_text(
        "\n".join(pretty)+"\n",encoding="utf-8")
    return result


def science_report(root,manifest,folds,y,scores,output):
    scl=pd.read_csv(root/SCL)
    ood=pd.read_csv(root/OOD)
    physical=get_json(root/PHYSICAL)
    stac=get_json(root/STAC)
    if len(scl)!=783 or len(ood)!=783:
        raise ValueError("Scientific audit inputs no longer span 783 records")
    if not np.array_equal(manifest.sample_id.astype(str),scl.sample_id.astype(str)):
        raise ValueError("SCL sample ID order differs")
    if not np.array_equal(manifest.sample_id.astype(str),ood.sample_id.astype(str)):
        raise ValueError("OOD sample ID order differs")
    if physical["historical_conflicting_sites"]!=25 or stac["n_unique_scenes"]!=111:
        raise ValueError("Scientific audit identity/status differs")
    sites=(manifest.latitude.round(5).astype(str)+"|"+
           manifest.longitude.round(5).astype(str))
    lab=manifest.Li_icpms.astype(float)>=21.6
    mixed=set(
        pd.DataFrame({"site":sites,"label":lab}).groupby("site")
        .label.nunique().loc[lambda x:x>1].index
    )
    if len(mixed)!=25:
        raise ValueError("Site label conflicts changed")
    qa=pd.DataFrame({
        "sample_id":manifest.sample_id.astype(str),"test_fold":folds,
        "Li_icpms":manifest.Li_icpms.astype(float),
        "fold_specific_high_Li_label":y,
        "source_cohort":np.where(np.arange(783)<706,"historical","GEOROC_provisional"),
        "site":sites,
        "mixed_label_site_at_Li21p6":sites.isin(mixed),
        "image_sha256":scl.image_sha256,
        "six_band_valid_fraction":scl.sixband_joint_valid_fraction,
        "train_quality_pass_95":(scl.sixband_joint_valid_fraction>=.95),
        "SCL_proxy_accepted":scl.scientifically_accepted_proxy.fillna(False).astype(bool),
        "SCL_rejection_reasons":scl.rejection_reasons.fillna(""),
        "SCL_nonvegetated_fraction":scl.scl_class5_not_vegetated_fraction,
        "SCL_blocked_fraction":scl.local_scl_blocked_fraction,
        "OOD_outside_historical_percentile_range":ood.ood_from_706,
        "OOD_above_25pct":ood.ood_from_706>.25,
        "latitude":manifest.latitude,
        "field_geology_verified":False,
    })
    qa.to_csv(output/"scientific_sample_audit_783.csv",index=False)
    masks={
        "all":np.ones(783,bool),
        "historical":qa.source_cohort.eq("historical").to_numpy(),
        "GEOROC_provisional":qa.source_cohort.eq("GEOROC_provisional").to_numpy(),
        "train_qa95_pass":qa.train_quality_pass_95.to_numpy(bool),
        "train_qa95_fail_but_test_retained":~qa.train_quality_pass_95.to_numpy(bool),
        "SCL_accepted_proxy":qa.SCL_proxy_accepted.to_numpy(bool),
        "SCL_abstention":~qa.SCL_proxy_accepted.to_numpy(bool),
        "OOD_over_25pct":qa.OOD_above_25pct.to_numpy(bool),
        "mixed_label_sites":qa.mixed_label_site_at_Li21p6.to_numpy(bool),
        "lat_10S_20S":qa.latitude.between(-20,-10,inclusive="neither").to_numpy(),
        "lat_20S_30S":qa.latitude.between(-30,-20,inclusive="neither").to_numpy(),
    }
    rows=[]
    for name,prob in sorted(scores.items()):
        for subset,mask in masks.items():
            if mask.any():
                rows.append({"algorithm":name,"subset":subset,
                    "validation":"exploratory_OOF_not_field_verified",
                    **metrics(y[mask],prob[mask])})
    pd.DataFrame(rows).to_csv(output/"scientific_stratified_metrics.csv",index=False)
    summary={
        "state":"complete","samples":783,
        "historical":706,"GEOROC_provisional":77,
        "spectral_QA_pass":int(qa.train_quality_pass_95.sum()),
        "spectral_QA_fail_but_test_retained":int((~qa.train_quality_pass_95).sum()),
        "SCL_accepted_as_optical_proxy":int(qa.SCL_proxy_accepted.sum()),
        "SCL_abstentions":int((~qa.SCL_proxy_accepted).sum()),
        "historical_conflicting_label_sites":len(mixed),
        "OOD_above_25pct":int(qa.OOD_above_25pct.sum()),
        "STAC_scenes":stac["n_unique_scenes"],
        "no_field_rock_certifications":True,
        "no_training_labels_changed":True,
        "no_test_cases_removed":True,
        "not_independent_external_validation":True,
    }
    save_json(output/"scientific_diagnostics_summary.json",summary)
    lines=[
        "# Diagnóstico científico de la competencia Modelo 2","",
        "Los hallazgos son DESCRIPTIVOS; no son prueba independiente de campo.",
        "",
        "- Registros: 783 (706 históricos y 77 GEOROC provisionales).",
        "- QA 95%: "+str(summary["spectral_QA_pass"])+
            " válidos para ajuste; "+str(summary["spectral_QA_fail_but_test_retained"])+
            " se retienen en evaluación.",
        "- Proxy óptico SCL: "+str(summary["SCL_accepted_as_optical_proxy"])+
            " aceptados; "+str(summary["SCL_abstentions"])+
            " abstenciones, NO equivalen a roca de campo.",
        "- Sitios históricos con etiquetas contradictorias: 25.",
        "- Fuera de dominio >25% frente al histórico: "+str(summary["OOD_above_25pct"])+".",
        "",
        "Revisar scientific_sample_audit_783.csv y scientific_stratified_metrics.csv.",
        "Las conclusiones no justifican excluir nuevas filas ni promover el modelo.",
    ]
    (output/"SCIENTIFIC_DIAGNOSTICS.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    return summary


def three_zone_report(manifest_path,folds,labels,scores,output):
    data=save_three_zone_competition(manifest_path,scores,folds,labels,output)
    cases=pd.read_csv(output/"three_zone_competition_cases.csv")
    if set(data["arms"])!=set(scores) or not len(cases):
        raise ValueError("Missing successful competitor in three-zone evidence")
    if not cases.all_three_distinct_sites.all():
        raise ValueError("Triplets contain non-distinct sites")
    return {"state":"complete","algorithms":sorted(scores),
            "case_rows":len(cases),"summary":data,
            "can_regenerate_without_retraining":True}


def finalize_reports(manifest_path,source_tables,output,*,root=PROJECT_ROOT,strict=False):
    manifest_path=Path(manifest_path)
    source_tables=Path(source_tables)
    output=Path(output)
    root=Path(root)
    output.mkdir(parents=True,exist_ok=True)
    status={"state":"running","manifest":str(manifest_path),
            "source_tables":str(source_tables),
            "output":str(output),
            "runs_no_training":True,"runs_no_activation":True,
            "reports":{}}
    try:
        source=pd.read_csv(manifest_path)
        scores,folds,labels=load_predictions(source,source_tables)
        status["algorithms"]=sorted(scores)
        status["outer_test_rows"]=len(source)
    except Exception as exc:
        status["state"]="blocked_input"
        status["error"]=f"{type(exc).__name__}: {exc}"
        save_json(output/"evidence_report_status.json",status)
        (output/"EVIDENCE_REPORT_STATUS.md").write_text(
            "# Informes Modelo 2: entradas invalidas\n\n"+
            status["error"]+"\n\nNo se han modificado modelos ni checkpoints.\n",
            encoding="utf-8")
        if strict:raise
        return status
    tasks={
        "old_release_comparison":lambda:fixed_old_release_comparison(
            root,source,folds,scores,output),
        "scientific_diagnostics":lambda:science_report(
            root,source,folds,labels,scores,output),
        "three_zone":lambda:three_zone_report(
            manifest_path,folds,labels,scores,output),
    }
    for name,task in tasks.items():
        try:
            status["reports"][name]={"state":"complete","result":task()}
        except Exception as exc:
            status["reports"][name]={
                "state":"requires_recovery",
                "error":f"{type(exc).__name__}: {exc}",
                "traceback":traceback.format_exc(limit=6),
            }
        save_json(output/"evidence_report_status.json",status)
    errors=[k for k,v in status["reports"].items() if v["state"]!="complete"]
    status["state"]="complete" if not errors else "partial_requires_recovery"
    status["errors"]=errors
    cmd=("python -m lithiumscope.model_2.training.postprocess"+
         ' --manifest "'+str(manifest_path)+'"'+
         ' --tables "'+str(source_tables)+'"'+
         ' --output "'+str(output)+'"')
    status["recovery_command"]=cmd
    save_json(output/"evidence_report_status.json",status)
    lines=["# Estado de informes de Modelo 2","",
           "**Estado:** "+status["state"],""]
    for name,item in status["reports"].items():
        lines.append("- "+name+": "+item["state"]+
                     ((" - "+item["error"]) if "error" in item else ""))
    lines+=["","Recuperación sin reentrenar:",cmd,"",
            "La comparación es interna, no justifica release ni certifica litio."]
    (output/"EVIDENCE_REPORT_STATUS.md").write_text("\n".join(lines)+"\n",
                                                    encoding="utf-8")
    if strict and errors:
        raise RuntimeError("Incomplete reporting; run: "+cmd)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--tables",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--strict",action="store_true")
    args=p.parse_args()
    resolve=lambda s:Path(s) if Path(s).is_absolute() else PROJECT_ROOT/s
    status=finalize_reports(resolve(args.manifest),resolve(args.tables),
                            resolve(args.output),strict=args.strict)
    print(json.dumps({"state":status["state"],
                      "reports":{k:v["state"] for k,v in status.get("reports",{}).items()},
                      "recovery":status.get("recovery_command")},ensure_ascii=False))


if __name__=="__main__":
    main()
