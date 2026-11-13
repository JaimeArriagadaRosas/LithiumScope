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
    if len(manifest) not in (706,783):
        raise ValueError("Expected a frozen 706- or expanded 783-row M2 manifest")
    scores={}
    common_fold=None
    common_y=None
    for path in sorted(Path(tables).glob("oof_predictions_*.csv")):
        name=path.stem.removeprefix("oof_predictions_")
        df=pd.read_csv(path)
        required={"y_true","li_ppm","outer_fold","prospectivity_score"}
        if not required.issubset(df) or len(df)!=len(manifest):
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


def fixed_old_release_comparison(root, manifest, folds, scores, output):
    from lithiumscope.model_2.training.postprocess_prior_release import fixed_old_release_comparison as _implementation
    return _implementation(root, manifest, folds, scores, output)


def science_report(root, manifest, folds, y, scores, output):
    from lithiumscope.model_2.training.postprocess_science import science_report as _implementation
    return _implementation(root, manifest, folds, y, scores, output)


def diagnostic_report(manifest_path, tables, output, *, root=PROJECT_ROOT):
    """Four-way confusion/error audit for all completed competitors."""
    from lithiumscope.model_2.training.diagnostic_audit import (
        analyse_saved_competition,
    )
    return analyse_saved_competition(
        manifest_path, tables, output, root=root,
    )


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
        "competition_diagnostics":lambda:diagnostic_report(
            manifest_path,source_tables,output,root=root),
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
