"""Build an identity-bound, JSON-only scientific annex for any selected candidate pair."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.results.eda_artifacts import verify_eda


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path,columns):
    frame=pd.read_csv(path)
    return json.loads(frame[[c for c in columns if c in frame]].to_json(orient="records"))


def _selected(group,identity,root):
    run_id=identity.get("run_id","")
    result={"run_id":run_id,"algorithm":identity.get("algorithm"),
            "dataset_sha256":identity.get("dataset_sha256"),"model_sha256":identity.get("model_sha256"),
            "metrics":identity.get("training_metrics",{}),"status":identity.get("candidate_status"),
            "state":"metadata_only"}
    if not run_id or Path(run_id).name!=run_id or any(c in run_id for c in ("/",chr(92),":","..")):
        return result
    directory=root/"results"/group/"runs"/run_id
    try:
        run=read(directory/"run.json")
        if run["state"]!="completed" or not result["dataset_sha256"] or run["summary"]["dataset_sha256"]!=result["dataset_sha256"]:
            raise ValueError("Training source identity mismatch")
        ranking=directory/"tables"/"competition_ranking.csv"
        result["ranking"]=_rows(ranking,("algorithm","label","status","rmse_mean","mae_mean","r2_mean",
                "roc_auc_mean","average_precision_mean","brier_score_mean","balanced_accuracy_mean"))
        result["ranking_sha256"]=file_sha256(ranking)
        result["state"]="training_tables_available"
    except (OSError,ValueError,KeyError,TypeError,pd.errors.ParserError) as exc:
        result["training_tables_reason"]=str(exc)
        return result
    try:
        link=read(directory/"manifests"/"eda_evidence.json")
        name=link["eda_run_id"]
        if Path(name).name!=name:
            raise ValueError("Invalid EDA link")
        evidence=root/"results"/group/"eda"/name
        manifest=read(evidence/"manifest.json")
        if (file_sha256(evidence/"manifest.json")!=link["manifest_sha256"]
                or manifest["source_sha256"]!=result["dataset_sha256"]
                or manifest["development_protocol"]!=link["development_protocol"]
                or not verify_eda(evidence,link["signature"])):
            raise ValueError("Training EDA link not verified")
        result["eda"]={"run_id":name,"rows_raw":manifest["rows_raw"],"rows_prepared":manifest["rows_prepared"],
                       "features":manifest["features"],"manifest_sha256":link["manifest_sha256"],
                       "source_sha256":manifest["source_sha256"],"fold_protocol":manifest["development_protocol"],
                       "cohort_summary":_rows(evidence/"tables"/"cohort_summary.csv",("source","stage","rows","target_min","target_median","target_max"))}
    except (OSError,ValueError,KeyError,TypeError,pd.errors.ParserError) as exc:
        result["eda_reason"]=str(exc)
    for filename in ("winner_treatment.json","winner_spectral_treatment.json"):
        path=directory/"tables"/filename
        if path.is_file():
            try:
                result["winner_treatment"]={"file":filename,"sha256":file_sha256(path),"data":read(path)}
            except (ValueError,OSError):
                pass
    return result


def _paired(group,selected,root):
    folder=root/"results"/"scientific_simulations"
    candidates=list(folder.glob("sim_*/"+group+"/paired_oof/comparison.json"))
    candidates+=list(folder.glob("version_comparisons/*/"+group+"/comparison.json"))
    for path in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
        try:
            data=read(path)
            identity=data["candidate"]
            if (identity["run_id"]==selected["run_id"] and selected.get("model_sha256")
                    and identity["sha256"]==selected["model_sha256"]):
                return {"state":"paired_development_only","sha256":file_sha256(path),"data":data}
        except (OSError,KeyError,ValueError,TypeError):
            continue
    return {"state":"unavailable","reason":"No matching saved OOF comparison for these model bytes"}


def _targeted(selected,root):
    folder=root/"results"/"generalization"/"model_1_targeted"
    for path in sorted(folder.glob("*/manifest.json"),reverse=True):
        try:
            data=read(path)
            if (data["state"]!="complete" or data["reference_run"]!=selected["run_id"]
                    or data["source_sha256"]!=selected["dataset_sha256"]):
                continue
            summary=data["summary"]
            return {"run_id":path.parent.name,"manifest_sha256":file_sha256(path),
                    "configurations":data["configurations"],"attempted_evaluations":data["attempted_evaluations"],
                    "calibration_fits":data["calibration_fits"],
                    "summary":[r for r in summary if r["policy"] in {"control","selected_inner"}],
                    "scope":"exploratory_fixed_archived_folds","models_saved":False}
        except (OSError,ValueError,TypeError,KeyError):
            continue
    return None


def gather_current(m1,m2,*,root):
    selected={group:_selected(group,ident,root) for group,ident in (("model_1",m1),("model_2",m2))}
    return {"state":"CURRENT_SELECTION_EVIDENCE","models":selected,
            "paired":{g:_paired(g,s,root) for g,s in selected.items()},
            "m1_targeted":_targeted(selected["model_1"],root),
            "independent_holdout":False,"user_release_approval_required":True}
