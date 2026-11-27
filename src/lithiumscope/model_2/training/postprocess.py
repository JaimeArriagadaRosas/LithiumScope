"""Orchestrate recoverable M2 reports without fitting or modifying candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
import pandas as pd
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.logging.session import get_logger
from lithiumscope.model_2.evaluation.saved_oof import load_predictions, save_json
from lithiumscope.model_2.diagnostics.prior_release_evidence import fixed_old_release_comparison
from lithiumscope.model_2.diagnostics.science_evidence import science_report
from lithiumscope.model_2.diagnostics.competition_audit import analyse_saved_competition as diagnostic_report
from lithiumscope.model_2.evaluation.ranking import save_three_zone_competition

logger = get_logger("model_2.postprocess")


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
    logger.info("M2 evidence reports started: %s", output)
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
        logger.exception("M2 saved OOF validation failed")
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
            logger.info("M2 evidence report completed: %s",name)
        except Exception as exc:
            logger.exception("M2 evidence report requires recovery: %s",name)
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
    logger.info("M2 evidence reports finished: %s",status["state"])
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
