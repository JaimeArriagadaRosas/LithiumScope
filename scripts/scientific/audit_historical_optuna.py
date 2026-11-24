"""Read archived Optuna SQLite bytes without opening historical studies for writing."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd
import yaml

from lithiumscope.core.hashing import file_sha256


def distribution_evidence(value, spec):
    name, attr = spec["name"], spec["attributes"]
    if name == "CategoricalDistribution":
        if value != int(value) or not 0 <= int(value) < len(attr["choices"]):
            raise ValueError("Invalid categorical index")
        return attr["choices"][int(value)], None, None, None
    if name not in {"IntDistribution", "FloatDistribution"}:
        raise ValueError(f"Unsupported archived distribution {name}")
    lo, hi = attr["low"], attr["high"]
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError("Archived parameter outside declared bounds")
    if hi == lo:
        return value, lo, hi, "fixed"
    boundary = "low" if math.isclose(value, lo, rel_tol=1e-12, abs_tol=0.) else (
        "high" if math.isclose(value, hi, rel_tol=1e-12, abs_tol=0.) else "interior")
    return value, lo, hi, boundary


def audit_db(path, root):
    relative = path.relative_to(root)
    group, _, run, _, algorithm, filename = relative.parts
    key = relative.as_posix()
    if path.with_name(path.name + "-wal").exists() or path.with_name(path.name + "-journal").exists():
        raise ValueError(f"Active SQLite sidecar; archival status not certified: {key}")
    before = file_sha256(path)
    run_root = root/group/"runs"/run
    cfg_path = run_root/"config"/f"{group}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    opt = cfg["optimization"]
    family = algorithm.split("__")[0]
    budget = opt.get("per_algorithm_trials", {}).get(family, opt["n_trials"])
    inner = cfg["validation"]["inner_folds"]
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        if db.execute("pragma quick_check").fetchone()[0] != "ok":
            raise ValueError(f"SQLite integrity error: {key}")
        studies = db.execute("select study_id,study_name from studies").fetchall()
        if len(studies) != 1:
            raise ValueError(f"Expected one archived study: {key}")
        study_id, name = studies[0]
        direction = db.execute("select direction from study_directions where study_id=? and objective=0",(study_id,)).fetchone()[0]
        trials = db.execute("""select t.trial_id,t.number,t.state,v.value,v.value_type,
                    t.datetime_start,t.datetime_complete from trials t
                    left join trial_values v on v.trial_id=t.trial_id and v.objective=0
                    where t.study_id=? order by t.number""",(study_id,)).fetchall()
        params = {}
        for tid, param, val, dist in db.execute("select trial_id,param_name,param_value,distribution_json from trial_params"):
            params.setdefault(tid, {})[param] = (val,json.loads(dist))
        steps = {}
        for tid, step, val, kind in db.execute("select trial_id,step,intermediate_value,intermediate_value_type from trial_intermediate_values"):
            steps.setdefault(tid, {})[step] = (val,kind)
    if file_sha256(path) != before:
        raise ValueError(f"Archive changed during audit: {key}")
    counts = Counter(t[2] for t in trials)
    completed = [t for t in trials if t[2]=="COMPLETE" and t[4]=="FINITE" and t[3] is not None and math.isfinite(t[3])]
    best = min(completed,key=lambda t:(t[3] if direction=="MINIMIZE" else -t[3],t[1])) if completed else None
    trial_rows, parameter_rows = [], []
    invalid_complete = 0
    for tid, number, state, value, kind, start, finish in trials:
        reports=steps.get(tid,{})
        full = sorted(reports)==list(range(inner)) and all(v is not None and k=="FINITE" and math.isfinite(v) for v,k in reports.values())
        objective_matches = bool(full and value is not None and math.isclose(value,reports[inner-1][0],rel_tol=1e-10,abs_tol=1e-10))
        if state=="COMPLETE" and not objective_matches:
            invalid_complete += 1
        chosen = best is not None and tid==best[0]
        trial_rows.append({"database":key,"trial":number,"state":state,"objective":value,
            "reported_steps":len(reports),"full_expected_folds":full,"objective_matches_last_report":objective_matches,
            "is_best":chosen,"started_at":start,"finished_at":finish})
        for pname,(val,spec) in params.get(tid,{}).items():
            decoded,lo,hi,bound=distribution_evidence(val,spec)
            parameter_rows.append({"database":key,"trial":number,"state":state,"is_best":chosen,
                "parameter":pname,"value":decoded,"low":lo,"high":hi,"boundary":bound,
                "distribution":json.dumps(spec,sort_keys=True)})
    record={"database":key,"sha256":before,"model":group,"run":run,"algorithm":algorithm,"study":name,
        "scope":filename.removeprefix("optuna_").removesuffix(".db"),"direction":direction,
        "planned_attempts":budget,"attempted":len(trials),"completed":counts["COMPLETE"],
        "pruned":counts["PRUNED"],"failed":counts["FAIL"],"running":counts["RUNNING"],"waiting":counts["WAITING"],
        "expected_inner_folds":inner,"reported_inner_steps":sum(len(v) for v in steps.values()),
        "complete_without_full_objective_evidence":invalid_complete,
        "best_value":best[3] if best else None,"best_trial":best[1] if best else None,
        "best_numeric_boundaries":sum(r["is_best"] and r["boundary"] in {"low","high"} for r in parameter_rows),
        "best_numeric_parameters":sum(r["is_best"] and r["boundary"] in {"low","high","interior"} for r in parameter_rows),
        "archived_config_sha256":file_sha256(cfg_path)}
    return record,trial_rows,parameter_rows


def run(root,output):
    root,output=root.resolve(),output.resolve()
    if output.is_relative_to(root) or not root.is_dir():
        raise ValueError("Output must be outside historical results")
    paths=sorted(root.glob("model_*/runs/*/checkpoints/**/*.db"))
    if not paths:
        raise ValueError("No archived Optuna databases")
    records,trials,params=[],[],[]
    for path in paths:
        r,t,p=audit_db(path,root)
        records.append(r);trials.extend(t);params.extend(p)
    table=pd.DataFrame(records)
    trial_table=pd.DataFrame(trials)
    parameter_table=pd.DataFrame(params)
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/"studies.csv",index=False)
    trial_table.to_csv(output/"trials.csv",index=False)
    parameter_table.to_csv(output/"parameters.csv",index=False)
    fields=["planned_attempts","attempted","completed","pruned","failed","running","waiting",
            "reported_inner_steps","complete_without_full_objective_evidence","best_numeric_boundaries","best_numeric_parameters"]
    grouped=table.groupby(["model","run","algorithm"],as_index=False).agg({"database":"count",**{k:"sum" for k in fields}})
    grouped.rename(columns={"database":"studies"}).to_csv(output/"algorithm_runs.csv",index=False)
    summary={"databases":len(table),"runs":table.run.nunique(),"by_model":table.groupby("model")[fields].sum().to_dict("index"),
        "studies_without_completed":int((table.completed==0).sum()),
        "studies_attempt_budget_mismatch":int((table.attempted!=table.planned_attempts).sum()),
        "studies_with_best_at_numeric_bound":int((table.best_numeric_boundaries>0).sum()),
        "source_bytes_unchanged":all(file_sha256(root/r["database"])==r["sha256"] for r in records),
        "training_performed":False,"sufficiency_certified":False,
        "limitation":"Historical configured budget is not proof of sufficiency. Reported steps are not an exact count of model fits; failed/aborted fits may have no report."}
    if not summary["source_bytes_unchanged"]:
        raise ValueError("Archive changed during audit")
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-results",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    run(args.historical_results,args.output)
