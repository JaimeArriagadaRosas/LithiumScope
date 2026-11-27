"""Read-only registered simulation suite for installed releases and local candidates."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.experiments.shared.comparison_models import discover_models
from lithiumscope.experiments.shared.saved_comparison import compare_saved
from lithiumscope.experiments.scenarios.scenarios import evaluate_model_1
from lithiumscope.experiments.scenarios.batch import run_m1_batch,run_m2_batch,summarize
from lithiumscope.experiments.reporting.reporting import compare_geographies
from lithiumscope.results.atomic_exports import write_json, write_table


def run_registered_simulations(output_root: Path | None = None, *,
                               groups=("model_1", "model_2"), include_saved_oof=True) -> Path:
    groups = tuple(dict.fromkeys(groups))
    if not groups or not set(groups) <= {"model_1", "model_2"}:
        raise ValueError("Invalid model scope")
    root=output_root if output_root is not None else RESULTS_DIR/"scientific_simulations"
    destination=root/f"sim_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid4().hex[:8]}"
    destination.mkdir(parents=True,exist_ok=False)
    protocols,identities=[],[]
    def record(name,action):
        try:
            evidence=action()
            result={"protocol":name,"state":"COMPLETADO",**evidence}
        except FileNotFoundError as exc:
            result={"protocol":name,"state":"BLOQUEADO","reason":str(exc)}
        except Exception as exc:
            result={"protocol":name,"state":"ERROR","reason":str(exc)}
        protocols.append(result)
        print(f"{name}: {result['state']} {result.get('reason','')}",flush=True)
        return result
    models={group:discover_models(group) for group in groups}
    for group,(available,issues) in models.items():
        for issue in issues:
            protocols.append({"protocol":f"{group}: inventario {issue['role']}",**issue})
            print(f"{group} {issue['role']}: {issue['state']} {issue['reason']}")
        for role,(_,identity) in available.items():
            identities.append({"role":role,**identity})
            print(f"{group} {role}: {identity['run_id']} | {identity['algorithm']}")
        if include_saved_oof:
            record(group+": OOF histórico/candidato",lambda g=group,m=available:
                   {"artifact":str(compare_saved(g,m,destination/g/"paired_oof"))})
    cases_path=PROJECT_ROOT/"examples"/"integrated_demo"/"cases.csv"
    m1_outputs={}
    for role,(bundle,identity) in models.get("model_1", ({}, []))[0].items():
        folder=destination/"model_1"/role
        def m1_single():
            result=evaluate_model_1(bundle,pd.read_csv(cases_path),folder/"single")
            return {"artifact":str(result),"scenario_rows":len(pd.read_csv(result))}
        def m1_batch():
            result=run_m1_batch(bundle,pd.read_csv(cases_path),folder/"batch")
            m1_outputs[role]=result
            return {"artifact":str(result),"summary":str(summarize(result,result.parent))}
        record("M1 "+role+": 18 escenarios",m1_single)
        record("M1 "+role+": escenarios masivos cinco semillas",m1_batch)
    if {"release","candidate"}<=set(m1_outputs):
        def pair_m1():
            a,b=(pd.read_csv(m1_outputs[k]) for k in ("release","candidate"))
            paired=a.merge(b,on=["seed","scenario","family","intensity"],suffixes=("_release","_candidate"),validate="one_to_one")
            for name in ("rmse_ppm","mae_ppm","prediction_shift_ppm","ood_rate"):
                paired[name+"_delta"]=paired[name+"_candidate"]-paired[name+"_release"]
            return {"artifact":str(write_table(destination/"model_1"/"paired_stress.csv",paired)),"paired_rows":len(paired)}
        record("M1: comparación de robustez pareada",pair_m1)
    available=models.get("model_2", ({}, []))[0]
    matrix={}
    def common():
        from lithiumscope.experiments.model_2.evaluation.descriptor_stress import load_common_matrix
        matrix["frame"],matrix["ids"]=load_common_matrix()
        return {"rows":len(matrix["frame"]),"scope":"historical_cohort_may_overlap_training"}
    if available:
        record("M2: verificación de matriz espectral común",common)
    descriptor_outputs={}
    raw_outputs={}
    pilot=RESULTS_DIR/"scientific_simulations"/"sentinel2_real"/"manifest.csv"
    # Existing seasonal/shifted catalogs only: no automatic external acquisition.
    catalogs={"pilot":pilot}
    for folder in sorted((RESULTS_DIR/"scientific_simulations").glob("**/extended_manifest.csv")):
        catalogs["extended_"+str(len(catalogs))]=folder
    for role,(bundle,identity) in available.items():
        folder=destination/"model_2"/role
        def descriptor():
            if not matrix:
                raise FileNotFoundError("Matriz espectral local no verificada")
            from lithiumscope.experiments.model_2.evaluation.descriptor_stress import descriptor_stress
            output=descriptor_stress(bundle,matrix["frame"],matrix["ids"],folder)
            descriptor_outputs[role]=output
            return {"artifact":str(output),"summary":str(summarize(output,folder/"descriptor_summary"))}
        record("M2 "+role+": perturbaciones de descriptores congelados",descriptor)
        for name,source in catalogs.items():
            def raw():
                if not source.is_file():
                    raise FileNotFoundError("Catálogo piloto ausente; adquisición externa no automática")
                output=run_m2_batch(bundle,source,folder/name)
                table=pd.read_csv(output)
                if table.empty:
                    raise FileNotFoundError("No hay parches aceptados")
                raw_outputs[(role,name)]=output
                report=summarize(output,output.parent)
                scored=int(table.status.eq("SCORED").sum())
                return {"state":"COMPLETADO" if table.loc[table.scenario.eq("baseline"),"status"].eq("SCORED").all() and scored else "PARCIAL" if scored else "BLOQUEADO",
                        "artifact":str(output),"summary":str(report),"scored":scored,"abstentions":len(table)-scored,
                        "geography":str(compare_geographies(output,output.parent)[1])}
            record(f"M2 {role}: espectral/geográfico/temporal {name}",raw)
    if {"release","candidate"}<=set(descriptor_outputs):
        def pair_m2():
            a,b=(pd.read_csv(descriptor_outputs[k]) for k in ("release","candidate"))
            paired=a.merge(b,on=["sample_id","seed","scenario"],suffixes=("_release","_candidate"),validate="one_to_one")
            paired["score_delta"]=paired.score_candidate-paired.score_release
            paired["shift_delta"]=paired.absolute_score_shift_candidate-paired.absolute_score_shift_release
            summary=paired.groupby("scenario")[["score_delta","shift_delta"]].mean().reset_index()
            write_table(destination/"model_2"/"paired_descriptor_rows.csv",paired)
            return {"artifact":str(write_table(destination/"model_2"/"paired_descriptor_summary.csv",summary)),
                    "paired_rows":len(paired),"scope":"not independent accuracy"}
        record("M2: comparación pareada de descriptores",pair_m2)
    for name in catalogs:
        if all((role,name) in raw_outputs for role in ("release","candidate")):
            def pair_raw():
                a,b=(pd.read_csv(raw_outputs[(role,name)]) for role in ("release","candidate"))
                paired=a.merge(b,on=["image_id","scene_id","year","scenario"],suffixes=("_release","_candidate"),validate="one_to_one")
                paired["both_scored"]=paired.status_release.eq("SCORED")&paired.status_candidate.eq("SCORED")
                paired["score_delta"]=paired.score_candidate-paired.score_release
                return {"artifact":str(write_table(destination/"model_2"/f"paired_{name}.csv",paired)),
                        "paired_rows":len(paired),"both_scored":int(paired.both_scored.sum())}
            record("M2: comparación pareada de imágenes "+name,pair_raw)
    states={p["state"] for p in protocols}
    state="COMPLETADO" if states=={"COMPLETADO"} else "PARCIAL" if states&{"COMPLETADO","PARCIAL"} else "ERROR" if "ERROR" in states else "BLOQUEADO"
    return write_json(destination/"manifest.json",{"run_id":destination.name,"state":state,
        "scope":"read_only_robustness_and_saved_oof_not_independent_validation",
        "models":identities,"protocols":protocols,"groups":list(groups),"training":False,"promotion":False,
        "external_acquisition":False,
        "historical_scripts":"Not automatically replayed: may acquire, fit or depend on old frozen protocols."})
