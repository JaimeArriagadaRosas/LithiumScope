"""Audit already acquired M1 candidates against the frozen cohort, without fitting."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.datasets.management.research_sources import M1_RESEARCH_SOURCES
from lithiumscope.experiments.data_lab.candidate_screening import screen, num, CHEM
from lithiumscope.results.eda_artifacts import verify_eda
from lithiumscope.results.atomic_exports import write_json, write_table

REFERENCE="training_20261106_111653_m0300"
FOLD_SHA256="b4d2af7501933169e69ecc99073266a6479fa3fdfb4efc7e978113460f5fe977"


def run_audit():
    source=local_training_source("model_1")
    base_sha=file_sha256(source)
    folds_path=RESULTS_DIR/"model_1"/"runs"/REFERENCE/"tables"/"outer_fold_manifest.csv"
    if file_sha256(folds_path)!=FOLD_SHA256:
        raise ValueError("Historical spatial fold manifest differs")
    groups=pd.read_csv(folds_path).spatial_group
    if len(groups)!=3420 or groups.nunique()!=244:
        raise ValueError("Frozen reference cohort differs")
    frames,inputs=[],[]
    for spec in M1_RESEARCH_SOURCES:
        folders=RESULTS_DIR/"model_1"/"candidate_eda"/spec.key
        for run in sorted(folders.glob("eda_*"),reverse=True):
            try:
                manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
                if manifest.get("base_sha256")!=base_sha or not verify_eda(run,manifest["signature"]):
                    continue
                path=run/"tables"/"candidate_view.csv"
                digest=file_sha256(path)
                frame=pd.read_csv(path,low_memory=False)
                if not frame.source_dataset.eq(spec.key).all():
                    raise ValueError("Candidate source identity mismatch")
                frames.append(frame)
                inputs.append({"source":spec.key,"run_id":run.name,"candidate_sha256":digest,
                    "manifest_sha256":file_sha256(run/"manifest.json")})
                break
            except (OSError,ValueError,TypeError,KeyError):
                continue
        else:
            raise FileNotFoundError("No intact candidate EDA for "+spec.key+"; do not download again")
    candidates=pd.concat(frames,ignore_index=True)
    base=pd.read_csv(source,low_memory=False)
    result=screen(candidates,base,groups)
    summary=[]
    for (key,state),part in result.groupby(["source_dataset","screening_state"]):
        summary.append({"source":key,"state":state,"rows":len(part),
            "spatial_groups":part.spatial_group.nunique(),
            "new_groups":part.loc[~part.existing_spatial_group,"spatial_group"].nunique(),
            "publications":part.source_publication.nunique(),
            "above_696":int(part.Li_icpms.gt(696).sum()),
            "above_1000":int(part.Li_icpms.gt(1000).sum()),
            "metadata_review":int(part.metadata_review_required.sum()),"approved_rows":0})
    output=RESULTS_DIR/"model_1"/"candidate_audit"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    write_table(output/"row_screening.csv",result)
    write_table(output/"summary.csv",pd.DataFrame(summary))
    profiles=[]
    for scope,frame in [("base_raw",base),*[(name,candidates.loc[result.screening_state.eq(name)]) for name in
                          ("convergent_review","high_transfer_review","other_domain_review")]]:
        for col,series in num(frame,CHEM).items():
            profiles.append({"scope":scope,"feature":col,"observed_fraction":float(series.notna().mean()) if len(series) else None,
                             "q05":series.quantile(.05),"median":series.median(),"q95":series.quantile(.95)})
    write_table(output/"domain_shift.csv",pd.DataFrame(profiles))
    write_json(output/"manifest.json",{"state":"review_required","scope":"candidate_screening_not_training",
        "base_sha256":base_sha,"frozen_fold_sha256":FOLD_SHA256,"inputs":inputs,
        "duplicate_rules":{"coordinates_degrees":.001,"li_relative_tolerance":.01,
            "li_absolute_floor":.05,"minimum_shared_geochemistry":5,"feature_relative_tolerance":.02},
        "summary":summary,"approved_rows":0,"training":False,"acquisition":False,
        "outputs":{name:file_sha256(output/name) for name in ("row_screening.csv","summary.csv","domain_shift.csv")}})
    print(pd.DataFrame(summary).to_string(index=False))
    print("Revisión de procedencia y método pendiente; ninguna fila incorporada.")
    print(f"Evidencias: {output}")
    return output


if __name__=="__main__":
    run_audit()
