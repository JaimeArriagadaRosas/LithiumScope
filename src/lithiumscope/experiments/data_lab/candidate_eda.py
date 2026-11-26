"""Candidate-only descriptive audit. Existing training data never gets overwritten."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.adapters.georoc_ingestion import ingest_georoc_files
from lithiumscope.datasets.management.research_sources import M1_RESEARCH_SOURCES
from lithiumscope.results.eda_artifacts import complete_run, find_compatible, new_run, fail_run
from lithiumscope.tools.acquisition.georoc.research import acquire


def _keys(frame):
    """Conservative cross-source identity flags; not proof of independence."""
    keys = pd.DataFrame(index=frame.index)
    for name in ("source_unique_id","source_sample","source_publication"):
        values = frame.get(name,pd.Series("",index=frame.index)).fillna("").astype(str).str.strip().str.casefold()
        keys[name] = values.replace({"nan":"","none":"","<na>":""})
    keys["coordinate_li"] = (
        pd.to_numeric(frame["Longitude"],errors="coerce").round(5).astype(str)+"|"+
        pd.to_numeric(frame["Latitude"],errors="coerce").round(5).astype(str)+"|"+
        pd.to_numeric(frame["Li_icpms"],errors="coerce").round(6).astype(str))
    return keys


def overlap_flags(candidate, base):
    left,right = _keys(candidate),_keys(base)
    result = pd.DataFrame(index=candidate.index)
    for key in ("source_unique_id","source_sample","source_publication","coordinate_li"):
        result[key+"_overlap"] = left[key].ne("") & left[key].isin(set(right[key])- {""})
    result["within_candidate_coordinate_li"] = left.coordinate_li.duplicated(keep=False)
    return result


def _plots(frame, folder):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    y = pd.to_numeric(frame.Li_icpms,errors="coerce")
    fig = Figure(figsize=(8,4))
    FigureCanvasAgg(fig)
    ax=fig.add_subplot(111)
    ax.hist(np.log1p(y[y.ge(0)].dropna()),bins=40)
    ax.set(xlabel="log(1 + Li ppm)",ylabel="Muestras",title="Distribución de Li (candidatas)")
    fig.tight_layout()
    fig.savefig(folder/"lithium_distribution.png",dpi=150)
    fig=Figure(figsize=(7,5))
    FigureCanvasAgg(fig)
    ax=fig.add_subplot(111)
    ax.scatter(frame.Longitude,frame.Latitude,s=5,alpha=.5)
    ax.set(xlabel="Longitud",ylabel="Latitud",title="Cobertura espacial (candidatas)")
    fig.tight_layout()
    fig.savefig(folder/"coordinates.png",dpi=150)


def profile(spec, paths, receipt, base_path, *, output_root=None):
    paths = tuple(map(Path,paths))
    cfg=load_config("model_1")
    digests={p.name:file_sha256(p) for p in paths}
    base_digest=file_sha256(base_path)
    from lithiumscope.datasets.adapters import georoc_ingestion_schema, georoc_ingestion_transform
    from importlib.metadata import version
    key=canonical_json_hash({"source":spec.key,"files":digests,"base":base_digest,"cfg":cfg["data_sources"]["georoc"],
        "code":[file_sha256(Path(__file__)),file_sha256(Path(georoc_ingestion_schema.__file__)),
                file_sha256(Path(georoc_ingestion_transform.__file__))],
        "versions":{p:version(p) for p in ("pandas","numpy","matplotlib")}})
    root=Path(output_root) if output_root else RESULTS_DIR/"model_1"/"candidate_eda"/spec.key
    cached=find_compatible(root,key)
    if cached:
        print(f"[OK] EDA candidato vigente: {cached}")
        return cached
    directory=new_run(root)
    try:
        # Harmonized view is an analysis artifact, not a training source.
        tables=directory/"tables"
        settings=cfg["data_sources"]["georoc"]
        harmonized=ingest_georoc_files(paths,tables/"candidate_view.csv",
            audit_path=tables/"ingestion_audit.json",source_name=spec.key,
            minimum_predictors=int(settings["minimum_predictors"]),
            allowed_material_types=tuple(settings["allowed_material_types"]),
            iron_total_mode=settings["iron_total_mode"])
        frame=pd.read_csv(harmonized.path,low_memory=False)
        base=pd.read_csv(base_path,low_memory=False)
        flags=overlap_flags(frame,base)
        # No row gets approved automatically; flags are screening evidence.
        frame[["source_sample","source_file","Longitude","Latitude","Li_icpms"]].join(flags).to_csv(tables/"overlap_audit.csv",index=False)
        y=pd.to_numeric(frame.Li_icpms,errors="coerce")
        quality=pd.to_numeric(frame["SUM (no water)"],errors="coerce")
        bounds=cfg["data"]
        qc=quality.between(bounds["quality_min"],bounds["quality_max"])
        unique=~flags[["source_unique_id_overlap","coordinate_li_overlap","within_candidate_coordinate_li"]].any(axis=1)
        base_y=pd.to_numeric(base.Li_icpms,errors="coerce")
        cut=[-np.inf,100,696,1000,np.inf]
        distribution=pd.DataFrame({"base_raw":pd.cut(base_y,cut).value_counts(sort=False),
            "candidate":pd.cut(y,cut).value_counts(sort=False),
            "candidate_qc_no_identity_flag":pd.cut(y[qc&unique],cut).value_counts(sort=False)})
        distribution.index=distribution.index.astype(str)
        distribution.to_csv(tables/"lithium_bands.csv")
        frame.isna().sum().rename("missing").to_csv(tables/"missingness.csv")
        frame.groupby("Rock_type",dropna=False).Li_icpms.agg(["size","min","median","max"]).to_csv(tables/"lithology.csv")
        _plots(frame,directory/"figures")
        if file_sha256(base_path)!=base_digest or any(file_sha256(p)!=digests[p.name] for p in paths):
            raise ValueError("Input changed during candidate EDA")
        summary={"raw_rows":harmonized.rows_read,"candidate_rows":len(frame),"qc_rows":int(qc.sum()),
                 "qc_no_identity_flag_rows":int((qc&unique).sum()),"above_696_qc_no_identity_flag":int((qc&unique&y.gt(696)).sum()),
                 "above_1000_qc_no_identity_flag":int((qc&unique&y.gt(1000)).sum()),"approved_rows":0}
        print(pd.DataFrame([summary]).to_string(index=False))
        return complete_run(directory,signature_value=key,details={"model_group":"model_1",
            "source_key":spec.key,"doi":spec.doi,"source_sha256":digests,"base_sha256":base_digest,
            "acquisition_receipt":str(receipt),"summary":summary,"eligibility":"requires_review",
            "training_reusable":False})
    except Exception as exc:
        fail_run(directory,exc)
        raise


def run_candidates(base_path):
    records=[]
    for spec in M1_RESEARCH_SOURCES:
        try:
            paths,receipt=acquire(spec,DATA_DIR/"raw"/"model_1"/"research")
            result=profile(spec,paths,receipt,base_path)
            record=json.loads((result/"manifest.json").read_text(encoding="utf-8"))
            records.append({"source":spec.key,"state":"complete","eda_path":str(result),**record["summary"]})
            print(f"[OK] {spec.key}: {record['summary']}")
        except Exception as exc:
            records.append({"source":spec.key,"state":"blocked","reason":str(exc)})
            print(f"[BLOQUEADO] {spec.key}: {exc}")
    return records
