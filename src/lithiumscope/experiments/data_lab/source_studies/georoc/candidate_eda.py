"""Candidate-only descriptive audit. Existing training data never gets overwritten."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from lithiumscope.datasets.adapters.georoc.research_view import normalize_research_export

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.scientific.eda_statistics import missingness_table, feature_profiles, cohort_summary
from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.management.snapshots import dataset_implementation_revision
from lithiumscope.core.reproducibility.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.adapters.georoc.ingestion import ingest_georoc_files
from lithiumscope.results.eda_artifacts import complete_run, find_compatible, new_run, fail_run


def _keys(frame):
    """Conservative cross-source identity flags; not proof of independence."""
    keys = pd.DataFrame(index=frame.index)
    for name in ("source_unique_id","source_sample","source_publication"):
        values = frame.get(name,pd.Series("",index=frame.index)).fillna("").astype(str).str.strip().str.casefold()
        keys[name] = values.replace({"nan":"","none":"","<na>":""})
    numeric = frame.reindex(columns=["Longitude", "Latitude", "Li_icpms"]).apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric).all(axis=1)
    keys["coordinate_li"] = (
        numeric.Longitude.round(5).astype(str)+"|"+
        numeric.Latitude.round(5).astype(str)+"|"+
        numeric.Li_icpms.round(6).astype(str)).where(valid, "")
    return keys


def overlap_flags(candidate, base):
    left,right = _keys(candidate),_keys(base)
    result = pd.DataFrame(index=candidate.index)
    for key in ("source_unique_id","source_sample","source_publication","coordinate_li"):
        result[key+"_overlap"] = left[key].ne("") & left[key].isin(set(right[key])- {""})
    result["within_candidate_coordinate_li"] = left.coordinate_li.ne("") & left.coordinate_li.duplicated(keep=False)
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


def profile(spec, paths, receipt, base_path, *, output_root=None, availability_complete=True, return_reuse=False):
    paths = tuple(map(Path,paths))
    cfg=load_config("model_1")
    digests={p.name:file_sha256(p) for p in paths}
    if len(digests)!=len(paths) or not paths:
        raise ValueError("Candidate source files must have unique names")
    base_digest=file_sha256(base_path) if base_path is not None else None
    from lithiumscope.datasets.adapters.georoc import schema as georoc_ingestion_schema, transformation as georoc_ingestion_transform, ingestion as georoc_ingestion
    from lithiumscope.core.scientific import eda_statistics
    from lithiumscope.datasets.readers import georoc_raw_reader
    from lithiumscope.datasets.adapters.georoc import research_view
    from importlib.metadata import version
    key=canonical_json_hash({"source":spec.key,"files":digests,"base":base_digest,"availability_complete":availability_complete,"cfg":cfg["data_sources"]["georoc"],
        "implementation":dataset_implementation_revision("candidate-eda-v3"),
        "code":[file_sha256(Path(__file__)),file_sha256(Path(georoc_ingestion_schema.__file__)),
                file_sha256(Path(georoc_ingestion_transform.__file__)),file_sha256(Path(georoc_ingestion.__file__)),
                file_sha256(Path(georoc_raw_reader.__file__)),file_sha256(Path(research_view.__file__)),file_sha256(Path(eda_statistics.__file__))],
        "versions":{p:version(p) for p in ("pandas","numpy","matplotlib")}})
    root=Path(output_root) if output_root else RESULTS_DIR/"model_1"/"candidate_eda"/spec.key
    cached=find_compatible(root,key)
    if cached:
        print(f"[OK] EDA candidato vigente: {cached}")
        return (cached, True) if return_reuse else cached
    directory=new_run(root)
    try:
        # Harmonized view is an analysis artifact, not a training source.
        tables=directory/"tables"
        settings=cfg["data_sources"]["georoc"]
        # Normalize encoding in a temporary view; original downloads remain immutable.
        with TemporaryDirectory() as temp:
            normalized=[]
            normalization={}
            for path in paths:
                target=Path(temp)/path.name
                normalization[path.name]=normalize_research_export(path,target)
                normalized.append(target)
            (tables/"raw_structure_audit.json").write_text(json.dumps(normalization,indent=2,ensure_ascii=False),encoding="utf-8")
            harmonized=ingest_georoc_files(normalized,tables/"candidate_view.csv",
                audit_path=tables/"ingestion_audit.json",source_name=spec.key,
                minimum_predictors=int(settings["minimum_predictors"]),
                allowed_material_types=tuple(settings["allowed_material_types"]),
                iron_total_mode=settings["iron_total_mode"])
        (tables/"raw_source_identity.json").write_text(json.dumps({"sha256":digests,"encodings":{name:record["encoding"] for name,record in normalization.items()}},indent=2),encoding="utf-8")
        frame=pd.read_csv(harmonized.path,low_memory=False)
        base=pd.read_csv(base_path,low_memory=False) if base_path is not None else pd.DataFrame(columns=["Longitude","Latitude","Li_icpms"])
        flags=overlap_flags(frame,base)
        # No row gets approved automatically; flags are screening evidence.
        frame[["source_sample","source_file","Longitude","Latitude","Li_icpms"]].join(flags).to_csv(tables/"overlap_audit.csv",index=False)
        y=pd.to_numeric(frame.Li_icpms,errors="coerce")
        quality=pd.to_numeric(frame["SUM (no water)"],errors="coerce")
        bounds=cfg["data"]
        qc=quality.between(bounds["quality_min"],bounds["quality_max"])
        unique=~flags[["source_unique_id_overlap","coordinate_li_overlap","within_candidate_coordinate_li"]].any(axis=1)
        if base_path is None:
            unique[:] = False  # No historical comparison, no eligibility claim.
        base_y=pd.to_numeric(base.Li_icpms,errors="coerce")
        cut=[-np.inf,0,5,10,20,40,60,80,100,126.12,696,1000,np.inf]
        distribution=pd.DataFrame({"base_raw":pd.cut(base_y,cut).value_counts(sort=False),
            "candidate":pd.cut(y,cut).value_counts(sort=False),
            "candidate_qc_no_identity_flag":pd.cut(y[qc&unique],cut).value_counts(sort=False)})
        distribution.index=distribution.index.astype(str)
        distribution.to_csv(tables/"lithium_bands.csv")
        missingness_table(frame).to_csv(tables/"missingness.csv")
        feature_profiles(frame,list(frame.columns),stage="candidate").to_csv(tables/"feature_profiles.csv",index=False)
        cohort_summary(frame,"Li_icpms").to_csv(tables/"cohort_summary.csv",index=False)
        if "Rock_type" in frame:
            frame.groupby("Rock_type",dropna=False).Li_icpms.agg(["size","min","median","max"]).to_csv(tables/"lithology.csv")
        coverage=pd.DataFrame({"longitude_cell":np.floor(frame.Longitude),
            "latitude_cell":np.floor(frame.Latitude),"li":y,"screened":qc&unique,
            "high_screened":qc&unique&y.gt(696)})
        coverage.groupby(["longitude_cell","latitude_cell"]).agg(
            rows=("li","size"),screened=("screened","sum"),
            high_screened=("high_screened","sum")).to_csv(tables/"spatial_coverage.csv")
        _plots(frame,directory/"figures")
        if (base_path is not None and file_sha256(base_path)!=base_digest) or any(file_sha256(p)!=digests[p.name] for p in paths):
            raise ValueError("Input changed during candidate EDA")
        summary={"raw_rows":harmonized.rows_read,"candidate_rows":len(frame),"qc_rows":int(qc.sum()),
                 "qc_no_identity_flag_rows":int((qc&unique).sum()),"above_696_qc_no_identity_flag":int((qc&unique&y.gt(696)).sum()),
                 "above_1000_qc_no_identity_flag":int((qc&unique&y.gt(1000)).sum()),"approved_rows":0}
        get_logger("experiments.data_lab").info("Candidate EDA source=%s rows=%d flagged=%d",spec.key,len(frame),int((~unique).sum()))
        print(pd.DataFrame([summary]).to_string(index=False))
        result = complete_run(directory,signature_value=key,details={"model_group":"model_1",
            "source_key":spec.key,"doi":spec.doi,"source_sha256":digests,"base_sha256":base_digest,
            "acquisition_receipt":str(receipt) if receipt is not None else None,"summary":summary,"eligibility":"requires_review",
            "all_originals_available":availability_complete,"historical_overlap_evaluated":base_path is not None,
            "training_reusable":False})
        return (result, False) if return_reuse else result
    except Exception as exc:
        fail_run(directory,exc)
        raise

