"""Predict before unsealing labels, using exact frozen local model bytes."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.evaluation.holdout.registry import (
    artifact, verified_registration, verify_artifact, write_sealed,
)


def _predict_m1(sites,bundle):
    from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
    from lithiumscope.core.scientific_checks import assert_no_target_leakage
    names=list(bundle["schema"].numeric)+list(bundle["schema"].categorical)
    assert_no_target_leakage("Li_icpms",names)
    output,_=predict_model_1_frame(sites,bundle=bundle)
    return pd.DataFrame({"sample_id":sites.sample_id.astype(str),
                         "prediction":output.Li_icpms_predicted,
                         "warning":output.applicability_warning})


def _predict_m2(sites,bundle,images,li_cutoff):
    from lithiumscope.model_2.prediction.predictor import score_model_2_image
    if bundle.get("threshold_ppm") is None or not math.isclose(float(bundle["threshold_ppm"]),li_cutoff,rel_tol=0,abs_tol=1e-10):
        raise ValueError("Model Li endpoint differs from the preregistered common endpoint")
    cut=float(bundle.get("operating_threshold",.5))
    if not np.isfinite(cut) or not 0<=cut<=1:
        raise ValueError("Model has invalid frozen operating threshold")
    records=[]
    for row,image in zip(sites.itertuples(index=False),images,strict=True):
        if str(row.sample_id)!=image["sample_id"]:
            raise ValueError("Frozen imagery order differs")
        path=verify_artifact(image["image"])
        verify_artifact(image["sidecar"])
        result,details=score_model_2_image(path,bundle=bundle,allow_acquisition=False)
        if int(bundle.get("feature_extractor_version",3))==4 and result["prospectivity_score"] is not None:
            for key in ("rededge_patch_sha256","scene_band_hash"):
                if getattr(row,key,None)!=details.get(key) or not details.get(key):
                    raise ValueError("V4 imagery evidence differs from preregistered red-edge hashes")
        records.append({"sample_id":str(row.sample_id),"prediction":result["prospectivity_score"],
                        "warning":result["applicability_warning"],"quality_status":result.get("quality_status","EVALUATED"),
                        "operating_threshold":cut})
    return pd.DataFrame(records)


def score_registered(root, *, trusted_local_bundles=False):
    root=Path(root).resolve()
    if not trusted_local_bundles:
        raise ValueError("Only deserialize trusted local model artifacts; explicit approval required")
    if any((root/name).exists() for name in ("scores.json","label_reveal.json","evaluation.json","reference_scores.csv","candidate_scores.csv")):
        raise FileExistsError("Prediction evidence already exists; no repeated scoring after reveal")
    record,sites=verified_registration(root)
    results={}
    for arm,model in record["models"].items():
        bundle=joblib.load(verify_artifact(model))
        if record["model_group"]=="model_1":
            frame=_predict_m1(sites,bundle)
        else:
            frame=_predict_m2(sites,bundle,record["imagery"],record["label_cutoff_ppm"])
        if frame.sample_id.duplicated().any() or set(frame.sample_id.astype(str))!=set(sites.sample_id.astype(str)):
            raise ValueError("Frozen model did not return exactly one record per site")
        results[arm]=frame
        verify_artifact(model)
    verified_registration(root)
    outputs={}
    for arm,frame in results.items():
        path=root/f"{arm}_scores.csv"
        with path.open("x",encoding="utf-8",newline="") as handle:
            frame.to_csv(handle,index=False)
        outputs[arm]=artifact(path)
    evidence={"scored_at_utc":datetime.now(timezone.utc).isoformat(),
              "registration_sha256":file_sha256(root/"registration.json"),"predictions":outputs,
              "rows":len(sites),"labels_accessed_by_scoring":False,
              "scope":"Predictions of frozen artifacts; independence still requires documentary review."}
    write_sealed(root/"scores.json",evidence)
    return evidence
