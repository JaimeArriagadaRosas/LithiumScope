"""Historical experimental v4 command delegates to the canonical M2 scorer."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from lithiumscope.model_2.prediction.predictor import score_model_2_image
from lithiumscope.model_2.prediction.spectral_contract import _source_metadata, SpectralAbstention
from scripts.scientific.m2_rededge_v4_competition import ARMS


def score_candidate(
    six_band_patch: Path,
    scene_id: str,
    latitude: float,
    longitude: float,
    *,
    candidate_path: Path,
    spectral_cache: Path,
    allow_acquisition: bool = False,
) -> dict:
    """Use only trusted user-owned bundles. Never invent provenance from arguments."""
    bundle=joblib.load(candidate_path)
    if (bundle.get("experimental_only") is not True
            or bundle.get("feature_extractor_version")!=4
            or bundle.get("algorithm") not in ARMS
            or len(bundle.get("features",[]))!=170
            or not bundle.get("rededge_same_scene_required")):
        raise ValueError("Expected an unpublished v4 170-feature M2 bundle")
    path=Path(six_band_patch)
    try:
        meta=_source_metadata(path)
    except SpectralAbstention as exc:
        return {"status":exc.status,"score":None,"reason":str(exc),"not_approved_for_release":True}
    if (meta["scene_id"]!=scene_id or not np.allclose(
            [float(meta["latitude"]),float(meta["longitude"])],[latitude,longitude],rtol=0,atol=1e-8)):
        return {"status":"ABSTAIN_MISSING_SCENE_PROVENANCE","score":None,
                "reason":"Arguments do not match patch sidecar","not_approved_for_release":True}
    payload,diagnostics=score_model_2_image(path,bundle=bundle,allow_acquisition=allow_acquisition,
                                           spectral_cache=spectral_cache)
    score=payload.get("prospectivity_score")
    return {"status":"EXPERIMENTAL_SCORED" if score is not None else payload["quality_status"],
            "score":score,"algorithm":bundle["algorithm"],
            "sixband_coverage":diagnostics.get("joint_valid_fraction"),
            "rededge_coverage":diagnostics.get("rededge_joint_valid_fraction"),
            "scene_id":scene_id,"not_a_calibrated_deposit_probability":True,
            "model_1_chemistry_used_as_m2_features":False,"not_approved_for_release":True,
            "diagnostics":diagnostics}
