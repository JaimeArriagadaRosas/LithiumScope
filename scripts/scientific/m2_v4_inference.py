"""Version-aware scoring for UNPUBLISHED Model 2 170-D spectral candidates.

Prevents the old 6-band-only scorer from silently filling 44 red-edge
features with zero. Both patches must belong to the SAME satellite scene.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from lithiumscope.model_2.prediction.image_parser import image_to_feature_frame
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.methodology.rededge_acquisition import cached_rededge_sample

from scripts.scientific.m2_rededge_v4_competition import ARMS


def score_candidate(
    six_band_patch: Path,
    scene_id: str,
    latitude: float,
    longitude: float,
    *,
    candidate_path: Path,
    spectral_cache: Path,
) -> dict:
    bundle = joblib.load(candidate_path)
    if (
        bundle.get("experimental_only") is not True
        or bundle.get("feature_extractor_version") != 4
        or bundle.get("algorithm") not in ARMS
        or len(bundle.get("features", [])) != 170
        or not bundle.get("rededge_same_scene_required")
    ):
        raise ValueError("Expected an unpublished v4 170-feature M2 bundle")
    patch = Path(six_band_patch)
    image = np.asarray(load_imagery(patch), float)
    if image.shape[0] != 6:
        raise ValueError("M2 v4 requires the original SIX bands for inference")
    coverage = float(np.isfinite(image).all(axis=0).mean())
    cutoff = float(bundle.get("quality_min_joint_valid", 0.95))
    if coverage < cutoff:
        return {
            "status": "ABSTAIN_OLD_BANDS_QA95",
            "score": None,
            "sixband_coverage": coverage,
        }
    frame = image_to_feature_frame(
        patch,
        band_names=bundle["base_band_names"],
        normalize_per_band=bool(bundle.get("normalize_per_band", False)),
    )
    base = frame.iloc[0].to_dict()
    if len(base) != 126:
        raise ValueError("M2 v4 original spectral feature schema drift")
    extr, diagnostics = cached_rededge_sample(
        SimpleNamespace(
            sample_id=f"predict_{scene_id}_{latitude:.6f}_{longitude:.6f}",
            sentinel_scene_id=scene_id,
            latitude=float(latitude),
            longitude=float(longitude),
            image_path=str(patch),
        ),
        output=spectral_cache,
        min_joint_valid=0.05,
    )
    if diagnostics["rededge_joint_valid_fraction"] < cutoff:
        return {
            "status": "ABSTAIN_NEW_BANDS_QA95",
            "score": None,
            "sixband_coverage": coverage,
            "rededge_coverage": diagnostics["rededge_joint_valid_fraction"],
        }
    base.update(extr)
    expected = bundle["features"]
    if len(base) != 170 or set(base) != set(expected):
        raise ValueError("M2 v4 candidate features differ from trained schema")
    import pandas as pd

    x = pd.DataFrame([{name: base[name] for name in expected}])
    score = float(bundle["estimator"].predict_proba(x)[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("M2 v4 candidate emitted invalid score")
    return {
        "status": "EXPERIMENTAL_SCORED",
        "score": score,
        "algorithm": bundle["algorithm"],
        "sixband_coverage": coverage,
        "rededge_coverage": diagnostics["rededge_joint_valid_fraction"],
        "scene_id": scene_id,
        "not_a_calibrated_deposit_probability": True,
        "model_1_chemistry_used_as_m2_features": False,
        "not_approved_for_release": True,
    }
