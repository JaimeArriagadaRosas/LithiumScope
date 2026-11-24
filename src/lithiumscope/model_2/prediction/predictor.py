from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_2.prediction.spectral_contract import prepare_spectral_input, SpectralAbstention
from lithiumscope.model_2.schema import out_of_range_fraction
from lithiumscope.core.scientific.applicability import complete_range_profile
from lithiumscope.persistence.load_model import load_latest_model


def score_model_2_image(
    path: Path,
    *,
    bundle: dict | None = None,
) -> tuple[dict, dict]:
    if bundle is None:
        bundle, _ = load_latest_model("model_2")

    required_coverage = bundle.get("quality_min_joint_valid")
    try:
        frame, spectral = prepare_spectral_input(path, bundle)
    except SpectralAbstention as exc:
        warning = "INSUFFICIENT_SPECTRAL_COVERAGE" if exc.status == "ABSTAIN_LOW_SPECTRAL_COVERAGE" else "INSUFFICIENT_SPECTRAL_EVIDENCE"
        return ({
            "image": str(path), "prospectivity_score": None, "priority": "abstencion",
            "operating_positive": None, "operating_threshold": bundle.get("operating_threshold", .5),
            "algorithm": bundle.get("algorithm", "unknown"), "applicability_warning": warning,
            "quality_status": exc.status, "warning": str(exc), **exc.details,
        }, {"quality_status": exc.status, "warning": str(exc), **exc.details})
    expected = list(bundle["features"])
    valid_coverage = spectral["joint_valid_fraction"]
    missing = []
    score = float(bundle["estimator"].predict_proba(frame[expected])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Estimator emitted an invalid prospectivity score")
    operating_threshold = float(
        bundle.get("operating_threshold", 0.5)
    )
    operating_positive = score >= operating_threshold
    label = "alta" if score >= 0.70 else "media" if score >= 0.40 else "baja"
    profile = bundle.get("applicability_profile", {})
    profile_complete = complete_range_profile(profile, expected)
    applicability = out_of_range_fraction(frame, profile) if profile_complete else float("nan")

    payload = {
        "image": str(path),
        "prospectivity_score": score,
        "priority": label,
        "operating_positive": bool(operating_positive),
        "operating_threshold": operating_threshold,
        "algorithm": bundle.get("algorithm", "unknown"),
        "out_of_training_range_fraction": applicability,
        "applicability_warning": ("OUT_OF_DOMAIN" if applicability > .25 else "OK") if profile_complete else "INSUFFICIENT_PROFILE",
        "warning": "Prioritization score, not deposit probability.",
    }
    if required_coverage is not None:
        payload["quality_status"] = "EVALUATED"
        payload["joint_valid_fraction"] = valid_coverage
    diagnostics = {
        "expected_features": expected,
        "missing_features": missing,
        "profile_complete": profile_complete,
        **spectral,
        "out_of_training_range_fraction": float(applicability),
        "threshold_ppm": bundle.get("threshold_ppm"),
        "target_quantile": bundle.get("target_quantile"),
        "operating_threshold": operating_threshold,
        "operating_threshold_objective": bundle.get(
            "operating_threshold_objective"
        ),
    }
    if required_coverage is not None:
        diagnostics["quality_status"] = "EVALUATED"
        diagnostics["joint_valid_fraction"] = valid_coverage
        diagnostics["required_joint_valid_fraction"] = float(required_coverage)
    return payload, diagnostics


def predict_model_2(
    path: Path,
    *,
    destination_dir: Path | None = None,
) -> tuple[dict, Path, Path]:
    bundle, model_path = load_latest_model("model_2")
    payload, diagnostics = score_model_2_image(path, bundle=bundle)
    destination = (
        (destination_dir or (RESULTS_DIR / "model_2" / "predictions"))
        / f"{path.stem}_prospectivity.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(destination, index=False)

    diagnostics.update(
        {
            "input_path": str(path),
            "input_sha256": file_sha256(path),
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
        }
    )
    destination.with_name(destination.stem + "_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return payload, destination, model_path
