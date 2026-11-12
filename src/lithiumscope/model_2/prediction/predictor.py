from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_2.prediction.image_parser import image_to_feature_frame
from lithiumscope.model_2.steps.step_05_spectral_features import DEFAULT_BAND_NAMES
from lithiumscope.model_2.schema import out_of_range_fraction
from lithiumscope.persistence.load_model import load_latest_model


def score_model_2_image(
    path: Path,
    *,
    bundle: dict | None = None,
) -> tuple[dict, dict]:
    if bundle is None:
        bundle, _ = load_latest_model("model_2")

    # Only newly trained experimental candidates opt in. The published Model 2
    # has no quality_min_joint_valid and retains its historical behavior.
    required_coverage = bundle.get("quality_min_joint_valid")
    valid_coverage = None
    if required_coverage is not None:
        import numpy as np
        from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
        raw = np.asarray(load_imagery(path), dtype=float)
        if raw.ndim != 3 or raw.shape[0] != 6:
            raise ValueError("Expected a six-band spectral patch for candidate QA")
        valid_coverage = float(np.isfinite(raw).all(axis=0).mean())
        if valid_coverage < float(required_coverage):
            return ({
                "image": str(path),
                "prospectivity_score": None,
                "priority": "abstencion",
                "operating_positive": None,
                "operating_threshold": bundle.get("operating_threshold", 0.5),
                "algorithm": bundle.get("algorithm", "unknown"),
                "applicability_warning": "INSUFFICIENT_SPECTRAL_COVERAGE",
                "quality_status": "ABSTAIN_LOW_SPECTRAL_COVERAGE",
                "joint_valid_fraction": valid_coverage,
                "warning": "No rank: insufficient valid Sentinel-2 pixel coverage.",
            }, {
                "quality_status": "ABSTAIN_LOW_SPECTRAL_COVERAGE",
                "joint_valid_fraction": valid_coverage,
                "required_joint_valid_fraction": float(required_coverage),
                "experimental_model": True,
            })

    frame = image_to_feature_frame(
        path,
        band_names=bundle.get("band_names", DEFAULT_BAND_NAMES),
        normalize_per_band=bool(bundle.get("normalize_per_band", False)),
    )
    expected = list(bundle["features"])
    missing = [column for column in expected if column not in frame.columns]
    for column in missing:
        frame[column] = 0.0

    score = float(bundle["estimator"].predict_proba(frame[expected])[0, 1])
    operating_threshold = float(
        bundle.get("operating_threshold", 0.5)
    )
    operating_positive = score >= operating_threshold
    label = "alta" if score >= 0.70 else "media" if score >= 0.40 else "baja"
    applicability = out_of_range_fraction(
        frame[expected],
        bundle.get("applicability_profile", {}),
    )

    payload = {
        "image": str(path),
        "prospectivity_score": score,
        "priority": label,
        "operating_positive": bool(operating_positive),
        "operating_threshold": operating_threshold,
        "algorithm": bundle.get("algorithm", "unknown"),
        "out_of_training_range_fraction": applicability,
        "applicability_warning": "OUT_OF_DOMAIN" if applicability > 0.25 else "OK",
        "warning": "Prioritization score, not deposit probability.",
    }
    if required_coverage is not None:
        payload["quality_status"] = "EVALUATED"
        payload["joint_valid_fraction"] = valid_coverage
    diagnostics = {
        "expected_features": expected,
        "missing_features": missing,
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
