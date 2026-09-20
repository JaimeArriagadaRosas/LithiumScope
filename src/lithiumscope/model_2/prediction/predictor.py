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
    label = "alta" if score >= 0.70 else "media" if score >= 0.40 else "baja"
    applicability = out_of_range_fraction(
        frame[expected],
        bundle.get("applicability_profile", {}),
    )

    payload = {
        "image": str(path),
        "prospectivity_score": score,
        "priority": label,
        "algorithm": bundle.get("algorithm", "unknown"),
        "out_of_training_range_fraction": applicability,
        "applicability_warning": "OUT_OF_DOMAIN" if applicability > 0.25 else "OK",
        "warning": "Prioritization score, not deposit probability.",
    }
    diagnostics = {
        "expected_features": expected,
        "missing_features": missing,
        "out_of_training_range_fraction": float(applicability),
        "threshold_ppm": bundle.get("threshold_ppm"),
        "target_quantile": bundle.get("target_quantile"),
    }
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
