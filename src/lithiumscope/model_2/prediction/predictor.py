from pathlib import Path

import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_2.prediction.image_parser import image_to_feature_frame
from lithiumscope.model_2.steps.step_05_spectral_features import DEFAULT_BAND_NAMES
from lithiumscope.model_2.schema import out_of_range_fraction
from lithiumscope.persistence.load_model import load_latest_model


def predict_model_2(path: Path) -> tuple[dict, Path, Path]:
    bundle, model_path = load_latest_model("model_2")
    frame = image_to_feature_frame(
        path,
        band_names=bundle.get("band_names", DEFAULT_BAND_NAMES),
        normalize_per_band=bool(
            bundle.get("normalize_per_band", False)
        ),
    )
    expected = list(bundle["features"])
    for column in expected:
        if column not in frame.columns:
            frame[column] = 0.0

    score = float(
        bundle["estimator"].predict_proba(frame[expected])[0, 1]
    )
    label = (
        "alta"
        if score >= 0.70
        else "media"
        if score >= 0.40
        else "baja"
    )
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
        "applicability_warning": (
            "OUT_OF_DOMAIN" if applicability > 0.25 else "OK"
        ),
        "warning": (
            "Prioritization score, not deposit probability."
        ),
    }
    destination = (
        RESULTS_DIR
        / "model_2"
        / "predictions"
        / f"{path.stem}_prospectivity.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(destination, index=False)
    return payload, destination, model_path
