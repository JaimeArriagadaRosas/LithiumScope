from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_1.pipeline import prepare_prediction_data
from lithiumscope.model_1.prediction.input_parser import load_prediction_input
from lithiumscope.model_1.schema import applicability_fraction
from lithiumscope.persistence.load_model import load_latest_model

logger = get_logger("model_1.predictor")


def predict_model_1_frame(
    frame: pd.DataFrame,
    *,
    bundle: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if bundle is None:
        bundle, _ = load_latest_model("model_1")

    schema = bundle["schema"]
    expected = list(schema.numeric) + list(schema.categorical)
    missing = [column for column in expected if column not in frame.columns]
    x = prepare_prediction_data(
        frame,
        bundle["algorithm"],
        schema,
    )

    predictions = bundle["estimator"].predict(x)
    output = frame.copy()
    output["Li_icpms_predicted"] = predictions

    interval = bundle.get("oof_absolute_residual_q90")
    if interval is not None:
        output["Li_icpms_interval_low_q90"] = predictions - float(interval)
        output["Li_icpms_interval_high_q90"] = predictions + float(interval)

    profile = bundle.get("applicability_profile", {})
    fractions = applicability_fraction(x, profile)
    output["out_of_training_range_fraction"] = fractions
    output["applicability_warning"] = fractions.map(
        lambda value: "OUT_OF_DOMAIN" if value > 0.25 else "OK"
    )
    diagnostics = {
        "algorithm": bundle.get("algorithm", "unknown"),
        "rows": len(frame),
        "expected_columns": expected,
        "missing_expected_columns": missing,
        "out_of_domain_rows": int((fractions > 0.25).sum()),
        "mean_out_of_training_range_fraction": float(fractions.mean()) if len(fractions) else 0.0,
        "oof_absolute_residual_q90": interval,
    }
    return output, diagnostics


def predict_model_1(
    path: Path,
    *,
    destination_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    bundle, model_path = load_latest_model("model_1")
    frame = load_prediction_input(path)
    output, diagnostics = predict_model_1_frame(frame, bundle=bundle)

    destination = (
        (destination_dir or (RESULTS_DIR / "model_1" / "predictions"))
        / f"{path.stem}_predictions.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)

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
    logger.info("Model 1 predictions written to %s", destination)
    return output, destination, model_path
