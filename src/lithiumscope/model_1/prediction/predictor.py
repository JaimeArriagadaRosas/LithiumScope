from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_1.preparation.prediction_data import prepare_prediction_data
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.model_1.schema import applicability_fraction
from lithiumscope.core.scientific.applicability import complete_range_profile
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
    missing_from_raw_input = [
        column for column in expected if column not in frame.columns
    ]
    x = prepare_prediction_data(
        frame,
        bundle["algorithm"],
        schema,
    )
    missing_after_preparation = [
        column for column in expected if column not in x.columns
    ]
    generated_during_preparation = [
        column
        for column in missing_from_raw_input
        if column in x.columns
    ]

    provenance = x.attrs.get("prediction_provenance", {})
    missing_after_preparation = provenance.get("missing_expected_columns", missing_after_preparation)
    generated_during_preparation = provenance.get("generated_during_preparation", generated_during_preparation)
    estimator=bundle["estimator"]
    treatment_projection=None
    if bundle.get("treatment_policy") is not None and hasattr(estimator,"prediction_evidence"):
        predictions,treatment_projection=estimator.prediction_evidence(x)
        predictions=np.asarray(predictions,dtype=float)
    else:
        predictions=np.asarray(estimator.predict(x),dtype=float)
    if predictions.shape != (len(x),):
        raise ValueError("M1 estimator did not emit one prediction per input row")
    output = frame.copy()
    output["Li_icpms_predicted"] = predictions

    interval = bundle.get("oof_absolute_residual_q90")
    if interval is not None and np.isfinite(float(interval)) and float(interval) >= 0:
        lower=predictions-float(interval)
        output["Li_icpms_interval_low_q90"] = np.maximum(lower,0.) if treatment_projection is not None else lower
        output["Li_icpms_interval_high_q90"] = predictions + float(interval)

    profile = bundle.get("applicability_profile", {})
    profile_complete = complete_range_profile(profile, schema.numeric)
    fractions = applicability_fraction(x, profile) if profile_complete else pd.Series(np.nan, index=x.index)
    invalid_output = ~np.isfinite(predictions) | (predictions < 0)

    output["out_of_training_range_fraction"] = fractions
    output["applicability_warning"] = fractions.map(
        lambda value: "INSUFFICIENT_PROFILE" if pd.isna(value) else "OUT_OF_DOMAIN" if value > .25 else "OK"
    )
    output.loc[invalid_output, "applicability_warning"] = "INVALID_PHYSICAL_PREDICTION"
    if schema.numeric:
        output.loc[x[list(schema.numeric)].isna().all(axis=1) & ~invalid_output, "applicability_warning"] = "INSUFFICIENT_INPUT"
    coordinate_status=provenance.get("coordinate_status")
    if coordinate_status is not None and {"Latitude","Longitude"}.intersection(schema.numeric):
        output["coordinate_status"]=coordinate_status
        unresolved=output.coordinate_status.ne("valid_numeric_coordinates")
        output.loc[unresolved & ~invalid_output,"applicability_warning"]="INSUFFICIENT_COORDINATE_EVIDENCE"
    multivariate = bundle.get("multivariate_domain_profile")
    if multivariate is not None:
        from lithiumscope.model_1.diagnostics.domain import evaluate_domain
        domain = evaluate_domain(x, multivariate)
        for col in domain:
            output[col] = domain[col]
        # Do not override existing invalid/insufficient warnings.
        replace = output.applicability_warning.eq("OK") & domain.multivariate_warning.ne("OK")
        output.loc[replace,"applicability_warning"] = domain.loc[replace,"multivariate_warning"]
    diagnostics = {
        "coordinate_report":provenance.get("coordinate_report"),
        "algorithm": bundle.get("algorithm", "unknown"),
        "treatment_policy":bundle.get("treatment_policy"),
        "effective_target_transform":bundle.get("effective_target_transform",bundle.get("target_transform")),
        "treatment_projection":treatment_projection,
        "rows": len(frame),
        "expected_columns": expected,
        "missing_raw_input_columns": missing_from_raw_input,
        "generated_during_preparation": generated_during_preparation,
        "missing_expected_columns": missing_after_preparation,
        "prepared_missing_counts": {name: int(x[name].isna().sum()) for name in expected},
        "profile_complete": profile_complete,
        "invalid_physical_predictions": int(invalid_output.sum()),
        "insufficient_evidence_rows": int(output["applicability_warning"].isin(["INSUFFICIENT_INPUT", "INSUFFICIENT_PROFILE", "INSUFFICIENT_COORDINATE_EVIDENCE"]).sum()),
        "out_of_domain_rows": int((fractions > 0.25).sum()),
        "mean_out_of_training_range_fraction": float(fractions.mean()) if fractions.notna().any() else None,
        "oof_absolute_residual_q90": interval,
        "interval_interpretation": "empirical_oof_residual_reference_not_calibrated",
        "multivariate_profile_available": multivariate is not None,
    }
    return output, diagnostics


def predict_model_1(
    path: Path,
    *,
    destination_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    bundle, model_path = load_latest_model("model_1")
    frame = load_data(path)
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
