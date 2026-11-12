"""Focused integrated prediction phase; stable original public entrypoint."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lithiumscope.prediction.integrated import pd

def _score_model_2_cases(
    cases: pd.DataFrame,
    bundle: dict,
) -> tuple[pd.DataFrame, dict]:
    from lithiumscope.prediction.integrated import Path, image_to_feature_frame, model_2_case_text, out_of_range_fraction, pd
    records: list[dict] = []
    missing_feature_counts: dict[str, int] = {}
    expected = list(bundle["features"])

    for _, row in cases.iterrows():
        if str(row.get("sentinel_status", "ready")) != "ready":
            continue
        image_path = Path(str(row["image_path"]))
        frame = image_to_feature_frame(
            image_path,
            band_names=bundle.get("band_names", ()),
            normalize_per_band=bool(bundle.get("normalize_per_band", False)),
        )
        missing = [column for column in expected if column not in frame.columns]
        for column in missing:
            missing_feature_counts[column] = missing_feature_counts.get(column, 0) + 1
            frame[column] = 0.0

        score = float(bundle["estimator"].predict_proba(frame[expected])[0, 1])
        operating_threshold = float(
            bundle.get("operating_threshold", 0.5)
        )
        priority = "alta" if score >= 0.70 else "media" if score >= 0.40 else "baja"
        applicability = out_of_range_fraction(
            frame[expected],
            bundle.get("applicability_profile", {}),
        )
        record = {
            "case_id": str(row["case_id"]),
            "Li_icpms": row.get("Li_icpms"),
            "image": str(image_path),
            "prospectivity_score": score,
            "priority": priority,
            "operating_positive": bool(
                score >= operating_threshold
            ),
            "operating_threshold": operating_threshold,
            "algorithm": bundle.get("algorithm", "unknown"),
            "out_of_training_range_fraction": float(applicability),
            "applicability_warning": "OUT_OF_DOMAIN" if applicability > 0.25 else "OK",
            "sentinel_scene_id": row.get("sentinel_scene_id"),
            "sentinel_cloud_cover": row.get("sentinel_cloud_cover"),
            "sentinel_datetime": row.get("sentinel_datetime"),
            "warning": "Prioridad exploratoria relativa; no es probabilidad de yacimiento.",
        }
        record["scientific_interpretation"] = model_2_case_text(pd.Series(record))
        records.append(record)

    diagnostics = {
        "expected_features": expected,
        "missing_feature_counts": missing_feature_counts,
        "ready_cases": len(records),
        "requested_cases": len(cases),
        "failed_cases": int(len(cases) - len(records)),
        "operating_threshold": float(
            bundle.get("operating_threshold", 0.5)
        ),
        "operating_threshold_objective": bundle.get(
            "operating_threshold_objective"
        ),
    }
    return pd.DataFrame(records), diagnostics
