"""Use the single M2 scorer for integrated and individual inference."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import pandas as pd
from lithiumscope.core.logging.session import get_logger

logger = get_logger("prediction.scoring")

from lithiumscope.model_2.prediction.predictor import score_model_2_image
from lithiumscope.prediction.analysis.interpretation import model_2_case_text


def _score_model_2_cases(cases: pd.DataFrame, bundle: dict, *, strict: bool = False, include_abstentions: bool = False) -> tuple[pd.DataFrame, dict]:
    records, failures = [], []
    missing_feature_counts = Counter()
    for _, row in cases.iterrows():
        case_id = str(row["case_id"])
        if str(row.get("sentinel_status", "" if strict else "ready")) != "ready":
            failures.append({"case_id": case_id, "reason": str(row.get("sentinel_error", "Image unavailable"))})
            continue
        image = Path(str(row["image_path"]))
        try:
            prediction, detail = score_model_2_image(image, bundle=bundle)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("M2 scoring failed case=%s: %s", case_id, exc)
            if strict:
                raise
            failures.append({"case_id": case_id, "reason": str(exc)})
            continue
        missing_feature_counts.update(detail.get("missing_features", []))
        if prediction["prospectivity_score"] is None:
            failures.append({"case_id": case_id, "reason": prediction["warning"], "status": prediction["quality_status"]})
            if not include_abstentions:
                continue
        record = {**prediction, "case_id": case_id, "Li_icpms": row.get("Li_icpms"),
                  "sentinel_scene_id": row.get("sentinel_scene_id"),
                  "sentinel_cloud_cover": row.get("sentinel_cloud_cover"),
                  "sentinel_datetime": row.get("sentinel_datetime")}
        if not include_abstentions:
            record["scientific_interpretation"] = model_2_case_text(pd.Series(record))
        records.append(record)
    logger.info("M2 batch finished requested=%d returned=%d failed=%d", len(cases), len(records), len(failures))
    return pd.DataFrame(records), {
        "expected_features": list(bundle["features"]), "missing_feature_counts": dict(missing_feature_counts),
        "ready_cases": sum(row["prospectivity_score"] is not None for row in records), "requested_cases": len(cases), "failed_cases": len(failures),
        "failed_case_details": failures, "operating_threshold": float(bundle.get("operating_threshold", .5)),
        "operating_threshold_objective": bundle.get("operating_threshold_objective"),
    }
