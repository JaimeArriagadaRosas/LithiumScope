"""Focused integrated prediction phase; stable original public entrypoint."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lithiumscope.prediction.integrated import IntegratedPredictionResult

def run_automatic_demonstration() -> IntegratedPredictionResult:
    from lithiumscope.prediction.integrated import IntegratedPredictionResult, PROJECT_ROOT, PredictionSession, _run_integrated, audit_demo_overlap, demonstration_imagery_cache, ensure_case_ids, file_sha256, load_config, load_demonstration_cases, prepare_case_images
    session = PredictionSession.create("demonstration")
    cases = ensure_case_ids(load_demonstration_cases())
    demo_snapshot = session.inputs / "demonstration_cases.csv"
    cases.to_csv(demo_snapshot, index=False)
    overlap = audit_demo_overlap(cases)
    demo_cfg = load_config("prediction")["demonstration"]
    imagery = prepare_case_images(
        cases,
        demonstration_imagery_cache(),
        datetime_override=str(demo_cfg["sentinel_datetime"]),
    )
    input_info = {
        "demonstration": {
            "cases_path": str(demo_cfg["cases_path"]),
            "snapshot_path": str(demo_snapshot),
            "cases_sha256": file_sha256(
                PROJECT_ROOT / str(demo_cfg["cases_path"])
            ),
            "source_name": demo_cfg["source_name"],
            "source_repository": demo_cfg["source_repository"],
            "source_commit": demo_cfg["source_commit"],
            "source_path": demo_cfg["source_path"],
            "source_license": demo_cfg["source_license"],
            "sentinel_datetime": demo_cfg["sentinel_datetime"],
        }
    }
    return _run_integrated(
        session=session,
        model_1_input=cases,
        model_2_cases=imagery,
        input_info=input_info,
        overlap_audit=overlap,
    )
