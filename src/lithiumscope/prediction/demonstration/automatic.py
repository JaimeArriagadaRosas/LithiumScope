"""Execute the versioned integrated demonstration."""
from __future__ import annotations
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.config import load_config
from lithiumscope.prediction.inputs.case_identity import ensure_case_ids
from lithiumscope.prediction.inputs.imagery import prepare_case_images
from lithiumscope.prediction.demonstration.cases import (
    audit_demo_overlap, demonstration_imagery_cache, load_demonstration_cases,
)
from lithiumscope.prediction.execution.session import PredictionSession
from lithiumscope.prediction.execution.contracts import IntegratedPredictionResult
from lithiumscope.prediction.execution.integrated import _run_integrated


def run_automatic_demonstration() -> IntegratedPredictionResult:
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
