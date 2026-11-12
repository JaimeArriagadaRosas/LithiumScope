"""Focused integrated prediction phase; stable original public entrypoint."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lithiumscope.prediction.integrated import IntegratedPredictionResult, Path

def run_complete_prediction(
    model_1_path: Path,
    model_2_path: Path,
) -> IntegratedPredictionResult:
    from lithiumscope.prediction.integrated import IntegratedPredictionResult, Path, PredictionSession, _model_2_cases_from_input, _run_integrated, ensure_case_ids, file_sha256, load_prediction_input, shutil
    session = PredictionSession.create("complete")

    model_1_snapshot = (
        session.inputs / f"model_1_input{model_1_path.suffix.lower()}"
    )
    model_2_snapshot = (
        session.inputs / f"model_2_input{model_2_path.suffix.lower()}"
    )
    shutil.copy2(model_1_path, model_1_snapshot)
    shutil.copy2(model_2_path, model_2_snapshot)

    model_1_input = ensure_case_ids(
        load_prediction_input(model_1_snapshot)
    )
    model_2_cases, model_2_input_info = _model_2_cases_from_input(
        model_2_path,
        session,
        model_1_input,
    )
    input_info = {
        "model_1": {
            "original_path": str(model_1_path),
            "snapshot_path": str(model_1_snapshot),
            "sha256": file_sha256(model_1_snapshot),
        },
        "model_2": {
            **model_2_input_info,
            "original_path": str(model_2_path),
            "snapshot_path": str(model_2_snapshot),
            "snapshot_sha256": file_sha256(model_2_snapshot),
        },
        "pairing_rule": (
            "case_id; una imagen única se empareja automáticamente "
            "solo cuando Modelo 1 contiene un único caso"
        ),
    }
    return _run_integrated(
        session=session,
        model_1_input=model_1_input,
        model_2_cases=model_2_cases,
        input_info=input_info,
    )
