"""Execute a complete prediction using verified input snapshots."""
from __future__ import annotations
from pathlib import Path
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.prediction.inputs.case_identity import ensure_case_ids
from lithiumscope.prediction.inputs.model_2_cases import _model_2_cases_from_input
from lithiumscope.prediction.inputs.snapshots import snapshot_input
from lithiumscope.prediction.execution.session import PredictionSession
from lithiumscope.prediction.execution.contracts import IntegratedPredictionResult
from lithiumscope.prediction.execution.integrated import _run_integrated


def run_complete_prediction(
    model_1_path: Path,
    model_2_path: Path,
) -> IntegratedPredictionResult:
    model_1_path, model_2_path = Path(model_1_path).resolve(), Path(model_2_path).resolve()
    session = PredictionSession.create("complete")

    model_1_snapshot = (
        session.inputs / f"model_1_input{model_1_path.suffix.lower()}"
    )
    model_2_snapshot = (
        session.inputs / f"model_2_input{model_2_path.suffix.lower()}"
    )
    m1_digest = snapshot_input(model_1_path, model_1_snapshot)
    m2_digest = snapshot_input(model_2_path, model_2_snapshot)
    sidecars = {}
    if model_2_path.suffix.lower() in {".npy", ".tif", ".tiff"}:
        # Both supported provenance conventions are preserved, including an
        # ambiguous pair that must still be rejected by the spectral contract.
        for source, target in (
            (model_2_path.with_suffix(model_2_path.suffix + ".json"), model_2_snapshot.with_suffix(model_2_snapshot.suffix + ".json")),
            (model_2_path.with_suffix(".json"), model_2_snapshot.with_suffix(".json")),
        ):
            if source.is_file():
                sidecars[target.relative_to(session.root).as_posix()] = snapshot_input(source, target)

    model_1_input = ensure_case_ids(
        load_data(model_1_snapshot)
    )
    model_2_cases, model_2_input_info = _model_2_cases_from_input(
        model_2_snapshot,
        session,
        model_1_input,
        reference_dir=model_2_path.parent,
    )
    input_info = {
        "model_1": {
            "original_path": str(model_1_path),
            "snapshot_path": model_1_snapshot.relative_to(session.root).as_posix(),
            "sha256": m1_digest,
        },
        "model_2": {
            **model_2_input_info,
            "original_path": str(model_2_path),
            "snapshot_path": model_2_snapshot.relative_to(session.root).as_posix(),
            "snapshot_sha256": m2_digest,
            "sidecars": sidecars,
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
