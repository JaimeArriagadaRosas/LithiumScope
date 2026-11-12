from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import joblib

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import MODELS_DIR, RESULTS_DIR
from lithiumscope.persistence.candidates import read_candidate_status
from lithiumscope.persistence.load_model import load_latest_model


def _timestamp_with_offset() -> str:
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    sign = "p" if offset.startswith("+") else "m"
    digits = offset[1:] if offset else "0000"
    return now.strftime("%Y%m%d_%H%M%S") + f"_{sign}{digits}"


@dataclass(frozen=True)
class PredictionSession:
    run_id: str
    mode: str
    root: Path
    model_1: Path
    model_2: Path
    inputs: Path
    cross_model: Path
    figures: Path

    @classmethod
    def create(cls, mode: str) -> "PredictionSession":
        run_id = f"{mode}_{_timestamp_with_offset()}"
        root = RESULTS_DIR / "predictions" / run_id
        model_1 = root / "model_1"
        model_2 = root / "model_2"
        inputs = root / "inputs"
        cross_model = root / "cross_model"
        figures = cross_model / "figures"
        for path in (model_1, model_2, inputs, cross_model, figures):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id,
            mode,
            root,
            model_1,
            model_2,
            inputs,
            cross_model,
            figures,
        )


def load_model_with_identity(model_group: str, *, candidate_run_id: str | None = None):
    """Select an explicit pending candidate for a *single prediction only*.

    Never modifies active_models.json, candidate_status.json or a model bundle.
    An omitted candidate_run_id preserves the published-active model selection.
    """
    if model_group not in {"model_1", "model_2"}:
        raise ValueError(f"Grupo de modelo desconocido: {model_group}")
    if candidate_run_id is None:
        bundle, model_path = load_latest_model(model_group)
        selected_for = "active_release"
        candidate_status = None
    else:
        run_id = str(candidate_run_id)
        if not run_id.startswith("training_") or Path(run_id).name != run_id or any(
            value in run_id for value in ("..", "/", "\\", ":")
        ):
            raise ValueError("ID del candidato inválido")
        model_path = MODELS_DIR / model_group / "trained" / run_id / "model.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe el candidato solicitado: {model_path}")
        candidate_status = read_candidate_status(model_path).get("status")
        if candidate_status != "pending":
            raise ValueError(
                f"El modelo {model_group}/{run_id} no está pendiente de aprobación "
                f"(estado: {candidate_status!r})."
            )
        bundle = joblib.load(model_path)
        selected_for = "experimental_demonstration"

    metadata_path = model_path.parent / "metadata.json"
    metadata: dict = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    identity = {
        "model_group": model_group,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "run_id": metadata.get("run_id", model_path.parent.name),
        "algorithm": metadata.get("algorithm", bundle.get("algorithm", "unknown")),
        "training_metrics": metadata.get("metrics", {}),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
        "selection": selected_for,
        "candidate_status": candidate_status,
        "release_promoted_by_demonstration": False,
    }
    return bundle, model_path, identity
