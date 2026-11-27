from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from lithiumscope.persistence.model_deserialization import load_model_artifact

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger

logger = get_logger("prediction.models")
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
        if mode not in {"complete", "demonstration"}:
            raise ValueError("Modo de predicción inválido")
        run_id = f"{mode}_{_timestamp_with_offset()}_{uuid4().hex[:12]}"
        root = RESULTS_DIR / "predictions" / run_id
        root.mkdir(parents=True, exist_ok=False)
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


def _latest_local_candidate(model_group: str) -> tuple[str, bytes]:
    """Newest saved pending candidate, not an assertion of scientific superiority."""
    candidates = []
    directory = MODELS_DIR / model_group / "trained"
    for path in directory.glob("training_*/model.joblib"):
        status = read_candidate_status(path)
        if status.get("status") != "pending":
            continue
        metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("run_id") != path.parent.name
                or status.get("run_id") != path.parent.name
                or status.get("model_group") != model_group
                or metadata.get("activation_at_save") != "candidate"):
            raise ValueError("Identidad del candidato local no verificada")
        saved = datetime.fromisoformat(metadata["saved_at_utc"])
        if saved.tzinfo is None:
            raise ValueError("Fecha del candidato local sin zona horaria")
        candidates.append((saved, path.parent.name, path, metadata))
    if not candidates:
        raise FileNotFoundError(f"No hay candidato local pendiente de {model_group}; entrene primero.")
    _, run_id, path, metadata = max(candidates, key=lambda item: (item[0], item[1]))
    integrity = json.loads(path.with_name("artifact_integrity.json").read_text(encoding="utf-8"))
    files = integrity.get("files", {})
    if (not isinstance(files, dict) or not {"model.joblib", "metadata.json"} <= set(files)
            or any(Path(name).name != name or not (path.parent / name).is_file()
                   or file_sha256(path.parent / name) != digest for name, digest in files.items())):
        raise ValueError("Integridad del candidato local no verificada")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != metadata.get("model_sha256") or files["model.joblib"] != metadata.get("model_sha256"):
        raise ValueError("Hash del candidato local no coincide")
    return run_id, payload


def load_model_with_identity(model_group: str, *, candidate_run_id: str | None = None):
    """Select an explicit candidate or published release for a *single demo*.

    Never modifies active_models.json, candidate_status.json or a model bundle.
    An omitted candidate_run_id preserves the published-active model selection.
    """
    if model_group not in {"model_1", "model_2"}:
        raise ValueError(f"Grupo de modelo desconocido: {model_group}")
    payload_digest = None
    if candidate_run_id is None:
        bundle, model_path = load_latest_model(model_group)
        selected_for = "active_release"
        candidate_status = None
    else:
        local_payload = None
        if candidate_run_id == "latest_local_candidate":
            candidate_run_id, local_payload = _latest_local_candidate(model_group)
        run_id = str(candidate_run_id)
        if not run_id.startswith("training_") or Path(run_id).name != run_id or any(
            value in run_id for value in ("..", "/", "\\", ":")
        ):
            raise ValueError("ID del candidato inválido")
        model_path = MODELS_DIR / model_group / "trained" / run_id / "model.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe el candidato solicitado: {model_path}")
        candidate_status = read_candidate_status(model_path).get("status")
        if candidate_status not in {"pending", "promoted"}:
            raise ValueError(
                f"El modelo {model_group}/{run_id} no es un candidato pendiente "
                f"ni un release publicado (estado: {candidate_status!r})."
            )
        if local_payload is not None and candidate_status != "pending":
            raise ValueError("El candidato local ya no está pendiente")
        payload = local_payload if local_payload is not None else model_path.read_bytes()
        payload_digest = sha256(payload).hexdigest()
        bundle = load_model_artifact(BytesIO(payload), model_group=model_group)
        selected_for = (
            "pinned_published_release_for_demonstration"
            if candidate_status == "promoted"
            else "experimental_demonstration"
        )

    metadata_path = model_path.parent / "metadata.json"
    metadata: dict = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    if payload_digest is not None and metadata.get("model_sha256") not in (None, payload_digest):
        raise ValueError(f"Model artifact hash differs from metadata: {model_path}")
    identity = {
        "model_group": model_group,
        "model_path": str(model_path),
        "model_sha256": payload_digest or file_sha256(model_path),
        "run_id": metadata.get("run_id", model_path.parent.name),
        "algorithm": metadata.get("algorithm", bundle.get("algorithm", "unknown")),
        "training_metrics": metadata.get("metrics", {}),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
        "selection": selected_for,
        "candidate_status": candidate_status,
        "release_promoted_by_demonstration": False,
    }
    if model_group == "model_2":
        bands = bundle.get("bands", bundle.get("band_names"))
        identity["bands"] = list(bands) if isinstance(bands, (list, tuple)) else None
        identity["feature_extractor_version"] = bundle.get("feature_extractor_version")
        identity["feature_count"] = len(bundle.get("features", [])) or None
    logger.info("Model selected group=%s run=%s algorithm=%s sha256=%s selection=%s",
                model_group, identity["run_id"], identity["algorithm"], identity["model_sha256"], selected_for)
    return bundle, model_path, identity
