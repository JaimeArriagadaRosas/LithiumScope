from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
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


def load_model_with_identity(model_group: str):
    bundle, model_path = load_latest_model(model_group)
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
    }
    return bundle, model_path, identity
