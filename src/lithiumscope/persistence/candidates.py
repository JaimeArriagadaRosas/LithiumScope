from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.persistence.active_models import active_model_path


def _status_path(
    model_path: Path,
) -> Path:
    return model_path.parent / "candidate_status.json"


def read_candidate_status(
    model_path: Path,
) -> dict:
    path = _status_path(model_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}
    return (
        payload
        if isinstance(payload, dict)
        else {}
    )


def pending_candidate_paths(
    model_group: str,
) -> list[Path]:
    directory = (
        MODELS_DIR
        / model_group
        / "trained"
    )
    if not directory.is_dir():
        return []
    active = active_model_path(
        model_group
    )
    candidates: list[Path] = []
    for model_path in directory.glob(
        "*/model.joblib"
    ):
        if (
            active is not None
            and model_path.resolve()
            == active.resolve()
        ):
            continue
        status = read_candidate_status(
            model_path
        )
        if status.get("status") == "pending":
            candidates.append(model_path)
    return sorted(
        candidates,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def latest_pending_candidate(
    model_group: str,
) -> Path | None:
    candidates = pending_candidate_paths(
        model_group
    )
    return (
        candidates[0]
        if candidates
        else None
    )


def update_candidate_status(
    model_path: Path,
    *,
    status: str,
    evaluation_id: str,
    details: dict | None = None,
) -> Path:
    payload = read_candidate_status(
        model_path
    )
    payload.update(
        {
            "schema_version": 1,
            "status": status,
            "evaluation_id": evaluation_id,
            "evaluated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "details": details or {},
        }
    )
    path = _status_path(model_path)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return path
