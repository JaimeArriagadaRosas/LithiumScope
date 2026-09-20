from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from lithiumscope.core.paths import MODELS_DIR, PROJECT_ROOT

ACTIVE_MODELS_PATH = MODELS_DIR / "active_models.json"


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_active_models() -> dict[str, Any]:
    if not ACTIVE_MODELS_PATH.exists():
        return {"schema_version": 1, "models": {}}
    try:
        payload = json.loads(
            ACTIVE_MODELS_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "models": {}}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "models": {}}
    payload.setdefault("schema_version", 1)
    payload.setdefault("models", {})
    return payload


def active_model_path(model_group: str) -> Path | None:
    payload = read_active_models()
    entry = payload.get("models", {}).get(model_group)
    if not isinstance(entry, dict):
        return None
    raw_path = entry.get("model_path")
    if not raw_path:
        return None
    path = _resolve_path(str(raw_path))
    return path if path.is_file() else None


def set_active_models(
    models: dict[str, Path],
    *,
    source: dict[str, Any] | None = None,
) -> Path:
    payload = read_active_models()
    entries = payload.setdefault("models", {})
    now = datetime.now(timezone.utc).isoformat()

    for model_group, model_path in models.items():
        entries[model_group] = {
            "model_path": _project_relative(model_path),
            "activated_at_utc": now,
            "source": source or {"type": "local"},
        }

    ACTIVE_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_MODELS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    temporary.replace(ACTIVE_MODELS_PATH)
    return ACTIVE_MODELS_PATH


def set_active_model(
    model_group: str,
    model_path: Path,
    *,
    source: dict[str, Any] | None = None,
) -> Path:
    return set_active_models(
        {model_group: model_path},
        source=source,
    )
