from __future__ import annotations

import json

from lithiumscope.core.paths import MODELS_DIR


def list_metadata(model_group: str) -> list[dict]:
    directory = MODELS_DIR / model_group / "metadata"
    if not directory.exists():
        return []
    records: list[dict] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_metadata_path"] = str(path)
            records.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return records
