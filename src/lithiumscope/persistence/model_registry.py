from __future__ import annotations

import json

from lithiumscope.core.paths import MODELS_DIR


def list_metadata(model_group: str) -> list[dict]:
    directory = MODELS_DIR / model_group
    if not directory.exists():
        return []

    candidates = list((directory / "metadata").glob("*.json"))
    candidates += list((directory / "trained").glob("*/metadata.json"))

    records: list[dict] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_metadata_path"] = str(path)
            records.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return records
