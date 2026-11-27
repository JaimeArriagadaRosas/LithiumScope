"""Resolve frozen manifest paths without accessing the original workspace."""
from __future__ import annotations

import json
from pathlib import Path

from lithiumscope.core.paths import PROJECT_ROOT


def read_relocation(root: Path) -> dict:
    path = root / "data/processed/model_2/relocation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_image_path(value: str | Path, *, root: Path | None = None,
                       relocation: dict | None = None) -> Path:
    root = (root or PROJECT_ROOT).resolve()
    raw = Path(value)
    registry = read_relocation(root) if relocation is None else relocation
    mapped = registry.get("images", {}).get(str(raw))
    if mapped is not None:
        result = (root / mapped["relative_path"]).resolve()
        if not result.is_relative_to(root):
            raise ValueError("Relocated image escapes project root")
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    previous = registry.get("source_root")
    if previous and raw.is_absolute() and raw.is_relative_to(Path(previous)):
        raise ValueError(f"Historical image is not in relocation registry: {raw.name}")
    result = raw if raw.is_absolute() else (root / raw).resolve()
    if not raw.is_absolute() and not result.is_relative_to(root):
        raise ValueError("Relative image path escapes project root")
    return result
