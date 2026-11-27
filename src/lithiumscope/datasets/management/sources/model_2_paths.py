"""Resolve historical M2 imagery across workspaces without falling back to old roots."""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT


def read_relocation(root: Path) -> dict:
    path = Path(root) / "data/processed/model_2/relocation.json"
    registry = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(registry, dict) or not isinstance(registry.get("images", {}), dict):
        raise ValueError("Invalid image relocation registry")
    return registry


def _windows_absolute(value):
    return PureWindowsPath(str(value)).is_absolute()


def _mapped_record(value, registry):
    images = registry.get("images", {})
    raw = str(value)
    if _windows_absolute(raw):
        aliases = [record for name, record in images.items()
                   if _windows_absolute(name) and PureWindowsPath(name) == PureWindowsPath(raw)]
        if len(aliases) > 1 and any(record != aliases[0] for record in aliases[1:]):
            raise ValueError("Ambiguous Windows relocation aliases")
        if aliases:
            return aliases[0]
    return images.get(raw)


def resolve_image_path(value: str | Path, *, root: Path | None = None,
                       relocation: dict | None = None, verify_hash: bool = False) -> Path:
    root = Path(root or PROJECT_ROOT).resolve()
    raw = Path(value)
    registry = read_relocation(root) if relocation is None else relocation
    mapped = _mapped_record(value, registry)
    if mapped is not None:
        result = (root / mapped["relative_path"]).resolve()
        if not result.is_relative_to(root):
            raise ValueError("Relocated image escapes project root")
        if not result.is_file():
            raise FileNotFoundError(result)
        expected = mapped.get("sha256")
        if verify_hash and expected and file_sha256(result) != expected:
            raise ValueError("Relocated image hash mismatch")
        return result
    previous = registry.get("source_root")
    if previous:
        historical = (PureWindowsPath(str(value)).is_relative_to(PureWindowsPath(previous))
                      if _windows_absolute(value) and _windows_absolute(previous)
                      else raw.is_absolute() and raw.is_relative_to(Path(previous)))
        if historical:
            raise ValueError(f"Historical image is not in relocation registry: {raw.name}")
    if _windows_absolute(value) and not raw.is_absolute():
        raise ValueError("Windows image path requires relocation on this platform")
    relative = Path(str(value).replace(chr(92), "/")) if not raw.is_absolute() else raw
    result = relative if relative.is_absolute() else (root / relative).resolve()
    if not relative.is_absolute() and not result.is_relative_to(root):
        raise ValueError("Relative image path escapes project root")
    return result


def inspect_image_links(values, *, root: Path, verify_hash: bool = False) -> dict:
    registry = read_relocation(root)
    cache = {}
    unavailable = 0
    total = 0
    for raw in values:
        total += 1
        value = str(raw)
        if value not in cache:
            try:
                path = resolve_image_path(value, root=root, relocation=registry, verify_hash=verify_hash)
                cache[value] = path.is_file() and path.stat().st_size > 0
            except (OSError, ValueError, TypeError, KeyError):
                cache[value] = False
        unavailable += not cache[value]
    return {"total": total, "unavailable": unavailable, "unique_references": len(cache),
            "relocation_hashes_checked": verify_hash, "spectral_content_verified": False}
