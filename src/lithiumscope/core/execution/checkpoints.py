from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _seal(payload: dict) -> dict:
    return {"version": 2, "payload": payload, "sha256": _digest(payload)}


def _read_verified(path: Path) -> dict | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if envelope["version"] == 2 and isinstance(payload, dict) and _digest(payload) == envelope["sha256"]:
            return payload
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def read_checkpoint_evidence(path: Path, *, allow_legacy: bool = False) -> dict:
    """Read historical evidence without making it resumable. Invalid seals always fail."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint evidence is not an object")
    if {"version", "payload", "sha256"}.intersection(raw):
        payload = _read_verified(Path(path))
        if payload is None:
            raise ValueError("Invalid checkpoint evidence seal")
        return payload
    if not allow_legacy:
        raise ValueError("Unsealed historical evidence cannot be resumed")
    return raw


def _indices(values) -> list[int]:
    from numbers import Integral
    items = list(values)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in items):
        raise ValueError("Checkpoint indices must be nonnegative integers")
    return [int(v) for v in items]


def _valid_fold(payload: dict, fold: int, expected_indices: list[int]) -> bool:
    try:
        predictions = payload["predictions"]
        return (
            payload["fold"] == fold
            and payload["test_indices"] == expected_indices
            and len(expected_indices) == len(set(expected_indices))
            and all(i >= 0 for i in expected_indices)
            and len(predictions) == len(expected_indices)
            and all(math.isfinite(v) for v in predictions)
            and isinstance(payload["metrics"], dict)
        )
    except (ValueError, TypeError, KeyError):
        return False



class FoldCheckpointStore:
    def __init__(self, root: Path, algorithm: str, *, identity: str | None = None) -> None:
        if not algorithm or Path(algorithm).name != algorithm or algorithm in {".", ".."}:
            raise ValueError("Invalid algorithm checkpoint directory")
        self.root = root / algorithm
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        if identity is not None:
            marker = self.root / "checkpoint_identity.json"
            if marker.exists():
                if _read_verified(marker) != {"identity": identity, "algorithm": algorithm}:
                    raise ValueError("Checkpoint directory belongs to a different evaluation identity")
            elif any(self.root.iterdir()):
                raise ValueError("Unbound checkpoint directory is historical evidence, not reusable training state")
            else:
                _atomic_json_write(marker, _seal({"identity":identity,"algorithm":algorithm}))

    def _read(self, path):
        payload = _read_verified(path)
        if payload is None or payload.get("_evaluation_identity") != self.identity:
            return None
        return payload

    def _bound(self, payload):
        return {**payload, "_evaluation_identity": self.identity}

    def study_name(self, name: str) -> str:
        return f"{name}__{self.identity}" if self.identity is not None else name

    def fold_path(self, fold: int) -> Path:
        return self.root / f"fold_{fold:02d}.json"

    def load_fold(self, fold: int, expected_test_indices) -> dict | None:
        payload = self._read(self.fold_path(fold))
        expected = _indices(expected_test_indices)
        return payload if payload is not None and _valid_fold(payload, fold, expected) else None

    def save_fold(
        self,
        fold: int,
        test_indices,
        predictions,
        metrics: dict,
        **extra,
    ) -> Path:
        path = self.fold_path(fold)
        payload = {
            **extra,
            "fold": fold,
            "test_indices": _indices(test_indices),
            "predictions": [float(value) for value in predictions],
            "metrics": metrics,
        }
        if not _valid_fold(payload, fold, payload["test_indices"]):
            raise ValueError("Invalid fold checkpoint: predictions or indices")
        _atomic_json_write(path, _seal(self._bound(payload)))
        return path

    @property
    def algorithm_path(self) -> Path:
        return self.root / "algorithm.json"

    def load_algorithm(self) -> dict | None:
        return self._read(self.algorithm_path)

    def save_algorithm(self, payload: dict) -> Path:
        _atomic_json_write(self.algorithm_path, _seal(self._bound(payload)))
        return self.algorithm_path
