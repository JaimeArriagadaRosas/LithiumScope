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
    def __init__(self, root: Path, algorithm: str) -> None:
        self.root = root / algorithm
        self.root.mkdir(parents=True, exist_ok=True)

    def fold_path(self, fold: int) -> Path:
        return self.root / f"fold_{fold:02d}.json"

    def load_fold(self, fold: int, expected_test_indices) -> dict | None:
        payload = _read_verified(self.fold_path(fold))
        expected = [int(value) for value in expected_test_indices]
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
            "test_indices": [int(value) for value in test_indices],
            "predictions": [float(value) for value in predictions],
            "metrics": metrics,
        }
        if not _valid_fold(payload, fold, payload["test_indices"]):
            raise ValueError("Invalid fold checkpoint: predictions or indices")
        _atomic_json_write(path, _seal(payload))
        return path

    @property
    def algorithm_path(self) -> Path:
        return self.root / "algorithm.json"

    def load_algorithm(self) -> dict | None:
        return _read_verified(self.algorithm_path)

    def save_algorithm(self, payload: dict) -> Path:
        _atomic_json_write(self.algorithm_path, _seal(payload))
        return self.algorithm_path
