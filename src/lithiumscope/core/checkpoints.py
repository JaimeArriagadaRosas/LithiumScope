from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


class FoldCheckpointStore:
    def __init__(self, root: Path, algorithm: str) -> None:
        self.root = root / algorithm
        self.root.mkdir(parents=True, exist_ok=True)

    def fold_path(self, fold: int) -> Path:
        return self.root / f"fold_{fold:02d}.json"

    def load_fold(self, fold: int, expected_test_indices) -> dict | None:
        path = self.fold_path(fold)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if list(payload.get("test_indices", [])) != [int(value) for value in expected_test_indices]:
            return None
        return payload

    def save_fold(
        self,
        fold: int,
        test_indices,
        predictions,
        metrics: dict,
        **extra,
    ) -> Path:
        path = self.fold_path(fold)
        _atomic_json_write(
            path,
            {
                "fold": fold,
                "test_indices": [int(value) for value in test_indices],
                "predictions": [float(value) for value in predictions],
                "metrics": metrics,
                **extra,
            },
        )
        return path

    @property
    def algorithm_path(self) -> Path:
        return self.root / "algorithm.json"

    def load_algorithm(self) -> dict | None:
        if not self.algorithm_path.exists():
            return None
        try:
            return json.loads(self.algorithm_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save_algorithm(self, payload: dict) -> Path:
        _atomic_json_write(self.algorithm_path, payload)
        return self.algorithm_path
