from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4


class BoundedRunLog:
    """Small persistent log that keeps only the newest diagnostic lines."""

    def __init__(
        self,
        path: Path,
        *,
        max_lines: int = 160,
        max_chars: int = 48_000,
    ) -> None:
        self.path = path
        self._lock = RLock()
        self.max_lines = max(20, int(max_lines))
        self.max_chars = max(4_000, int(max_chars))
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        try:
            existing = self.path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except FileNotFoundError:
            existing = []
        self._lines = existing[-self.max_lines :]
        self.event("run", "inicio")

    @staticmethod
    def _clean(value: object) -> str:
        text = " ".join(str(value).split())
        return text[:600]

    def event(
        self,
        stage: str,
        message: str = "",
        **fields: object,
    ) -> None:
        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )
        parts = [
            timestamp,
            f"stage={self._clean(stage)}",
        ]
        if message:
            parts.append(
                f"message={self._clean(message)}"
            )
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(
                f"{self._clean(key)}={self._clean(value)}"
            )
        with self._lock:
            self._lines.append(" | ".join(parts))
            self._trim()
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def _trim(self) -> None:
        self._lines = self._lines[-self.max_lines :]
        while (
            self._lines
            and sum(len(line) + 1 for line in self._lines)
            > self.max_chars
        ):
            self._lines.pop(0)
