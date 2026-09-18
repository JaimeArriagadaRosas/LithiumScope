from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetStatus:
    key: str
    ready: bool
    path: str | None
    detail: str = ""
