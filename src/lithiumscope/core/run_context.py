from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lithiumscope.core.paths import RESULTS_DIR


@dataclass(frozen=True)
class RunContext:
    model_group: str
    run_id: str
    root: Path
    figures: Path
    tables: Path
    exports: Path

    @classmethod
    def create(cls, model_group: str) -> "RunContext":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = RESULTS_DIR / model_group / "runs" / run_id
        figures = root / "figures"
        tables = root / "tables"
        exports = root / "exports"
        for directory in (root, figures, tables, exports):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(model_group, run_id, root, figures, tables, exports)
