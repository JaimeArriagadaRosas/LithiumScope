from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("pipeline_audit")


@dataclass
class PipelineAudit:
    records: list[dict] = field(default_factory=list)

    def capture(self, stage: str, frame: pd.DataFrame, note: str = "") -> None:
        record = {
            "stage": stage,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "missing_cells": int(frame.isna().sum().sum()),
            "note": note,
        }
        self.records.append(record)
        logger.info(
            "STEP %-28s rows=%d cols=%d missing=%d %s",
            stage,
            record["rows"],
            record["columns"],
            record["missing_cells"],
            note,
        )
        print(
            f"    ✓ {stage:<28} "
            f"filas={record['rows']:<5} columnas={record['columns']:<4} "
            f"faltantes={record['missing_cells']}"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
