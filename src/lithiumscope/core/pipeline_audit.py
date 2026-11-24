from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from lithiumscope.core.logging.session import get_logger

logger = get_logger("pipeline_audit")


@dataclass
class PipelineAudit:
    records: list[dict] = field(default_factory=list)
    source_records: list[dict] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    _previous: dict = field(default_factory=dict, repr=False)

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
        sources = frame["source_dataset"].astype("string").fillna("unknown") if "source_dataset" in frame else pd.Series("unknown", index=frame.index)
        for source, group in frame.groupby(sources, sort=True):
            self.source_records.append({"stage": stage, "source": str(source), "rows": len(group),
                                        "missing_cells": int(group.isna().sum().sum())})
        if frame.index.is_unique:
            ids = frame["source_sample"] if "source_sample" in frame else pd.Series(frame.index, index=frame.index)
            current = {index: (str(sources.loc[index]), str(ids.loc[index])) for index in frame.index}
            for index, (source, sample_id) in self._previous.items():
                if index not in current:
                    self.exclusions.append({"stage": stage, "row_index": str(index),
                                            "source": source, "sample_id": sample_id})
            self._previous = current
        else:
            logger.warning("Cannot track stage row exclusions: index is not unique")
            self._previous = {}
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
