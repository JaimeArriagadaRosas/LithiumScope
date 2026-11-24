"""Deterministic preparation lineage; every discarded row must have an explicit cause."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json

import pandas as pd

from lithiumscope.core.logging.session import get_logger

logger = get_logger("pipeline_audit")


def stable_source_row_id(dataset_sha256, row_index):
    if not dataset_sha256:
        raise ValueError("Stable row identity requires a source dataset hash")
    return sha256(f"{dataset_sha256}:{row_index}".encode("utf-8")).hexdigest()


def _identity_value(value):
    text=str(value)
    return "unknown" if pd.isna(value) or text.strip().lower() in {"","nan","none","null","na","n/a","-","<na>"} else text


def _digest(frame):
    schema={"columns":list(map(str,frame.columns)),"dtypes":list(map(str,frame.dtypes))}
    descriptor=json.dumps(schema,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    hashes=pd.util.hash_pandas_object(frame,index=True).to_numpy(dtype="<u8")
    return sha256(descriptor+hashes.tobytes()).hexdigest(),sha256(descriptor).hexdigest()


@dataclass
class PipelineAudit:
    records: list[dict] = field(default_factory=list)
    source_records: list[dict] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    censoring: list[dict] = field(default_factory=list)
    column_records: list[dict] = field(default_factory=list)
    dataset_sha256: str | None = None
    _previous: dict = field(default_factory=dict, repr=False)
    _previous_columns: list = field(default_factory=list,repr=False)

    def capture(self, stage: str, frame: pd.DataFrame, note: str = "", *,
                exclusion_reasons=None, allowed_added_columns=(), allowed_removed_columns=()) -> None:
        if not frame.index.is_unique or not frame.columns.is_unique or not frame.index.astype(str).is_unique:
            raise ValueError(f"Stage {stage}: row/column identities are not unique")
        if any(row["stage"]==stage for row in self.records):
            raise ValueError(f"Duplicate preparation stage {stage}")
        sources=frame["source_dataset"].astype("string").fillna("unknown") if "source_dataset" in frame else pd.Series("unknown",index=frame.index)
        ids=frame["source_sample"] if "source_sample" in frame else pd.Series(frame.index,index=frame.index)
        current={index:(_identity_value(source),_identity_value(sample)) for index,source,sample in zip(frame.index,sources,ids,strict=True)}
        columns=list(frame.columns)
        added=sorted(set(columns)-set(self._previous_columns),key=str)
        removed=sorted(set(self._previous_columns)-set(columns),key=str)
        declared={} if exclusion_reasons is None else {str(r["row_index"]):r for r in exclusion_reasons}
        if exclusion_reasons is not None and len(declared)!=len(exclusion_reasons):
            raise ValueError(f"Stage {stage}: duplicate exclusion reasons")
        if self.records:
            if any(idx not in self._previous for idx in current):
                raise ValueError(f"Stage {stage}: introduced unknown rows")
            if list(current)!=[idx for idx in self._previous if idx in current]:
                raise ValueError(f"Stage {stage}: reordered retained rows")
            if any(value!=self._previous[idx] for idx,value in current.items()):
                raise ValueError(f"Stage {stage}: changed source observation identity")
            if set(added)-set(allowed_added_columns) or set(removed)-set(allowed_removed_columns):
                raise ValueError(f"Stage {stage}: undeclared column change")
            if [c for c in columns if c in self._previous_columns]!=[c for c in self._previous_columns if c in columns]:
                raise ValueError(f"Stage {stage}: reordered existing columns")
            excluded=[idx for idx in self._previous if idx not in current]
            if set(map(str,excluded))!=set(declared):
                raise ValueError(f"Stage {stage}: row exclusions lack exact declared reasons")
            for idx in excluded:
                source,sample=self._previous[idx]
                item=declared[str(idx)]
                if not str(item.get("reason","")).strip():
                    raise ValueError(f"Stage {stage}: empty exclusion reason")
                self.exclusions.append({"stage":stage,"row_index":str(idx),"source":source,"sample_id":sample,
                    "source_row_id":stable_source_row_id(self.dataset_sha256,idx) if self.dataset_sha256 else None,
                    "column":item.get("column"),"original":item.get("original"),"reason":item["reason"]})
        elif declared:
            raise ValueError("Initial stage cannot declare exclusions")
        digest,schema_hash=_digest(frame)
        record={"stage":stage,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "rows":int(frame.shape[0]),"columns":int(frame.shape[1]),
            "missing_cells":int(frame.isna().sum().sum()),"note":note,
            "source_dataset_sha256":self.dataset_sha256,"frame_digest_sha256":digest,"schema_sha256":schema_hash,
            "digest_method":f"pandas_hash_rows_{pd.__version__}_plus_schema",
            "added_columns":json.dumps(added,ensure_ascii=False),"removed_columns":json.dumps(removed,ensure_ascii=False),
            "rows_excluded":len(declared)}
        self.records.append(record)
        for source,group in frame.groupby(sources,sort=True):
            missing=group.isna().sum()
            self.source_records.append({"stage":stage,"source":str(source),"rows":len(group),
                                        "missing_cells":int(missing.sum())})
            self.column_records.extend({"stage":stage,"source":str(source),"column":str(name),
                                        "rows":len(group),"missing_cells":int(count)}
                                       for name,count in missing.items())
        self._previous=current
        self._previous_columns=columns
        logger.info("STEP %-28s rows=%d cols=%d missing=%d %s",stage,record["rows"],record["columns"],record["missing_cells"],note)
        print(f"    ✓ {stage:<28} filas={record['rows']:<5} columnas={record['columns']:<4} faltantes={record['missing_cells']}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def exclusions_frame(self):
        return pd.DataFrame(self.exclusions,columns=["stage","row_index","source","sample_id","source_row_id","column","original","reason"])
