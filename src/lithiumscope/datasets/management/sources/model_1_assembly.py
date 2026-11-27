"""Merge harmonized inputs and persist the source-merge evidence."""
import json
from uuid import uuid4
from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.provenance.publication import publish_pair
import pandas as pd

def _source_name(frame: pd.DataFrame, fallback: str) -> pd.DataFrame:
    result = frame.copy()
    if "source_dataset" not in result.columns:
        result["source_dataset"] = fallback
    return result


def assemble_sources(base_dataset, ingestion, merge_path, audit_path, georoc, *, load_data, merge_harmonized_sources, cache_identity=None):
    base_digest = file_sha256(base_dataset)
    external_digest = file_sha256(ingestion.path)
    base = _source_name(
        load_data(base_dataset, quiet=True),
        "Mamani09 bootstrap",
    )
    external = load_data(
        ingestion.path,
        quiet=True,
    )
    merged, merge_audit = merge_harmonized_sources(
        [base, external]
    )

    merge_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if file_sha256(base_dataset) != base_digest or file_sha256(ingestion.path) != external_digest:
        raise ValueError("Source changed during assembly")
    if merged.empty:
        raise ValueError("Combined dataset is empty")
    if len({p.resolve() for p in (base_dataset, ingestion.path, merge_path, audit_path)}) != 4:
        raise ValueError("Source and assembly output paths must be distinct")
    audit = {
        "cache_identity": cache_identity,
        "base_dataset": str(base_dataset),
        "base_sha256": base_digest,
        "harmonized_sha256": external_digest,
        "georoc_files": list(ingestion.files),
        "georoc_ingestion": {
            "rows_read": ingestion.rows_read,
            "rows_kept": ingestion.rows_kept,
            "audit_path": str(ingestion.audit_path),
            "iron_total_mode": str(georoc.get("iron_total_mode", "original")),
        },
        "merge": merge_audit,
        "combined_path": str(merge_path),
    }
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data_stage = merge_path.with_name(f".{merge_path.name}.{uuid4().hex}.tmp")
    audit_stage = audit_path.with_name(f".{audit_path.name}.{uuid4().hex}.tmp")
    try:
        merged.to_csv(data_stage, index=False)
        audit["combined_sha256"] = file_sha256(data_stage)
        audit_stage.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        publish_pair(data_stage, merge_path, audit_stage, audit_path)
    finally:
        data_stage.unlink(missing_ok=True)
        audit_stage.unlink(missing_ok=True)

    return merged, merge_audit
