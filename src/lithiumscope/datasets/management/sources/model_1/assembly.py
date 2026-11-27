"""Merge harmonized inputs and persist the source-merge evidence."""
import json
import pandas as pd

def _source_name(frame: pd.DataFrame, fallback: str) -> pd.DataFrame:
    result = frame.copy()
    if "source_dataset" not in result.columns:
        result["source_dataset"] = fallback
    return result


def assemble_sources(base_dataset, ingestion, merge_path, audit_path, georoc, *, load_data, merge_harmonized_sources):
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
    merged.to_csv(
        merge_path,
        index=False,
    )

    audit = {
        "base_dataset": str(base_dataset),
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
    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return merged, merge_audit
