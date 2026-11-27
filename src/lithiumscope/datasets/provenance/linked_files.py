"""Ordered content digests for files linked from a dataset manifest."""
from pathlib import Path
from typing import Any
import pandas as pd
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path

def linked_file_metadata(path: Path, frame: pd.DataFrame | None, model_group: str) -> dict[str, Any]:
    # M2 preparation supplies a feature matrix; the raw source manifest has paths.
    if frame is not None and "image_path" in frame:
        source = frame
    elif model_group == "model_2" and path.suffix.lower() == ".csv":
        source = pd.read_csv(path)
    else:
        return {}
    if "image_path" not in source:
        return {}
    records, cache = [], {}
    for position, raw in enumerate(source["image_path"]):
        digest = None
        if pd.notna(raw) and str(raw).strip():
            try:
                resolved = resolve_image_path(str(raw))
                if resolved not in cache:
                    cache[resolved] = file_sha256(resolved) if resolved.is_file() and resolved.stat().st_size else None
                digest = cache[resolved]
            except (OSError, ValueError):
                pass
        records.append({"position": position, "sha256": digest})
    missing = sum(r["sha256"] is None for r in records)
    return {"linked_file_count": len(records), "linked_file_missing_count": missing,
            "linked_files_complete": bool(records) and missing == 0,
            "linked_hash_policy": "ordered_resolved_content_v2",
            "linked_files_sha256": canonical_json_hash({"rows": records})}


