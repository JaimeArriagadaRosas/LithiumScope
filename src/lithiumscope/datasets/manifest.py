from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lithiumscope.core.hashing import file_set_sha256, file_sha256
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.core.states import DatasetState


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    dataset_name: str
    model_group: str
    stage: str
    source_path: str
    source_sha256: str
    state: str
    rows: int | None
    columns: int | None
    column_names: list[str]
    created_at_utc: str
    runtime: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _linked_file_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".csv":
        return {}
    try:
        source = pd.read_csv(path)
    except Exception:
        return {}
    if "image_path" not in source.columns:
        return {}

    linked = [Path(str(value)) for value in source["image_path"].dropna().tolist()]
    return {
        "linked_file_count": len(linked),
        "linked_files_sha256": file_set_sha256(linked),
    }


def build_tabular_manifest(
    path: Path,
    dataset_name: str,
    model_group: str,
    stage: str,
    frame: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    if frame is None:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path, encoding_errors="replace")
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)

    merged_metadata = {
        **_linked_file_metadata(path),
        **(metadata or {}),
    }

    return DatasetManifest(
        schema_version=1,
        dataset_name=dataset_name,
        model_group=model_group,
        stage=stage,
        source_path=str(path),
        source_sha256=file_sha256(path),
        state=DatasetState.READY.value,
        rows=int(frame.shape[0]) if frame is not None else None,
        columns=int(frame.shape[1]) if frame is not None else None,
        column_names=[str(column) for column in frame.columns] if frame is not None else [],
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        runtime=runtime_fingerprint(),
        metadata=merged_metadata,
    )


def write_manifest(manifest: DatasetManifest, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
