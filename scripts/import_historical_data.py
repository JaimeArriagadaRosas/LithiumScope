"""Copy verified historical datasets; never rewrite the source or frozen manifest."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

from lithiumscope.core.hashing import file_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("data/processed/model_2/training_manifest.csv")
INVENTORY = Path("results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv")
V4 = Path("results/scientific_simulations/m2_rededge_v4_20261113/cohort_706")
FROZEN_HASH = "5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def import_data(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Source and destination must be independent workspaces")
    if file_sha256(source / MANIFEST) != FROZEN_HASH:
        raise ValueError("Historical manifest is not the declared frozen cohort")
    manifest = rows(source / MANIFEST)
    inventory = rows(source / INVENTORY)[:706]
    quality = rows(source / V4 / "rededge_quality.csv")
    state = json.loads((source / V4 / "build_status.json").read_text(encoding="utf-8"))
    if len(manifest) != 706 or len(inventory) != 706 or len(quality) != 706:
        raise ValueError("Historical cohort counts differ")
    if state.get("manifest_sha256") != FROZEN_HASH or file_sha256(source / V4 / "training_v4_170.csv") != state.get("matrix_sha256"):
        raise ValueError("Historical spectral v4 matrix identity mismatch")
    sources = {}
    images = {}
    def register(relative, expected=None):
        relative = Path(relative)
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Source file unavailable or outside historical root: {relative}")
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Source hash mismatch: {relative}")
        sources[relative.as_posix()] = digest
        return digest

    for index, (row, qa, red) in enumerate(zip(manifest, inventory, quality)):
        if row["sample_id"] != qa["sample_id"] or row["sample_id"] != red["sample_id"] or int(qa["manifest_index"]) != index:
            raise ValueError(f"Historical sample order mismatch at {index}")
        if qa["image_sha256"] != red["original_patch_sha256"]:
            raise ValueError(f"Original patch audit mismatch at {index}")
        original = Path(row["image_path"])
        relative = original.resolve().relative_to(source)
        if not relative.is_relative_to(Path("data/raw/model_2")):
            raise ValueError("Historical patch outside M2 raw data")
        digest = register(relative, qa["image_sha256"])
        images[str(original)] = {"relative_path": relative.as_posix(), "sha256": digest}
        for sidecar in (original.with_suffix(".json"), original.with_suffix(original.suffix + ".json")):
            if sidecar.is_file():
                register(sidecar.resolve().relative_to(source))
    for folder in ("data/raw/model_1", "data/interim/model_1", "data/processed/model_1"):
        for path in sorted((source / folder).rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                register(path.relative_to(source))
    for relative in (MANIFEST, INVENTORY, *(V4 / name for name in ("build_status.json", "rededge_quality.csv", "training_v4_170.csv"))):
        register(relative)
    # Refuse overwrites of different local inputs; interrupted copies can be resumed.
    for relative, digest in sources.items():
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Destination escapes project: {relative}")
        if target.exists() and file_sha256(target) != digest:
            raise FileExistsError(f"Different local artifact: {relative}")
    copied = 0
    for relative, digest in sources.items():
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            try:
                shutil.copyfile(source / relative, tmp)
                if file_sha256(tmp) != digest:
                    raise ValueError(f"Copy hash mismatch: {relative}")
                tmp.replace(target)
                copied += 1
            finally:
                tmp.unlink(missing_ok=True)
    registry = {"schema_version": 1, "source_root": str(source), "manifest_sha256": FROZEN_HASH,
                "images": images}
    atomic_json(destination / "data/processed/model_2/relocation.json", registry)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_root": str(source), "destination_root": str(destination),
              "files_verified": len(sources), "files_copied": copied, "manifest_rows": len(manifest),
              "unique_patch_paths": len(images), "files": sources,
              "scope": "Frozen M2 original patches and v4 feature evidence; M1 source snapshots; no models"}
    atomic_json(destination / "results/diagnostics/historical_import.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    record = import_data(args.source, ROOT)
    print(json.dumps({key: val for key, val in record.items() if key != "files"}, indent=2))
