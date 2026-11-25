"""Read-only, bounded regression integrity checks on critical local inputs."""
from __future__ import annotations

from pathlib import Path

from lithiumscope.tools.inspection.development.inventory import file_record, local_path


CRITICAL = (
    "models/active_models.json",
    "data/processed/model_1/training_combined.csv",
    "data/processed/model_2/training_manifest.csv",
    "data/processed/model_2/relocation.json",
    "config/model_1_frozen_spatial_folds_706.json",
    "config/model_2_frozen_spatial_folds_706.json",
    "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/training_v4_170.csv",
    "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/build_status.json",
    "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/rededge_quality.csv",
)


def protected_inventory(root: Path) -> dict:
    paths = set(CRITICAL)
    # All known versioned metadata/status records and model payloads are
    # checked without deserializing them. No raw imagery scan or repair.
    for group in ("model_1", "model_2"):
        for name in ("metadata.json", "candidate_status.json", "model.joblib"):
            base = root / "models" / group / "trained"
            for path in base.glob(f"*/{name}"):
                try:
                    paths.add(local_path(root, path).relative_to(root.resolve()).as_posix())
                except (OSError, ValueError):
                    paths.add(path.relative_to(root).as_posix())
    return {name: file_record(root, name) for name in sorted(paths)}


def compare_protected(before: dict, after: dict) -> dict:
    changes, unavailable = [], []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name, {"state": "MISSING"}), after.get(name, {"state": "MISSING"})
        if old.get("state") in {"ERROR", "BLOCKED"} or new.get("state") in {"ERROR", "BLOCKED"}:
            unavailable.append(name)
        elif (old.get("state"), old.get("sha256")) != (new.get("state"), new.get("sha256")):
            changes.append(name)
    return {"state": "ERROR" if changes else "WARNING" if unavailable else "OK",
            "checked_paths": len(set(before) | set(after)), "changed_paths": changes,
            "unverified_paths": unavailable, "scope": "selected scientific inputs and versioned model artifacts",
            "restoration_attempted": False, "raw_patches_checked": False}
