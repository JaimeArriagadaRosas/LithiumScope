"""Delete ONLY abandoned standalone M2-v4 fits/checkpoints; preserve input data.

By default performs a read-only preview. Explicit --execute is required.
No menu training is launched by this script.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from lithiumscope.core.paths import PROJECT_ROOT

RELATIVE_BASE = Path("results/scientific_simulations/m2_rededge_v4_20261113")
TRAINING_ARTIFACTS = (
    "comparison_706",
    "comparison_783",
    "nested_170_706",
    "data_ablation_170_new_sites",
    "candidate_bundle_not_published",
)
NEVER_DELETE = (
    "cohort_706",
    "cohort_783",
    "patches",
    "stac_cache",
    "new_sites",
    "integration_m1nov_m2sep",
)
LOG = Path("docs/models/M2_V4_STANDALONE_RESET_AUDIT_20261113.json")


def reset(*, root: Path = PROJECT_ROOT, execute: bool = False) -> dict:
    root = Path(root).resolve()
    base = (root / RELATIVE_BASE).resolve()
    if not base.is_dir() or not base.is_relative_to(root):
        raise ValueError("M2-v4 experimental results directory is not valid")
    removed = []
    for name in TRAINING_ARTIFACTS:
        target = base / name
        if not target.exists():
            continue
        if target.is_symlink() or target.resolve() != base / name:
            raise ValueError(f"Unexpected training directory link: {target}")
        if not target.is_dir():
            raise ValueError(f"Expected a training output directory: {target}")
        files = [p for p in target.rglob("*") if p.is_file()]
        removed.append({
            "relative_path": str(target.relative_to(root)),
            "files": len(files),
            "bytes": int(sum(f.stat().st_size for f in files)),
        })
    protected = {
        name: (base / name).is_dir() for name in NEVER_DELETE
    }
    if not all(protected.values()):
        raise ValueError("Refusing reset: a protected v4 input directory is missing")
    plan = {
        "operation": "remove_standalone_v4_training_only",
        "executed": bool(execute),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cleanup_scope": list(TRAINING_ARTIFACTS),
        "training_directories_found": removed,
        "protected_input_directories_before": protected,
        "historical_706_and_expanded_783_manifests_unchanged": True,
        "models_active_registry_untouched": True,
        "interactive_menu_not_executed": True,
    }
    if not execute:
        print("M2_V4_RESET_PREVIEW", json.dumps(plan, ensure_ascii=False), flush=True)
        return plan
    for entry in removed:
        shutil.rmtree(root / entry["relative_path"])
    for name in TRAINING_ARTIFACTS:
        if (base / name).exists():
            raise RuntimeError("M2 v4 stale training output still exists: " + name)
    if not all((base / name).is_dir() for name in NEVER_DELETE):
        raise RuntimeError("M2 v4 source data was affected by the reset")
    dest = root / LOG
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("M2_V4_STANDALONE_TRAINING_CLEANED", json.dumps(plan), flush=True)
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    reset(execute=args.execute)


if __name__ == "__main__":
    main()
